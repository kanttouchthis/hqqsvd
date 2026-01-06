import torch

TORCH_INT_MM = False
TRITON_INT_MM = False
try:
    a = torch.zeros((32, 32), device="cuda", dtype=torch.int8)
    b = torch.zeros((32, 32), device="cuda", dtype=torch.int8)
    torch._int_mm(a, b)
    TORCH_INT_MM = True
except:
    pass
try:
    from .triton_mm import scaled_int8_matmul

    TRITON_INT_MM = True
except:
    pass

torch._dynamo.config.cache_size_limit = max(
    8192, getattr(torch._dynamo.config, "cache_size_limit", 0)
)
torch._dynamo.config.accumulated_recompile_limit = max(
    8192, getattr(torch._dynamo.config, "accumulated_recompile_limit", 0)
)
from .quantize import quantize, dequantize


class Lora(torch.nn.Module):
    def __init__(self, name, lora_up, lora_down, alpha, strength=1.0):
        super().__init__()
        self.name = name
        self.lora_up = lora_up
        self.lora_down = lora_down
        self.alpha = alpha if alpha is not None else 1.0
        self.rank = lora_up.shape[1]
        self.scale = torch.tensor(
            [strength * self.alpha / self.rank], device="cuda", dtype=lora_up.dtype
        )
        self.strength = strength

    def get_weight(self, weight):
        return self.scale * self.lora_up @ self.lora_down


class ComfyLora:
    def __init__(self, name, comfy_lora, calculate_weight):
        self.comfy_lora = comfy_lora
        self.name = name
        self.calculate_weight = calculate_weight

    @torch._dynamo.disable
    def get_weight(self, weight):
        return (
            self.calculate_weight(
                [
                    self.comfy_lora,
                ],
                weight,
                self.name.split("|")[0],
            )
            - weight
        )


class HQQSVDLinear(torch.nn.Module):
    def __init__(
        self,
        W_q,
        svd_up,
        svd_down,
        scale,
        zero_point,
        bias,
        nbits,
        int8_matmul: bool = True,
    ):
        super().__init__()
        self.in_features = svd_down.shape[1]
        self.out_features = svd_up.shape[0]
        self.svd_rank = svd_down.shape[0]
        self.group_size = self.in_features // scale.shape[1]
        self.n_groups = scale.shape[1]
        self.q_shape = torch.Size((self.out_features, self.n_groups, self.group_size))
        self.o_shape = torch.Size((self.out_features, self.in_features))
        self.weight = torch.nn.Parameter(W_q, False)
        self.svd_up = torch.nn.Parameter(svd_up, False)
        self.svd_down = torch.nn.Parameter(svd_down, False)
        self.scale = torch.nn.Parameter(scale, False)
        self.zero_point = torch.nn.Parameter(zero_point, False)
        self.bias = torch.nn.Parameter(bias, False)
        self.nbits = torch.nn.Parameter(
            torch.tensor([nbits]), False
        )  # for serialization
        self._nbits = nbits
        self.int8_matmul = int8_matmul
        self.loras = {}
        self.forward_no_comfy = torch.compile(self._forward, fullgraph=True)
        self.forward_comfy = torch.compile(self._forward)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        svd_rank: int = 128,
        svd_steps: int = 8,
        group_size: int = 128,
        nbits: int = 4,
        fast:bool=True
    ):
        return cls.from_weights(linear.weight, svd_rank, svd_steps, group_size, nbits, fast, linear.bias)

    @classmethod
    def from_weights(
        cls,
        weight,
        svd_rank: int = 128,
        svd_steps: int = 8,
        group_size: int = 128,
        nbits: int = 4,
        fast:bool=True,
        bias=None
    ):
        if bias is None:
            # currently no bias is not supported, so we use bias = 0 instead
            bias = torch.zeros(
                (weight.shape[0],), dtype=weight.dtype, device=weight.device
            )
        return cls(
            *quantize(weight, svd_rank, svd_steps, group_size, nbits, fast), bias, nbits
        )

    def add_lora(self, lora):
        self.loras[lora.name] = lora

    def remove_lora(self, name):
        self.loras.pop(name)

    def dequantize(self, apply_lora=False):
        W_f = dequantize(
            self.weight,
            self.svd_up,
            self.svd_down,
            self.scale,
            self.zero_point,
            self.q_shape,
            self.o_shape,
            self._nbits,
        )
        if apply_lora:
            for lora in self.loras.values():
                W_f += lora.get_weight(W_f)
        return W_f

    def forward_int8_triton(self, x: torch.FloatTensor):
        original_shape = x.shape
        x = x.view((-1, x.shape[-1]))
        dtype = x.dtype
        W_f = self.dequantize(apply_lora=True).T

        scale_x = torch.amax(x.abs(), dim=1, keepdims=True).div_(127)
        x_q = torch.div(x, scale_x).round_().clamp_(-128, 127).to(dtype=torch.int8)

        scale_w = torch.amax(W_f.abs(), dim=0, keepdims=True).div_(127)
        W_q = torch.div(W_f, scale_w).round_().clamp_(-128, 127).to(dtype=torch.int8)

        output = scaled_int8_matmul(x_q, W_q, scale_x, scale_w).to(dtype)
        output = output.view(*original_shape[:-1], -1)

        return output + self.bias

    def forward_int8(self, x: torch.FloatTensor):
        original_shape = x.shape
        x = x.view((-1, x.shape[-1]))
        dtype = x.dtype
        W_f = self.dequantize(apply_lora=True).T

        scale_x = torch.amax(x.abs(), dim=1, keepdims=True).div_(127)
        x_q = torch.div(x, scale_x).round_().clamp_(-128, 127).to(dtype=torch.int8)

        scale_w = torch.amax(W_f.abs(), dim=0, keepdims=True).div_(127)
        W_q = torch.div(W_f, scale_w).round_().clamp_(-128, 127).to(dtype=torch.int8)

        output = torch._int_mm(x_q, W_q).to(dtype) * scale_x * scale_w
        output = output.view(*original_shape[:-1], -1)

        return output + self.bias

    def _forward(self, x: torch.FloatTensor):
        if self.int8_matmul and x.numel() / x.shape[-1] > 16:
            if TORCH_INT_MM:
                return self.forward_int8(x)
            if TRITON_INT_MM:
                return self.forward_int8_triton(x)
        return torch.nn.functional.linear(
            x, self.dequantize(apply_lora=True), self.bias
        )

    def forward(self, x: torch.FloatTensor):
        if any([isinstance(lora, ComfyLora) for lora in self.loras.values()]):
            return self.forward_comfy(x)
        return self.forward_no_comfy(x)
