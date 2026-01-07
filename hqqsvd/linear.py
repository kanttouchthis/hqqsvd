import math
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
        in_features: int,
        out_features: int,
        svd_rank: int,
        n_groups: int,
        nbits: int,
        int8_matmul: bool = True,
        bias: bool = True,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        assert in_features % n_groups == 0, "in_features must be divisible by n_groups"

        self.in_features = in_features
        self.out_features = out_features
        self.svd_rank = svd_rank
        self.n_groups = n_groups
        self.group_size = in_features // n_groups
        self._nbits = nbits

        self.q_shape = torch.Size((out_features, n_groups, self.group_size))
        self.o_shape = torch.Size((out_features, in_features))

        total_bits = out_features * in_features * nbits
        num_bytes = (total_bits + 7) // 8  # ceil div

        self.weight = torch.nn.Parameter(
            torch.empty(num_bytes, dtype=torch.uint8, device=device),
            requires_grad=False,
        )

        self.svd_up = torch.nn.Parameter(
            torch.empty((out_features, svd_rank), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.svd_down = torch.nn.Parameter(
            torch.empty((svd_rank, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )

        self.scale = torch.nn.Parameter(
            torch.empty((out_features, n_groups, 1), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.zero_point = torch.nn.Parameter(
            torch.empty((out_features, n_groups, 1), device=device, dtype=dtype),
            requires_grad=False,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.nbits = torch.nn.Parameter(
            torch.tensor([nbits], device=device),
            requires_grad=False,
        )

        self.int8_matmul = int8_matmul
        self.loras = {}

        self.forward_no_comfy = torch.compile(self._forward, fullgraph=True)
        self.forward_comfy = torch.compile(self._forward)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        svd_rank: int = 32,
        svd_steps: int = 8,
        group_size: int = 32,
        nbits: int = 4,
        fast: bool = True,
        int8_matmul: bool = True,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        return cls.from_weights(
            linear.weight,
            svd_rank,
            svd_steps,
            group_size,
            nbits,
            fast,
            linear.bias,
            int8_matmul,
            device,
            dtype,
        )

    @classmethod
    def from_weights(
        cls,
        weight,
        svd_rank: int = 32,
        svd_steps: int = 8,
        group_size: int = 32,
        nbits: int = 4,
        fast: bool = True,
        bias=None,
        int8_matmul: bool = True,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        qlinear = cls(
            weight.shape[1],
            weight.shape[0],
            svd_rank,
            weight.shape[1] // group_size,
            nbits,
            bias is not None,
            int8_matmul,
            device,
            dtype,
        )
        W_q, svd_up, svd_down, scale, zero = quantize(
            weight, svd_rank, svd_steps, group_size, nbits, fast
        )
        sd = {
            "weight": W_q,
            "svd_up": svd_up,
            "svd_down": svd_down,
            "scale": scale,
            "zero_point": zero,
        }
        if bias is not None:
            sd["bias"] = bias
        qlinear.load_state_dict(sd, strict=False)
        return qlinear

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

        return output

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

        return output

    def _forward(self, x: torch.FloatTensor):
        if self.int8_matmul and x.numel() / x.shape[-1] > 16:
            if TORCH_INT_MM:
                output = self.forward_int8(x)
            if TRITON_INT_MM:
                output = self.forward_int8_triton(x)
            if self.bias is not None:
                output += self.bias
            return output
        return torch.nn.functional.linear(
            x, self.dequantize(apply_lora=True), self.bias
        )

    def forward(self, x: torch.FloatTensor):
        if any([isinstance(lora, ComfyLora) for lora in self.loras.values()]):
            return self.forward_comfy(x)
        return self.forward_no_comfy(x)
