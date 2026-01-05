import torch
torch._dynamo.config.cache_size_limit = max(8192, getattr(torch._dynamo.config, "cache_size_limit", 0))
torch._dynamo.config.accumulated_recompile_limit = max(8192, getattr(torch._dynamo.config, "accumulated_recompile_limit", 0))
from .quantize import quantize, dequantize

class Lora(torch.nn.Module):
    def __init__(self, name, lora_up, lora_down, alpha, strength=1.0):
        super().__init__()
        self.name = name
        self.lora_up = lora_up
        self.lora_down = lora_down
        self.alpha = alpha
        self.rank = lora_up.shape[1]
        self.scale = torch.tensor([strength * alpha / self.rank], device="cuda", dtype=lora_up.dtype)
        self.strength = strength

    def get_weight(self):
        return self.scale * self.lora_up @ self.lora_down

class HQQSVDLinear(torch.nn.Module):
    def __init__(
        self, W_q, svd_up, svd_down, scale, zero_point, bias, nbits, int8_matmul:bool=True
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
        self.nbits = torch.nn.Parameter(torch.tensor([nbits]), False) # for serialization
        self._nbits = nbits
        self.int8_matmul = int8_matmul
        self.loras = {}

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        svd_rank: int = 128,
        svd_steps: int = 8,
        group_size: int = 128,
        nbits: int = 4,
    ):
        W_q, svd_up, svd_down, scale, zero_point = quantize(
            linear.weight, svd_rank, svd_steps, group_size, nbits
        )
        return cls(
            W_q, svd_up, svd_down, scale, zero_point, linear.bias, nbits
        )
    
    def add_lora(self, lora):
        self.loras[lora.name] = lora
    
    def remove_lora(self, name):
        self.loras.pop(name)
    
    def dequantize(self):
        W_f = dequantize(
            self.weight,
            self.svd_up,
            self.svd_down,
            self.scale,
            self.zero_point,
            self.q_shape,
            self.o_shape,
            self._nbits
        )
        for lora in self.loras.values():
            W_f += lora.get_weight()
        return W_f
    
    def forward_int8(self, x:torch.FloatTensor):
        x = x.view((-1, x.shape[-1]))
        dtype = x.dtype
        W_f = self.dequantize().T

        scale_x = torch.amax(x.abs(), dim=1, keepdims=True).div_(127)
        x_q = torch.div(x, scale_x).round_().clamp_(-128, 127).to(dtype=torch.int8)

        scale_w = torch.amax(W_f.abs(), dim=0, keepdims=True).div_(127)
        W_q = torch.div(W_f, scale_w).round_().clamp_(-128, 127).to(dtype=torch.int8)
        
        return (torch._int_mm(x_q, W_q).to(dtype) * scale_x * scale_w).unsqueeze(0) + self.bias

    @torch.compile(fullgraph=True)
    def forward(self, x:torch.FloatTensor):
        if self.int8_matmul and x.numel() / x.shape[-1] >= 16:
            return self.forward_int8(x)
        return torch.nn.functional.linear(x, self.dequantize(), self.bias)
