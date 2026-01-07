from typing import Tuple
import torch
from .optimize import optimize_weights
from .bitpack import pack, unpack

def apply_svdquant(weight: torch.FloatTensor, rank: int = 32, niter: int = 8) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2: # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1,-1)
    weight = weight.to(dtype=torch.float32)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank, niter=niter)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    weight = weight.sub(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(-1, (*weight_shape[1:],)) # pylint: disable=possibly-used-before-assignment
    return weight, svd_up, svd_down


def quantize(W, svd_rank:int=128, svd_steps:int=8, group_size:int=128, nbits:int=4, fast=True):
    dtype = W.dtype
    shape = W.shape

    W, svd_up, svd_down = apply_svdquant(W, rank=svd_rank, niter=svd_steps)

    W = W.reshape([-1, group_size])

    _min = W.min(axis=1, keepdim=True)[0]
    _max = W.max(axis=1, keepdim=True)[0]
    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]

    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    denom = (_max - _min)
    scale = (max_v / denom)  
    scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
    zero = -_min * scale

    if fast:
        W_q = (W * scale + zero).round_().clamp_(min_max[0], min_max[1])
    else:
        W_q, scale, zero = optimize_weights(W, scale, zero, min_max, 1)

    W_q = W_q.reshape((shape[0], -1, group_size))
    W_q = torch.clamp(W_q, min_v, max_v).to(torch.uint8)
    W_q = pack(W_q, nbits)
    scale = 1.0/scale.reshape((shape[0], -1, 1))
    zero = -zero.reshape((shape[0], -1, 1)) * scale
    return W_q, svd_up.to(dtype), svd_down.to(dtype), scale.to(dtype), zero.to(dtype)


def dequantize(W_q, svd_up, svd_down, scale, zero, q_shape, o_shape, nbits:int):
    W_f = unpack(W_q, q_shape, nbits).to(dtype=scale.dtype)
    W_f = torch.addcmul(zero, W_f, scale).view(o_shape)
    W_f.addmm_(svd_up, svd_down)
    return W_f
