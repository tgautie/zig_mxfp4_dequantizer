import torch
from safetensors.torch import save_file;

N = 10

tensors = {
    "tensor_with_zeros_scales": torch.zeros((N), dtype=torch.uint8),
    "tensor_with_zeros_blocks": torch.zeros((N, 16), dtype=torch.uint8),
}
save_file(tensors, "only_zeros.safetensors")
