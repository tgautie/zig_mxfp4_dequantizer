import torch
from safetensors.torch import save_file;

N = 1

tensors = {
    "tensor_with_zeros_scales": torch.zeros((N), dtype=torch.uint8),
    "tensor_with_zeros_blocks": torch.zeros((N, 16), dtype=torch.uint8),
    "tensor_with_ones_scales": torch.ones((N), dtype=torch.uint8),
    "tensor_with_ones_blocks": torch.ones((N, 16), dtype=torch.uint8),
}
save_file(tensors, "simple_test.safetensors")
