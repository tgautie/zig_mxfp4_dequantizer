import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.ones((2, 2)),
    "attention": torch.ones((2, 3)),
    "zeros": torch.zeros((2, 3, 4, 5))
}
save_file(tensors, "mixed.safetensors")
