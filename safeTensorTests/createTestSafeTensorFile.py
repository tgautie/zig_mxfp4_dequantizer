import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.ones((2, 2)),
    "attention": torch.ones((2, 3))
}
save_file(tensors, "ones.safetensors")
