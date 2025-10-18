from safetensors import safe_open

tensors = {}
with safe_open("test.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
        
print(tensors.keys())
print(tensors["embedding"])
print(tensors["attention"])