# Experiment with mxfp4 format tensors

# Example of mxfp4 files:
# - tensors of gpt-oss, see https://huggingface.co/openai/gpt-oss-20b 
# - smaller model: https://huggingface.co/huyhoangt2201/pruned_first_1?show_file_info=model.safetensors

# Support of MXFP4 in safetensors was added in https://github.com/huggingface/safetensors/pull/611/commits/4996e202d40b7f6c00d86ac9aeec324fa20a75b7
# Data type for MXFP4: blocks of values in `F4`, with block scale factors in `E8M0`

# However in gpt-oss files, the tensors appear sometimes with some `_blocks` and some `_scales` suffixes, both in U8 format. Other tensors are in BF16 format.