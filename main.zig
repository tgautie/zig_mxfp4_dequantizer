const std = @import("std");
const reader = @import("reader.zig");
const mxfp4Config = @import("mxfp4Config.zig");
const safetensors = @import("safetensors.zig");

// Insert your safetensors file path here 🙌
const file_path = "exampleSafetensors/test_mxfp4.safetensors";

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var tensor_configs = try safetensors.parseHeader(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    var mxfp4_tensor_configs = try mxfp4Config.extractMxfp4TensorConfigs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        std.debug.print("MXFP4 tensor config: name {s}\nblocks_count {d}, scales_dtype {s}, blocks_dtype {s}, scales_shape {any}, blocks_shape {any}, scales_absolute_offsets {any}, blocks_absolute_offsets {any}\n", .{ mxfp4_tensor_config.mxfp4_tensor_name, mxfp4_tensor_config.blocks_count, mxfp4_tensor_config.scales_dtype, mxfp4_tensor_config.blocks_dtype, mxfp4_tensor_config.scales_shape, mxfp4_tensor_config.blocks_shape, mxfp4_tensor_config.scales_absolute_offsets, mxfp4_tensor_config.blocks_absolute_offsets });
    }
}
