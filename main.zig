const std = @import("std");
const reader = @import("reader.zig");
const mxfp4Config = @import("mxfp4Config.zig");
const safetensors = @import("safetensors.zig");

// Insert your file path here
const file_path = "exampleSafetensors/test_mxfp4.safetensors";

pub fn main() void {
    const allocator = std.heap.page_allocator;

    const tensor_configs = try safetensors.parseHeader(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    const mxfp4_tensor_configs = try mxfp4Config.extractMxfp4TensorConfigs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        std.debug.print("MXFP4 tensor config: {any}\n", .{mxfp4_tensor_config});
    }
}
