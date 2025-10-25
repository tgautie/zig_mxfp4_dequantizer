const std = @import("std");
const mxfp4TensorReader = @import("mxfp4TensorReader.zig");
const mxfp4Config = @import("mxfp4Config.zig");
const safetensors = @import("safetensors.zig");

// Insert your safetensors file path here 🙌
const file_path = "exampleSafetensors/test_mxfp4.safetensors";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tensor_configs = try safetensors.parseHeader(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    var mxfp4_tensor_configs = try mxfp4Config.extractMxfp4TensorConfigs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    var readers: std.ArrayList(mxfp4TensorReader.DequantizedMxfp4TensorReader) = .empty;
    errdefer readers.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        const buffer = try allocator.alloc(u8, 4096);
        errdefer allocator.free(buffer);

        const reader = try mxfp4TensorReader.DequantizedMxfp4TensorReader.init(buffer, file_path, mxfp4_tensor_config);
        try readers.append(allocator, reader);
    }

    for (readers.items) |reader| {
        std.debug.print("Reader: {any}\n", .{reader.total_blocks_count});
    }
}
