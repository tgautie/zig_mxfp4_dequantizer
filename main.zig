const std = @import("std");
const mxfp4TensorReader = @import("mxfp4TensorReader.zig");
const mxfp4Config = @import("mxfp4Config.zig");
const safetensors = @import("safetensors.zig");

// Insert your safetensors file path here 🙌
const file_path = "exampleSafetensors/only_zeros.safetensors";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tensor_configs = try safetensors.parseHeader(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    var mxfp4_tensor_configs = try mxfp4Config.extractMxfp4TensorConfigs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        std.debug.print("MXFP4 tensor config {s}: {any}\n", .{ mxfp4_tensor_config.mxfp4_tensor_name, mxfp4_tensor_config });
    }

    var readers: std.ArrayList(mxfp4TensorReader.DequantizedMxfp4TensorReader) = .empty;
    errdefer readers.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        const buffer = try allocator.alloc(u8, 4096);
        errdefer allocator.free(buffer);

        var reader = try mxfp4TensorReader.DequantizedMxfp4TensorReader.init(buffer, file_path, mxfp4_tensor_config);
        _ = &reader;
        try readers.append(allocator, reader);
    }

    for (readers.items) |*reader| {
        std.debug.print("Reader {s}: {d} blocks to dequantize\n", .{ reader.name, reader.total_blocks_count });
        std.debug.print("Peek byte in the two readers: scale {any}, block {any}\n", .{ reader.scales_reader.interface.peekByte(), reader.blocks_reader.interface.peekByte() });
        const buffer = try reader.interface.takeArray(100);
        for (buffer) |b| {
            std.debug.print("{x} ", .{b});
        }
        std.debug.print("\n", .{});
    }
}
