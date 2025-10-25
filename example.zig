const std = @import("std");
const tensorReaders = @import("tensorReaders.zig");

// Insert your safetensors file path here 🙌
const file_path = "testSafetensors/simple_test.safetensors";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var dequantizedSafetensors = try tensorReaders.DequantizedMxfp4TensorReaders.init(allocator, file_path);
    defer dequantizedSafetensors.deinit(allocator);

    std.debug.print("Dequantizing on the fly the following MXFP4 tensors found in the provided file:\n", .{});

    for (dequantizedSafetensors.readers.items) |reader| {
        std.debug.print("{s}\n", .{reader.name});
        const buffer = try reader.interface.takeArray(100);
        for (buffer) |b| {
            std.debug.print("{x} ", .{b});
        }
        std.debug.print("\n", .{});
    }
}
