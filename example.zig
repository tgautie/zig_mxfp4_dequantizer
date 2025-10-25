const std = @import("std");
const tensorReaders = @import("tensorReaders.zig");

const file_path = "exampleSafetensors/test_mxfp4.safetensors";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var readers = try tensorReaders.DequantizedMxfp4TensorReaders.init(allocator, file_path);
    defer readers.deinit(allocator);

    for (readers.readers.items) |reader| {
        std.debug.print("Peek byte in the two readers: scale {any}, block {any}\n", .{ reader.scales_reader.interface.peekByte(), reader.blocks_reader.interface.peekByte() });
        const buffer = try reader.interface.takeArray(100);
        for (buffer) |b| {
            std.debug.print("{x} ", .{b});
        }
        std.debug.print("\n", .{});
    }
}
