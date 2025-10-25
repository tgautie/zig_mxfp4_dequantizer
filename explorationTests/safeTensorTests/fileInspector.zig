const std = @import("std");

pub fn main() !void {
    const file = try std.fs.cwd().openFile("exampleSafetensors/test_mxfp4.safetensors", .{});
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);
    try file_reader.seekTo(336);

    const output = try file_reader.interface.takeArray(496 - 336);
    std.debug.print("Block content: {any}\n", .{output});
}
