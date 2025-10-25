const std = @import("std");

pub fn main() !void {
    const file = try std.fs.cwd().openFile("exampleSafetensors/only_zeros.safetensors", .{});
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);
    try file_reader.seekTo(176);

    const output = try file_reader.interface.takeArray(336 - 176);
    std.debug.print("Block content: {any}\n", .{output});

    const scales_output = try file_reader.interface.takeArray(346 - 336);
    std.debug.print("Scales content: {any}\n", .{scales_output});
}
