const std = @import("std");

pub fn main() !void {
    try test_double_streaming_shuffle();
}

fn test_double_streaming_shuffle() !void {
    var file1: std.fs.File = try std.fs.cwd().openFile("newIoTests/test.txt", .{ .mode = .read_only });
    var buffer1: [1024]u8 = undefined;
    var file_reader1 = file1.reader(&buffer1);

    try file_reader1.seekTo(0);

    var file2: std.fs.File = try std.fs.cwd().openFile("newIoTests/test.txt", .{ .mode = .read_only });
    var buffer2: [1024]u8 = undefined;
    var file_reader2 = file2.reader(&buffer2);

    try file_reader2.seekTo(36);

    var r_buf: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&r_buf);

    for (0..50) |_| {
        try file_reader1.interface.streamExact(&stdout_writer.interface, 1);
        try file_reader2.interface.streamExact(&stdout_writer.interface, 1);
    }

    try stdout_writer.interface.flush();
}
