const std = @import("std");

pub fn main() !void {
    try test_writer_with_buffer();
}

fn test_writer_without_buffer() !void {
    var stdout_writer: std.fs.File.Writer = std.fs.File.stdout().writer(&.{});
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    for (1..1001) |i| {
        try stdout.print("{d}. Hello \n", .{i});
    }
    // No flush needed
}

fn test_writer_with_buffer() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer: std.fs.File.Writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    for (1..1001) |i| {
        try stdout.print("{:0>3}. Hello \n", .{i});
    }

    // std.debug.print("\nBuffer content: \n\n{s}", .{stdout_buffer});

    // Flush remaining elements, if commented out, the prints will stop at line 946
    // try stdout.flush();

    // std.debug.print("\nBuffer content: \n\n{s}", .{stdout_buffer});
}

// Estimating the number of lines printed before last flush:
// 1 line = 13 bytes
// 1024 bytes fits 78.77 lines
// 1000 lines / 79 = 12.7 buffers needed
// After 12 buffers, we have 12 * 78.77 = 946 lines printed
// ✅ it checks out!
