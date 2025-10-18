const std = @import("std");

pub fn main() !void {
    try test_reader_discarding();
}

fn test_reader_with_buffer() !void {
    var stdin_buffer: [1024]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);

    while (stdin_reader.interface.takeByte()) |char| {
        if (char == '\n') continue;
        std.debug.print("Typed char: {c}\n", .{char});
        if (char == 'q') break;
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

// This panics at runtime, because the reader is not buffered
fn test_reader_without_buffer() !void {
    var stdin_reader = std.fs.File.stdin().reader(&.{});

    while (stdin_reader.interface.take(@as(usize, 1))) |char| {
        std.debug.print("Typed char: {x}\n", .{char});
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

fn test_reader_with_takeArray() !void {
    var stdin_buffer: [1024]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);

    while (stdin_reader.interface.takeArray(5)) |str| {
        std.debug.print("Typed: {s}\n", .{str});
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

// Panics at runtime
fn test_reader_taking_too_many_bytes() !void {
    var stdin_buffer: [10]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);

    while (stdin_reader.interface.takeArray(20)) |str| {
        std.debug.print("Typed: {s}\n", .{str});
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

fn test_reader_takeDelimiterExclusive() !void {
    var stdin_buffer: [10]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);

    while (stdin_reader.interface.takeDelimiterExclusive('-')) |str| {
        std.debug.print("Typed: {s}\n", .{str});
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

fn test_reader_allocating() !void {
    var stdin_buffer: [10]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);

    var alloc = std.heap.DebugAllocator(.{}).init;
    defer _ = alloc.deinit();
    const da = alloc.allocator();

    var allocating_writer = std.Io.Writer.Allocating.init(da);
    defer allocating_writer.deinit();

    while (stdin_reader.interface.streamDelimiter(&allocating_writer.writer, '-')) |_| {
        const line = allocating_writer.written();
        std.debug.print("Typed: {s}\n", .{line});
        allocating_writer.clearRetainingCapacity();
        stdin_reader.interface.toss(1);
    } else |err| {
        std.debug.print("Error reading byte: {}\n", .{err});
    }
}

fn test_reader_discarding() !void {
    var file: std.fs.File = try std.fs.cwd().openFile("newReaderTests/test.txt", .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    var r_buf: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&r_buf);

    var discard_writer = std.Io.Writer.Discarding.init(&.{});

    while (file_reader.interface.peekByte()) |char| switch (char) {
        '?' => try file_reader.interface.streamExact(&discard_writer.writer, 1),
        else => try file_reader.interface.streamExact(&stdout_writer.interface, 1),
    } else |err| switch (err) {
        error.EndOfStream => {
            try stdout_writer.interface.flush();
            std.debug.print("{d} bytes discarded", .{discard_writer.fullCount()});
        },
        else => std.debug.print("An error occured: {any}", .{err}),
    }
}
