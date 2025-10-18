const std = @import("std");

pub fn main() !void {
    const file = try std.fs.cwd().openFile("readerTests/test.txt", .{});
    var buffer: [1024]u8 = undefined;

    var fr = file.reader(&buffer);
    const size = try fr.getSize();
    std.debug.print("File size:{}\n", .{size});

    const ioReader = &fr.interface;
    const byte = try ioReader.takeByte();
    std.debug.print("First byte:{c}\n", .{byte});
    const second_byte = try ioReader.takeByte();
    std.debug.print("Second byte:{c}\n", .{second_byte});
    const third_byte = try ioReader.takeByte();
    std.debug.print("Third byte:{c}\n", .{third_byte});
}
