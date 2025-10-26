const std = @import("std");
const input = @embedFile("test.txt");

pub fn main() !void {
    var stream = std.Io.Reader.fixed(input);
    const stdin = &stream;

    while (true) {
        const read_byte: u8 = stdin.takeByte() catch |err| {
            std.debug.print("Error reading byte: {}\n", .{err});
            break;
        };
        if (read_byte == 0) {
            break;
        }
        std.debug.print("{c}\n", .{read_byte});
    }
}
