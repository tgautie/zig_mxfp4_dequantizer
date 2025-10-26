const std = @import("std");
const Reader = std.Io.Reader;
const Writer = std.Io.Writer;

pub const Bitstream = struct {
    buffer: u8 = 0,
    bits: u3 = 0,

    /// Initialize a new Bitstream.
    pub fn init() Bitstream {
        return Bitstream{};
    }

    /// Read a single bit from the provided Reader.
    pub fn read(self: *@This(), reader: *Reader) !u1 {
        if (self.bits == 0) {
            @branchHint(.unlikely);
            self.buffer = try reader.takeByte();
        }
        self.bits -%= 1;
        return @truncate(self.buffer >> self.bits);
    }

    /// Write a single bit to the provided Writer.
    pub fn write(self: *@This(), writer: *Writer, bit: u1) !void {
        const select: u8 = @as(u8, 1) << (7 - self.bits);
        self.buffer = (self.buffer & ~select) | (select * bit);
        if (self.bits == 7) {
            @branchHint(.unlikely);
            try writer.writeByte(self.buffer);
        }
        self.bits +%= 1;
    }

    /// Write the current buffer to the Writer, even if not full.
    pub fn flush(self: *@This(), writer: *Writer) !void {
        if (self.bits != 0) {
            try writer.writeByte(self.buffer);
        }
    }
};

pub fn main() !void {
    var stdin_buffer: [1024]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
    const stdin = &stdin_reader.interface;

    var bit_stream = Bitstream.init();

    std.debug.print("Enter 20 bytes:", .{});

    for (1..160) |_| {
        const read_bit: u1 = try bit_stream.read(stdin);
        std.debug.print("{x}\n", .{read_bit});
    }

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    var stdout = &stdout_writer.interface;
    try bit_stream.flush(stdout);
    try stdout.flush();
}
