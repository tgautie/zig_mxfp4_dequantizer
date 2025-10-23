const std = @import("std");

/// A minimal Reader implementation using the new std.Io.Reader interface.
/// This example generates repeating ASCII letters instead of reading real data.
pub const DummyReader = struct {
    // Our internal state
    next_byte: u8 = 'A',
    interface: std.Io.Reader,

    /// Initialize the reader with an empty or user-provided buffer.
    pub fn init(buffer: []u8) DummyReader {
        return .{
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    /// Return a pointer to the Reader interface (important: do NOT copy by value)
    pub fn reader(self: *DummyReader) *std.Io.Reader {
        return &self.interface;
    }

    // -------------------------------
    // Required vtable implementation
    // -------------------------------

    /// The only required method: stream()
    ///
    /// This must fill up to `limit` bytes into the Writer `w`.
    /// For simplicity, we'll just write a repeating pattern of letters.
    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) !usize {
        const self: *DummyReader = @fieldParentPtr("interface", r);

        const max_bytes = @intFromEnum(limit);
        var total: usize = 0;

        while (total < max_bytes) {
            const c = self.next_byte;
            var buf: [1]u8 = .{c};
            try w.writeAll(&buf);
            total += buf.len;

            // Cycle through 'A'..'Z'
            self.next_byte = if (self.next_byte == 'Z') 'A' else self.next_byte + 1;
        }
        return total;
    }

    pub const vtable = std.Io.Reader.VTable{
        .stream = stream,
        // discard, readVec, and rebase all have defaults
    };
};

/// Example usage
pub fn main() !void {
    var buffer: [64]u8 = undefined;

    var dummy = DummyReader.init(&buffer);
    const reader = dummy.reader();

    // Read 100 bytes and print them to stdout

    var r_buf: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&r_buf);

    try reader.streamExact(&stdout_writer.interface, 100);
    try stdout_writer.interface.flush();
}
