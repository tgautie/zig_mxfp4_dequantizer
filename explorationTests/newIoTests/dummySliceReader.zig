const std = @import("std");

/// A minimal Reader implementation using the new std.Io.Reader interface.
/// This example generates repeating 10-character blocks instead of single bytes.
pub const DummySliceReader = struct {
    // Our internal state - a 10-character block
    next_slice: [10]u8 = "ABCDEFGHIJ".*,
    block_index: usize = 0, // Track position within current block

    interface: std.Io.Reader,

    /// Initialize the reader with an empty or user-provided buffer.
    pub fn init(buffer: []u8) DummySliceReader {
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
    pub fn reader(self: *DummySliceReader) *std.Io.Reader {
        return &self.interface;
    }

    // -------------------------------
    // Required vtable implementation
    // -------------------------------

    /// The only required method: stream()
    ///
    /// This fills up to `limit` bytes into the Writer `w` from 10-byte blocks.
    /// When limit is not a multiple of 10, we track our position within the current block.
    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) !usize {
        const self: *DummySliceReader = @fieldParentPtr("interface", r);

        const max_bytes: usize = @intFromEnum(limit);
        var total: usize = 0;

        while (total < max_bytes) {
            std.debug.print("Block: {s}\n", .{self.next_slice});
            // How many bytes can we write from the current position in the block?
            const remaining_in_block = self.next_slice.len - self.block_index;
            const bytes_to_write: usize = @min(remaining_in_block, max_bytes - total);

            // Write the slice
            const slice_to_write = self.next_slice[self.block_index .. self.block_index + bytes_to_write];
            try w.writeAll(slice_to_write);
            total += bytes_to_write;
            self.block_index += bytes_to_write;

            // If we've consumed the entire block, generate the next one
            if (self.block_index >= self.next_slice.len) {
                self.generateNextBlock();
                self.block_index = 0;
            }
        }
        return total;
    }

    /// Generate the next 10-character block by cycling through the alphabet
    fn generateNextBlock(self: *DummySliceReader) void {
        for (&self.next_slice) |*char| {
            char.* = if (char.* > 'Z' - 10) char.* - 16 else char.* + 10;
        }
    }

    pub const vtable = std.Io.Reader.VTable{
        .stream = stream,
        // discard, readVec, and rebase all have defaults
    };
};

/// Example usage
pub fn main() !void {
    var buffer: [64]u8 = undefined;

    var dummy = DummySliceReader.init(&buffer);
    const reader = dummy.reader();

    // Read 100 bytes and print them to stdout
    var r_buf: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&r_buf);

    try reader.streamExact(&stdout_writer.interface, 100);
    try stdout_writer.interface.flush();
}
