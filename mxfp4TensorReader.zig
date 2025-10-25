const std = @import("std");
const mxfp4Config = @import("mxfp4Config.zig");
const mxfp4Dequantization = @import("mxfp4Dequantization.zig");

const Mxfp4TensorConfig = mxfp4Config.Mxfp4TensorConfig;

// The size of each decoded f32 block, in bytes
const decoded_block_byte_size = 4 * mxfp4Dequantization.block_size;

// Reader interface implementation that dequantizes MXFP4 tensors on the fly, from a safetensors file path and a MXFP4 tensor config.
// The reader provides the byte stream of decoded f32 values.
pub const DequantizedMxfp4TensorReader = struct {
    dequantized_blocks_count: usize,
    total_blocks_count: usize,
    current_block: [decoded_block_byte_size]u8,
    current_block_index: usize,
    scales_reader: std.fs.File.Reader,
    blocks_reader: std.fs.File.Reader,
    interface: std.Io.Reader,

    // Initializes the reader with the buffer, file path, and MXFP4 tensor config
    pub fn init(buffer: []u8, file_path: []const u8, mxfp4_tensor_config: Mxfp4TensorConfig) !DequantizedMxfp4TensorReader {
        var scales_reader = try getReader(file_path, mxfp4_tensor_config, false);
        _ = &scales_reader;
        var blocks_reader = try getReader(file_path, mxfp4_tensor_config, true);
        _ = &blocks_reader;

        return .{
            .dequantized_blocks_count = 0,
            .total_blocks_count = mxfp4_tensor_config.blocks_count,
            .current_block = undefined,
            .current_block_index = decoded_block_byte_size, // We initialize to the block size in order to trigger dequantization on the first stream call
            .scales_reader = scales_reader,
            .blocks_reader = blocks_reader,
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    // Returns a pointer to the Reader interface
    pub fn reader(self: *DequantizedMxfp4TensorReader) *std.Io.Reader {
        return &self.interface;
    }

    /// The stream method fills up to `limit` bytes into the Writer `w` from the dequantized 32-float blocks.
    /// When limit is not a multiple of the block size (32), we track our position within the current block.
    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) !usize {
        const self: *DequantizedMxfp4TensorReader = @fieldParentPtr("interface", r);

        const max_bytes: usize = @intFromEnum(limit);
        var total: usize = 0;

        while (total < max_bytes) {
            if (self.current_block_index >= decoded_block_byte_size) {
                self.dequantizeNextBlock() catch |err| switch (err) {
                    error.EndOfStream => return total,
                    else => return err,
                };
            }

            const remaining_bytes_in_block = decoded_block_byte_size - self.current_block_index;
            const bytes_to_write: usize = @min(remaining_bytes_in_block, max_bytes - total);

            const slice_to_write = self.current_block[self.current_block_index .. self.current_block_index + bytes_to_write];
            try w.writeAll(slice_to_write);

            total += bytes_to_write;
            self.current_block_index += bytes_to_write;
        }

        return total;
    }

    fn dequantizeNextBlock(self: *DequantizedMxfp4TensorReader) !void {
        if (self.isCompleted()) {
            return error.EndOfStream;
        }

        const scale = try self.scales_reader.interface.takeByte();
        const block = try self.blocks_reader.interface.takeArray(16);
        const decoded_mxfp4_block = mxfp4Dequantization.decodeBlock(scale, block.*);

        @memcpy(&self.current_block, std.mem.sliceAsBytes(&decoded_mxfp4_block));
        self.current_block_index = 0;
        self.dequantized_blocks_count += 1;
    }

    fn isCompleted(self: *DequantizedMxfp4TensorReader) bool {
        return self.dequantized_blocks_count >= self.total_blocks_count;
    }

    pub const vtable = std.Io.Reader.VTable{
        .stream = stream,
        // discard, readVec, and rebase all have defaults
    };
};

fn getReader(file_path: []const u8, mxfp4_tensor_config: Mxfp4TensorConfig, is_blocks: bool) !std.fs.File.Reader {
    var file: std.fs.File = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    try file_reader.seekTo(if (is_blocks) mxfp4_tensor_config.blocks_absolute_offsets[0] else mxfp4_tensor_config.scales_absolute_offsets[0]);

    return file_reader;
}
