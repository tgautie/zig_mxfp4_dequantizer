const std = @import("std");
const mxfp4Config = @import("mxfp4Config.zig");
const mxfp4Dequantization = @import("dequantization.zig");

const Mxfp4TensorConfig = mxfp4Config.Mxfp4TensorConfig;

// The size of the post-decoding f32 block, in bytes
const decoded_block_byte_size = 4 * mxfp4Dequantization.block_size;

// Size of the file reader buffers used internally to stream the blocks and scales from the provided file
const file_reader_buffer_size = 1024;

// This is an implementation of the std.Io.Reader interface, that dequantizes MXFP4 tensors on the fly.
// The reader provides access to the byte stream of decoded f32 values.
pub const DequantizedMxfp4TensorReader = struct {
    name: []const u8,
    // We keep track of the number of dequantized blocks and the total number of blocks
    dequantized_blocks_count: usize,
    total_blocks_count: usize,
    // We keep the last decoded block in memory, in order to stream it byte-by-byte
    current_block: [decoded_block_byte_size]u8,
    current_block_index: usize,
    // The file readers for the scales and blocks input files
    scales_reader: std.fs.File.Reader,
    blocks_reader: std.fs.File.Reader,
    // Heap-allocated buffers for the scales and blocks input readers
    scales_input_buffer: []u8,
    blocks_input_buffer: []u8,
    // std.Io.Reader interface 🌟
    interface: std.Io.Reader,

    // Initializes the reader with the buffer, file path, and MXFP4 tensor config
    pub fn init(reader_buffer: []u8, file_path: []const u8, allocator: std.mem.Allocator, mxfp4_tensor_config: Mxfp4TensorConfig) !DequantizedMxfp4TensorReader {
        var result: DequantizedMxfp4TensorReader = undefined;

        result.scales_input_buffer = try allocator.alloc(u8, file_reader_buffer_size);
        result.blocks_input_buffer = try allocator.alloc(u8, file_reader_buffer_size);

        result.name = try allocator.dupe(u8, mxfp4_tensor_config.mxfp4_tensor_name);
        result.dequantized_blocks_count = 0;
        result.total_blocks_count = mxfp4_tensor_config.blocks_count;
        result.current_block_index = decoded_block_byte_size; // Initialized to the end of the block, to be filled in the first dequantization

        const scales_file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
        result.scales_reader = scales_file.reader(result.scales_input_buffer);
        try result.scales_reader.seekTo(mxfp4_tensor_config.scales_absolute_offsets[0]);

        const blocks_file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
        result.blocks_reader = blocks_file.reader(result.blocks_input_buffer);
        try result.blocks_reader.seekTo(mxfp4_tensor_config.blocks_absolute_offsets[0]);

        result.interface = .{
            .vtable = &vtable,
            .buffer = reader_buffer,
            .seek = 0,
            .end = 0,
        };

        return result;
    }

    // Returns a pointer to the Reader interface
    pub fn reader(self: *DequantizedMxfp4TensorReader) *std.Io.Reader {
        return &self.interface;
    }

    // Cleanup method to properly close file handles
    pub fn deinit(self: *DequantizedMxfp4TensorReader, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.scales_input_buffer);
        allocator.free(self.blocks_input_buffer);
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

test DequantizedMxfp4TensorReader {
    const allocator = std.testing.allocator;

    const test_file_path = "testSafetensors/simple_test.safetensors";

    // Hardcoded MXFP4 tensor config corresponding to the test file
    const mxfp4_config = Mxfp4TensorConfig{
        .mxfp4_tensor_name = "tensor_with_ones",
        .blocks_count = 1,
        .scales_dtype = "U8",
        .blocks_dtype = "U8",
        .scales_shape = &[_]u32{1},
        .blocks_shape = &[_]u32{ 1, 16 },
        .scales_absolute_offsets = [2]u32{ 336, 337 },
        .blocks_absolute_offsets = [2]u32{ 320, 336 },
    };

    const reader_buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(reader_buffer);

    var reader = try DequantizedMxfp4TensorReader.init(reader_buffer, test_file_path, allocator, mxfp4_config);
    defer reader.deinit(allocator);

    const output_buffer = try reader.interface.takeArray(10);
    const expected: [10]u8 = .{
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00,
        0x00, 0x00,
    };

    inline for (0..10) |i| {
        try std.testing.expectEqual(expected[i], output_buffer.*[i]);
    }
}
