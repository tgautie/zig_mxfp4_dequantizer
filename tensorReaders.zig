const std = @import("std");
const tensorReader = @import("tensorReader.zig");
const mxfp4Config = @import("mxfp4Config.zig");
const safetensors = @import("safetensors.zig");

// This is the entry point to the library.
// It provides a DequantizedMxfp4TensorReader for each MXFP4 tensor found in the safetensors file.
// On initialization, this struct parses the safetensors header, extracts the MXFP4 tensor configs, initializes the reader buffers.
pub const DequantizedMxfp4TensorReaders = struct {
    readers: std.ArrayList(*tensorReader.DequantizedMxfp4TensorReader),
    buffers: std.ArrayList([]u8),

    pub fn init(allocator: std.mem.Allocator, file_path: []const u8) !DequantizedMxfp4TensorReaders {
        var tensor_configs = try safetensors.parseHeader(file_path, allocator);
        defer {
            for (tensor_configs.items) |config| {
                config.deinit(allocator);
            }
            tensor_configs.deinit(allocator);
        }

        var mxfp4_tensor_configs = try mxfp4Config.extractFromTensorConfigs(allocator, tensor_configs);
        defer mxfp4_tensor_configs.deinit(allocator);

        var readers: std.ArrayList(*tensorReader.DequantizedMxfp4TensorReader) = .empty;
        var buffers: std.ArrayList([]u8) = .empty;

        for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
            const reader_buffer = try allocator.alloc(u8, 4096);
            try buffers.append(allocator, reader_buffer);

            const reader = try allocator.create(tensorReader.DequantizedMxfp4TensorReader);
            reader.* = try tensorReader.DequantizedMxfp4TensorReader.init(reader_buffer, file_path, allocator, mxfp4_tensor_config);

            try readers.append(allocator, reader);
        }

        return DequantizedMxfp4TensorReaders{
            .readers = readers,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: *DequantizedMxfp4TensorReaders, allocator: std.mem.Allocator) void {
        for (self.readers.items) |reader| {
            reader.deinit(allocator);
        }
        self.readers.deinit(allocator);

        for (self.buffers.items) |buffer| {
            allocator.free(buffer);
        }
        self.buffers.deinit(allocator);
    }
};
