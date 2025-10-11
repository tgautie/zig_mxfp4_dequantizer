const std = @import("std");
const input = @embedFile("mixed.safetensors");

const TensorConfig = struct {
    tensor_name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    data_offsets: [2]u32,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const file_buffer = input;

    const header_size = read_header_size(file_buffer);
    std.debug.print("Header size: {}\n", .{header_size});

    try read_header_config(file_buffer, header_size, allocator);
}

fn read_header_size(file_buffer: []const u8) u64 {
    return std.mem.readInt(u64, file_buffer[0..8], .little);
}

fn read_header_config(
    file_buffer: []const u8,
    header_size: u64,
    allocator: std.mem.Allocator,
) !void {
    const header_buf = file_buffer[8 .. 8 + header_size];
    std.debug.print("Header content: {s}\n", .{header_buf});

    const parsed_json = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed_json.deinit();
    const root = parsed_json.value;

    if (root != .object) {
        std.debug.print("Root is not an object\n", .{});
        return;
    }

    var it = root.object.iterator();
    while (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        if (std.mem.eql(u8, tensor_name, "__metadata__")) {
            continue;
        }

        const value = entry.value_ptr.*;
        const tensor_config = try parseTensorConfig(tensor_name, value, allocator);
        std.debug.print("Tensor {s}: {s}, shape: {any}, data offsets: {any}\n", .{ tensor_config.tensor_name, tensor_config.dtype, tensor_config.shape, tensor_config.data_offsets });

        const tensor_values_buf = input[8 + header_size + tensor_config.data_offsets[0] .. 8 + header_size + tensor_config.data_offsets[1]];
        const tensor_values = std.mem.bytesAsSlice(f32, tensor_values_buf);
        std.debug.print("Tensor {s} values: {any}\n", .{ tensor_name, tensor_values });
    }
}

fn parseTensorConfig(
    tensor_name: []const u8,
    json_config: std.json.Value,
    allocator: std.mem.Allocator,
) !TensorConfig {
    const parsed_tensor_config = try std.json.parseFromValue(ParsedTensorConfig, allocator, json_config, .{});

    return TensorConfig{
        .tensor_name = tensor_name,
        .dtype = parsed_tensor_config.value.dtype,
        .shape = parsed_tensor_config.value.shape,
        .data_offsets = parsed_tensor_config.value.data_offsets,
    };
}

const ParsedTensorConfig = struct {
    dtype: []const u8,
    shape: []const u32,
    data_offsets: [2]u32,
};
