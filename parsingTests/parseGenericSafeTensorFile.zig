const std = @import("std");
const input = @embedFile("ones.safetensors");

const TensorConfig = struct {
    tensor_name: []const u8,
    dtype: []const u8,
    shape: []u32,
    data_offsets: [2]u32,
};

pub fn main() !void {
    const header_size = std.mem.readInt(u64, input[0..8], .little);
    std.debug.print("Header size: {}\n", .{header_size});

    const header_buf = input[8 .. 8 + header_size];
    std.debug.print("Header content: {s}\n", .{header_buf});

    const allocator = std.heap.page_allocator;
    const parsed_json = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed_json.deinit();

    try iterateOnTensors(parsed_json.value, header_size, allocator);
}

fn iterateOnTensors(
    root: std.json.Value,
    header_size: u64,
    allocator: std.mem.Allocator,
) !void {
    if (root == .object) {
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
    } else {
        std.debug.print("Root is not an object\n", .{});
    }
}

fn parseTensorConfig(
    tensor_name: []const u8,
    json_config: std.json.Value,
    allocator: std.mem.Allocator,
) !TensorConfig {
    const parsed_tensor_config = try std.json.parseFromValue(ParsedTensorConfig, allocator, json_config, .{});

    var shapes: []u32 = try allocator.alloc(u32, parsed_tensor_config.value.shape.len);
    for (parsed_tensor_config.value.shape, 0..) |b, i| {
        shapes[i] = @intCast(b);
    }

    var offsets: [2]u32 = undefined;
    for (parsed_tensor_config.value.data_offsets, 0..) |b, i| {
        offsets[i] = @intCast(b);
    }

    return TensorConfig{
        .tensor_name = tensor_name,
        .dtype = parsed_tensor_config.value.dtype,
        .shape = shapes,
        .data_offsets = offsets,
    };
}

const ParsedTensorConfig = struct {
    dtype: []const u8,
    shape: []const u8,
    data_offsets: []const u8,
};
