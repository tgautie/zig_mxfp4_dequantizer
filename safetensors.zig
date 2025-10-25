const std = @import("std");

pub const TensorConfig = struct {
    tensor_name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    data_absolute_offsets: [2]u32,
};

pub fn parseHeader(
    file_path: []const u8,
    allocator: std.mem.Allocator,
) !std.ArrayList(TensorConfig) {
    var file: std.fs.File = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    const header_size = std.mem.readInt(u64, try file_reader.interface.takeArray(8), .little);

    var tensor_configs = try std.ArrayList(TensorConfig).initCapacity(allocator, 1000000);
    errdefer tensor_configs.deinit(allocator);

    const header_buf = try file_reader.interface.readAlloc(allocator, header_size);
    defer allocator.free(header_buf);

    const parsed_json = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed_json.deinit();
    const root = parsed_json.value;

    if (root != .object) {
        return error.RootIsNotAnObject;
    }

    var it = root.object.iterator();
    while (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        if (std.mem.eql(u8, tensor_name, "__metadata__")) {
            continue;
        }

        const value = entry.value_ptr.*;
        const tensor_config = try parseTensorConfig(allocator, tensor_name, value, header_size);
        try tensor_configs.append(allocator, tensor_config);
    }

    return tensor_configs;
}

fn parseTensorConfig(
    allocator: std.mem.Allocator,
    tensor_name: []const u8,
    json_config: std.json.Value,
    header_size: u64,
) !TensorConfig {
    const parsed_tensor_config = try std.json.parseFromValue(ParsedTensorConfig, allocator, json_config, .{});
    defer parsed_tensor_config.deinit();

    const name_copy = try allocator.dupe(u8, tensor_name);
    const dtype_copy = try allocator.dupe(u8, parsed_tensor_config.value.dtype);
    const shape_copy = try allocator.dupe(u32, parsed_tensor_config.value.shape);
    const data_offsets = parsed_tensor_config.value.data_offsets;
    const total_header_size = 8 + @as(u32, @intCast(header_size));
    const absolute_data_offsets = [2]u32{ total_header_size + data_offsets[0], total_header_size + data_offsets[1] };

    return TensorConfig{
        .tensor_name = name_copy,
        .dtype = dtype_copy,
        .shape = shape_copy,
        .data_absolute_offsets = absolute_data_offsets,
    };
}

const ParsedTensorConfig = struct {
    dtype: []const u8,
    shape: []const u32,
    data_offsets: [2]u32,
};
