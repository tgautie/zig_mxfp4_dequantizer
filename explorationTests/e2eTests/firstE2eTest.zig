const std = @import("std");

pub fn main() void {
    const allocator = std.heap.page_allocator;
    parse_tensors(allocator) catch |err| {
        std.debug.print("Error: {}\n", .{err});
    };
}

const TensorConfig = struct {
    tensor_name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    data_offsets: [2]u32,
};

fn parse_tensors(allocator: std.mem.Allocator) !void {
    const file_path = "exampleSafetensors/mixed.safetensors";

    var tensor_configs = try parse_tensor_configs(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    std.debug.print("Contains {} tensors:\n", .{tensor_configs.items.len});
    for (tensor_configs.items) |tensor_config| {
        std.debug.print("- {s}: {any} {s}\n", .{ tensor_config.tensor_name, tensor_config.shape, tensor_config.dtype });
    }

    // std.debug.print("Reading tensor values:\n", .{});
    // for (tensor_configs.items) |tensor_config| {
    //     const tensor_values = read_tensor_values(
    //         file_reader,
    //         header_size,
    //         tensor_config,
    //     );
    //     if (std.mem.eql(u8, tensor_config.dtype, "F32")) {
    //         const tensor_values_f32 = std.mem.bytesAsSlice(f32, tensor_values);
    //         std.debug.print("- {s}: {any}\n", .{ tensor_config.tensor_name, tensor_values_f32 });
    //     } else {
    //         std.debug.print("Not implemented: {s}\n", .{tensor_config.dtype});
    //     }
    // }
}

fn parse_tensor_configs(
    file_path: []const u8,
    allocator: std.mem.Allocator,
) !std.ArrayList(TensorConfig) {
    var file: std.fs.File = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    const header_size = std.mem.readInt(u64, try file_reader.interface.takeArray(8), .little);
    std.debug.print("Header size: {any}\n", .{header_size});

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
        const tensor_config = try parse_tensor_config(tensor_name, value, allocator);
        try tensor_configs.append(allocator, tensor_config);
    }

    return tensor_configs;
}

fn parse_tensor_config(
    tensor_name: []const u8,
    json_config: std.json.Value,
    allocator: std.mem.Allocator,
) !TensorConfig {
    const parsed_tensor_config = try std.json.parseFromValue(ParsedTensorConfig, allocator, json_config, .{});

    const name_copy = try allocator.dupe(u8, tensor_name);
    const dtype_copy = try allocator.dupe(u8, parsed_tensor_config.value.dtype);
    const shape_copy = try allocator.dupe(u32, parsed_tensor_config.value.shape);

    return TensorConfig{
        .tensor_name = name_copy,
        .dtype = dtype_copy,
        .shape = shape_copy,
        .data_offsets = parsed_tensor_config.value.data_offsets,
    };
}

const ParsedTensorConfig = struct {
    dtype: []const u8,
    shape: []const u32,
    data_offsets: [2]u32,
};

fn read_tensor_values(
    file_reader: std.fs.File.Reader,
    header_size: u64,
    tensor_config: TensorConfig,
) []const u8 {
    const start_offset = 8 + header_size + tensor_config.data_offsets[0];
    try file_reader.interface.seekTo(start_offset);
    const end_offset = 8 + header_size + tensor_config.data_offsets[1];
    try file_reader.interface.seekTo(end_offset);
    const tensor_values_buf = try file_reader.takeArray(end_offset - start_offset);
    return tensor_values_buf;
}
