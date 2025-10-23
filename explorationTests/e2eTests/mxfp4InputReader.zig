const std = @import("std");

pub fn main() void {
    const allocator = std.heap.page_allocator;
    parse_mxfp4_tensors(allocator) catch |err| {
        std.debug.print("Error: {}\n", .{err});
    };
}

fn parse_mxfp4_tensors(allocator: std.mem.Allocator) !void {
    const file_path = "exampleSafetensors/test_mxfp4.safetensors";

    var tensor_configs = try parse_tensor_configs(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    std.debug.print("Contains {} tensors:\n", .{tensor_configs.items.len});
    for (tensor_configs.items) |tensor_config| {
        std.debug.print("- {s}: {any} {s}\n", .{ tensor_config.tensor_name, tensor_config.shape, tensor_config.dtype });
    }

    std.debug.print("Parsing MXFP4 tensor configs:\n", .{});
    var mxfp4_tensor_configs = try parse_mxfp4_tensor_configs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        std.debug.print("MXFP4 tensor config {s}: {any} \n", .{ mxfp4_tensor_config.mxfp4_tensor_name, mxfp4_tensor_config });
    }

    std.debug.print("Reading MXFP4 tensor values:\n", .{});
    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        _ = try read_mxfp4_tensor_values(allocator, file_path, mxfp4_tensor_config);
    }
}

const Mxfp4TensorConfig = struct {
    mxfp4_tensor_name: []const u8,
    scales_dtype: []const u8,
    blocks_dtype: []const u8,
    scales_shape: []const u32,
    blocks_shape: []const u32,
    scales_absolute_offsets: [2]u32,
    blocks_absolute_offsets: [2]u32,
};

fn parse_mxfp4_tensor_configs(
    allocator: std.mem.Allocator,
    tensor_configs: std.ArrayList(TensorConfig),
) !std.ArrayList(Mxfp4TensorConfig) {
    var mxfp4_tensor_configs = try std.ArrayList(Mxfp4TensorConfig).initCapacity(allocator, 1000000);
    errdefer mxfp4_tensor_configs.deinit(allocator);

    var scales_map = std.StringHashMap(TensorConfig).init(
        allocator,
    );
    defer scales_map.deinit();

    for (tensor_configs.items) |tensor_config| {
        if (is_mxfp4_scales_tensor_name(tensor_config.tensor_name)) try scales_map.put(get_mxfp4_tensor_name(tensor_config.tensor_name), tensor_config);
    }

    for (tensor_configs.items) |tensor_config| {
        if (!is_mxfp4_blocks_tensor_name(tensor_config.tensor_name)) continue;
        const mxfp4_tensor_name = get_mxfp4_tensor_name(tensor_config.tensor_name);
        if (scales_map.get(mxfp4_tensor_name)) |scales_tensor_config| {
            const mxfp4_tensor_config = format_mxfp4_tensor_config(mxfp4_tensor_name, scales_tensor_config, tensor_config);
            try mxfp4_tensor_configs.append(allocator, mxfp4_tensor_config);
        } else {
            std.debug.print("Scales tensor config not found for {s}\n", .{mxfp4_tensor_name});
            continue;
        }
    }

    return mxfp4_tensor_configs;
}

fn is_mxfp4_scales_tensor_name(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_scales");
}

fn is_mxfp4_blocks_tensor_name(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_blocks");
}

fn get_mxfp4_tensor_name(tensor_name: []const u8) []const u8 {
    if (is_mxfp4_scales_tensor_name(tensor_name)) {
        return tensor_name[0 .. tensor_name.len - "_scales".len];
    } else if (is_mxfp4_blocks_tensor_name(tensor_name)) {
        return tensor_name[0 .. tensor_name.len - "_blocks".len];
    } else {
        return tensor_name;
    }
}

fn format_mxfp4_tensor_config(mxfp4_tensor_name: []const u8, scales_tensor_config: TensorConfig, blocks_tensor_config: TensorConfig) Mxfp4TensorConfig {
    return Mxfp4TensorConfig{
        .mxfp4_tensor_name = mxfp4_tensor_name,
        .scales_dtype = scales_tensor_config.dtype,
        .blocks_dtype = blocks_tensor_config.dtype,
        .scales_shape = scales_tensor_config.shape,
        .blocks_shape = blocks_tensor_config.shape,
        .scales_absolute_offsets = scales_tensor_config.data_absolute_offsets,
        .blocks_absolute_offsets = blocks_tensor_config.data_absolute_offsets,
    };
}

const TensorConfig = struct {
    tensor_name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    data_absolute_offsets: [2]u32,
};

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
        const tensor_config = try parse_tensor_config(allocator, tensor_name, value, header_size);
        try tensor_configs.append(allocator, tensor_config);
    }

    return tensor_configs;
}

fn parse_tensor_config(
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

fn read_mxfp4_tensor_values(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    mxfp4_tensor_config: Mxfp4TensorConfig,
) ![]const u8 {
    var scales_reader = try get_reader(file_path, mxfp4_tensor_config, false);
    var blocks_reader = try get_reader(file_path, mxfp4_tensor_config, true);

    const scales_values_buf = scales_reader.interface.readAlloc(allocator, mxfp4_tensor_config.scales_absolute_offsets[1] - mxfp4_tensor_config.scales_absolute_offsets[0]);
    const blocks_values_buf = blocks_reader.interface.readAlloc(allocator, mxfp4_tensor_config.blocks_absolute_offsets[1] - mxfp4_tensor_config.blocks_absolute_offsets[0]);

    std.debug.print("MXFP4 tensor name: {s}\n", .{mxfp4_tensor_config.mxfp4_tensor_name});
    std.debug.print("Scales values: {any}\n", .{scales_values_buf});
    std.debug.print("Blocks values: {any}\n", .{blocks_values_buf});

    return "";
}

fn get_reader(file_path: []const u8, mxfp4_tensor_config: Mxfp4TensorConfig, is_blocks: bool) !std.fs.File.Reader {
    var file: std.fs.File = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    try file_reader.seekTo(if (is_blocks) mxfp4_tensor_config.blocks_absolute_offsets[0] else mxfp4_tensor_config.scales_absolute_offsets[0]);
    return file_reader;
}
