const std = @import("std");

pub fn main() void {
    const allocator = std.heap.page_allocator;
    dequantize_mxfp4_safetensors(allocator) catch |err| {
        std.debug.print("Error: {}\n", .{err});
    };
}

fn dequantize_mxfp4_safetensors(allocator: std.mem.Allocator) !void {
    const file_path = "exampleSafetensors/test_mxfp4.safetensors";

    var tensor_configs = try parse_tensor_configs(file_path, allocator);
    defer tensor_configs.deinit(allocator);

    var mxfp4_tensor_configs = try parse_mxfp4_tensor_configs(allocator, tensor_configs);
    defer mxfp4_tensor_configs.deinit(allocator);

    for (mxfp4_tensor_configs.items) |mxfp4_tensor_config| {
        const dequantized_mxfp4_tensor = try get_dequantized_mxfp4_tensor(allocator, file_path, mxfp4_tensor_config);
        defer allocator.free(dequantized_mxfp4_tensor);

        std.debug.print("Dequantized MXFP4 tensor {s}: {any} values\n", .{ mxfp4_tensor_config.mxfp4_tensor_name, dequantized_mxfp4_tensor.len });
    }
}

const Mxfp4TensorConfig = struct {
    mxfp4_tensor_name: []const u8,
    blocks_count: u32,
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
        .blocks_count = get_total_tensor_values(scales_tensor_config.shape),
        .scales_dtype = scales_tensor_config.dtype,
        .blocks_dtype = blocks_tensor_config.dtype,
        .scales_shape = scales_tensor_config.shape,
        .blocks_shape = blocks_tensor_config.shape,
        .scales_absolute_offsets = scales_tensor_config.data_absolute_offsets,
        .blocks_absolute_offsets = blocks_tensor_config.data_absolute_offsets,
    };
}

fn get_total_tensor_values(shape: []const u32) u32 {
    var total_values: u32 = 1;
    for (shape) |dimension_size| {
        total_values *= dimension_size;
    }
    return total_values;
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

fn get_dequantized_mxfp4_tensor(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    mxfp4_tensor_config: Mxfp4TensorConfig,
) ![]f32 {
    var scales_reader = try get_reader(file_path, mxfp4_tensor_config, false);
    var blocks_reader = try get_reader(file_path, mxfp4_tensor_config, true);

    var dequantized_mxfp4_values = try allocator.alloc(f32, mxfp4_tensor_config.blocks_count * 32);
    for (0..mxfp4_tensor_config.blocks_count) |i| {
        const scale = try scales_reader.interface.takeByte();
        const block = try blocks_reader.interface.takeArray(16);
        const decoded_mxfp4_block = dequantize_mxfp4_block(scale, block.*);
        @memcpy(dequantized_mxfp4_values[i * 32 .. (i + 1) * 32], &decoded_mxfp4_block);
    }

    return dequantized_mxfp4_values;
}

fn get_reader(file_path: []const u8, mxfp4_tensor_config: Mxfp4TensorConfig, is_blocks: bool) !std.fs.File.Reader {
    var file: std.fs.File = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&buffer);

    try file_reader.seekTo(if (is_blocks) mxfp4_tensor_config.blocks_absolute_offsets[0] else mxfp4_tensor_config.scales_absolute_offsets[0]);

    return file_reader;
}

fn dequantize_mxfp4_block(scale: u8, block: [16]u8) [32]f32 {
    const block_vec = get_block_f32_vec(block);
    const scale_vec = get_scale_f32_vec(scale);
    const decoded_mxfp4_vec = block_vec * scale_vec;
    const decoded_mxfp4_block: [32]f32 = decoded_mxfp4_vec;
    return decoded_mxfp4_block;
}

const fp4_to_f32_decode_table = [16]f32{
    0.0,  0.5,  1.0,  1.5,
    2.0,  3.0,  4.0,  6.0,
    0.0,  -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
};

fn get_block_f32_vec(block: [16]u8) @Vector(32, f32) {
    var result: @Vector(32, f32) = undefined;
    inline for (0..16) |i| {
        const b = block[i];
        result[i * 2] = fp4_to_f32_decode_table[b >> 4];
        result[i * 2 + 1] = fp4_to_f32_decode_table[b & 0x0F];
    }
    return result;
}

fn get_scale_f32_vec(x: u8) @Vector(32, f32) {
    const bias: i32 = 127;
    const exponent = @as(i32, x);
    const biased_exponent = @as(f32, @floatFromInt(exponent - bias));
    const scale_f32 = std.math.pow(f32, 2.0, biased_exponent);
    const scale_vec: @Vector(32, f32) = @splat(scale_f32);
    return scale_vec;
}
