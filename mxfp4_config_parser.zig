const std = @import("std");
const TensorConfig = @import("safetensors_header_parser.zig").TensorConfig;

pub const Mxfp4TensorConfig = struct {
    mxfp4_tensor_name: []const u8,
    blocks_count: u32,
    scales_dtype: []const u8,
    blocks_dtype: []const u8,
    scales_shape: []const u32,
    blocks_shape: []const u32,
    scales_absolute_offsets: [2]u32,
    blocks_absolute_offsets: [2]u32,
};

pub fn get_mxfp4_tensor_configs(
    allocator: std.mem.Allocator,
    tensor_configs: std.ArrayList(TensorConfig),
) !std.ArrayList(Mxfp4TensorConfig) {
    const scales_map = try get_scales_config_map(allocator, tensor_configs);
    defer scales_map.deinit();

    var mxfp4_tensor_configs = try std.ArrayList(Mxfp4TensorConfig).initCapacity(allocator, 1000000);
    errdefer mxfp4_tensor_configs.deinit(allocator);

    for (tensor_configs.items) |tensor_config| {
        if (!is_mxfp4_blocks_tensor_name(tensor_config.tensor_name)) continue;

        const mxfp4_tensor_name = get_mxfp4_tensor_name(tensor_config.tensor_name);
        if (scales_map.get(mxfp4_tensor_name)) |scales_tensor_config| {
            const mxfp4_tensor_config = format_mxfp4_tensor_config(mxfp4_tensor_name, scales_tensor_config, tensor_config);
            try mxfp4_tensor_configs.append(allocator, mxfp4_tensor_config);
        } else {
            std.debug.print("Corresponding scales tensor config not found for blocks tensor {s}\n", .{mxfp4_tensor_name});
            continue;
        }
    }

    return mxfp4_tensor_configs;
}

fn get_scales_config_map(allocator: std.mem.Allocator, tensor_configs: std.ArrayList(TensorConfig)) !std.StringHashMap(TensorConfig) {
    var scales_map = std.StringHashMap(TensorConfig).init(
        allocator,
    );

    for (tensor_configs.items) |tensor_config| {
        if (is_mxfp4_scales_tensor_name(tensor_config.tensor_name)) try scales_map.put(get_mxfp4_tensor_name(tensor_config.tensor_name), tensor_config);
    }
}

fn is_mxfp4_scales_tensor_name(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_scales");
}

test is_mxfp4_scales_tensor_name {
    try std.testing.expectEqual(true, is_mxfp4_scales_tensor_name("test_scales"));
    try std.testing.expectEqual(false, is_mxfp4_scales_tensor_name("test_blocks"));
}

fn is_mxfp4_blocks_tensor_name(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_blocks");
}

test is_mxfp4_blocks_tensor_name {
    try std.testing.expectEqual(true, is_mxfp4_blocks_tensor_name("test_blocks"));
    try std.testing.expectEqual(false, is_mxfp4_blocks_tensor_name("test_scales"));
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

test get_mxfp4_tensor_name {
    try std.testing.expectEqualStrings("test", get_mxfp4_tensor_name("test_scales"));
    try std.testing.expectEqualStrings("test", get_mxfp4_tensor_name("test_blocks"));
    try std.testing.expectEqualStrings("test", get_mxfp4_tensor_name("test"));
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

test get_total_tensor_values {
    try std.testing.expectEqual(5, get_total_tensor_values(&[_]u32{5}));
    try std.testing.expectEqual(6, get_total_tensor_values(&[_]u32{ 2, 3 }));
    try std.testing.expectEqual(24, get_total_tensor_values(&[_]u32{ 2, 3, 4 }));
}
