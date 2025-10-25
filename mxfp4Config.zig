const std = @import("std");
const safetensors = @import("safetensors.zig");

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

pub fn extractMxfp4TensorConfigs(
    allocator: std.mem.Allocator,
    tensor_configs: std.ArrayList(safetensors.TensorConfig),
) !std.ArrayList(Mxfp4TensorConfig) {
    var scales_map = try getScalesConfigMap(allocator, tensor_configs);
    defer scales_map.deinit();

    var mxfp4_tensor_configs: std.ArrayList(Mxfp4TensorConfig) = .empty;
    errdefer mxfp4_tensor_configs.deinit(allocator);

    for (tensor_configs.items) |tensor_config| {
        if (!isMxfp4BlocksTensorName(tensor_config.tensor_name)) continue;

        const mxfp4_tensor_name = getMxfp4TensorName(tensor_config.tensor_name);
        if (scales_map.get(mxfp4_tensor_name)) |scales_tensor_config| {
            const mxfp4_tensor_config = formatMxfp4TensorConfig(mxfp4_tensor_name, scales_tensor_config, tensor_config);
            try mxfp4_tensor_configs.append(allocator, mxfp4_tensor_config);
        } else {
            std.debug.print("Corresponding scales tensor config not found for blocks tensor {s}\n", .{mxfp4_tensor_name});
            continue;
        }
    }

    return mxfp4_tensor_configs;
}

test extractMxfp4TensorConfigs {
    const allocator = std.testing.allocator;

    // Create test data
    var tensor_configs = try std.ArrayList(safetensors.TensorConfig).initCapacity(allocator, 5);
    defer {
        for (tensor_configs.items) |config| {
            config.deinit(allocator);
        }
        tensor_configs.deinit(allocator);
    }

    // Helper function to create tensor configs
    const addTensor = struct {
        fn add(alloc: std.mem.Allocator, configs: *std.ArrayList(safetensors.TensorConfig), name: []const u8, dtype: []const u8, shape: []const u32, offsets: [2]u32) !void {
            try configs.append(alloc, safetensors.TensorConfig{
                .tensor_name = try alloc.dupe(u8, name),
                .dtype = try alloc.dupe(u8, dtype),
                .shape = try alloc.dupe(u32, shape),
                .data_absolute_offsets = offsets,
            });
        }
    }.add;

    // Add test tensors
    try addTensor(allocator, &tensor_configs, "tensor1_blocks", "U8", &[_]u32{ 2, 3 }, [2]u32{ 100, 200 });
    try addTensor(allocator, &tensor_configs, "tensor1_scales", "F32", &[_]u32{ 1, 1 }, [2]u32{ 50, 100 });
    try addTensor(allocator, &tensor_configs, "tensor2_blocks", "U8", &[_]u32{ 4, 5 }, [2]u32{ 300, 400 });
    try addTensor(allocator, &tensor_configs, "tensor2_scales", "F32", &[_]u32{ 1, 1 }, [2]u32{ 250, 300 });
    try addTensor(allocator, &tensor_configs, "regular_tensor", "F32", &[_]u32{ 2, 3 }, [2]u32{ 500, 600 });

    // Test the function
    var mxfp4_configs = try extractMxfp4TensorConfigs(allocator, tensor_configs);
    defer mxfp4_configs.deinit(allocator);

    // Basic validation
    try std.testing.expectEqual(@as(usize, 2), mxfp4_configs.items.len);

    // Check first config
    const config1 = mxfp4_configs.items[0];
    try std.testing.expectEqualStrings("tensor1", config1.mxfp4_tensor_name);
    try std.testing.expectEqual(@as(u32, 1), config1.blocks_count);
    try std.testing.expectEqualStrings("F32", config1.scales_dtype);
    try std.testing.expectEqualStrings("U8", config1.blocks_dtype);

    // Check second config
    const config2 = mxfp4_configs.items[1];
    try std.testing.expectEqualStrings("tensor2", config2.mxfp4_tensor_name);
    try std.testing.expectEqual(@as(u32, 1), config2.blocks_count);
    try std.testing.expectEqualStrings("F32", config2.scales_dtype);
    try std.testing.expectEqualStrings("U8", config2.blocks_dtype);
}

fn getScalesConfigMap(allocator: std.mem.Allocator, tensor_configs: std.ArrayList(safetensors.TensorConfig)) !std.StringHashMap(safetensors.TensorConfig) {
    var scales_map = std.StringHashMap(safetensors.TensorConfig).init(
        allocator,
    );

    for (tensor_configs.items) |tensor_config| {
        if (isMxfp4ScalesTensorName(tensor_config.tensor_name)) try scales_map.put(getMxfp4TensorName(tensor_config.tensor_name), tensor_config);
    }

    return scales_map;
}

fn isMxfp4ScalesTensorName(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_scales");
}

test isMxfp4ScalesTensorName {
    try std.testing.expectEqual(true, isMxfp4ScalesTensorName("test_scales"));
    try std.testing.expectEqual(false, isMxfp4ScalesTensorName("test_blocks"));
}

fn isMxfp4BlocksTensorName(tensor_name: []const u8) bool {
    return std.mem.endsWith(u8, tensor_name, "_blocks");
}

test isMxfp4BlocksTensorName {
    try std.testing.expectEqual(true, isMxfp4BlocksTensorName("test_blocks"));
    try std.testing.expectEqual(false, isMxfp4BlocksTensorName("test_scales"));
}

fn getMxfp4TensorName(tensor_name: []const u8) []const u8 {
    if (isMxfp4ScalesTensorName(tensor_name)) {
        return tensor_name[0 .. tensor_name.len - "_scales".len];
    } else if (isMxfp4BlocksTensorName(tensor_name)) {
        return tensor_name[0 .. tensor_name.len - "_blocks".len];
    } else {
        return tensor_name;
    }
}

test getMxfp4TensorName {
    try std.testing.expectEqualStrings("test", getMxfp4TensorName("test_scales"));
    try std.testing.expectEqualStrings("test", getMxfp4TensorName("test_blocks"));
    try std.testing.expectEqualStrings("test", getMxfp4TensorName("test"));
}

fn formatMxfp4TensorConfig(mxfp4_tensor_name: []const u8, scales_tensor_config: safetensors.TensorConfig, blocks_tensor_config: safetensors.TensorConfig) Mxfp4TensorConfig {
    return Mxfp4TensorConfig{
        .mxfp4_tensor_name = mxfp4_tensor_name,
        .blocks_count = getTotalTensorValues(scales_tensor_config.shape),
        .scales_dtype = scales_tensor_config.dtype,
        .blocks_dtype = blocks_tensor_config.dtype,
        .scales_shape = scales_tensor_config.shape,
        .blocks_shape = blocks_tensor_config.shape,
        .scales_absolute_offsets = scales_tensor_config.data_absolute_offsets,
        .blocks_absolute_offsets = blocks_tensor_config.data_absolute_offsets,
    };
}

fn getTotalTensorValues(shape: []const u32) u32 {
    var total_values: u32 = 1;
    for (shape) |dimension_size| {
        total_values *= dimension_size;
    }
    return total_values;
}

test getTotalTensorValues {
    try std.testing.expectEqual(5, getTotalTensorValues(&[_]u32{5}));
    try std.testing.expectEqual(6, getTotalTensorValues(&[_]u32{ 2, 3 }));
    try std.testing.expectEqual(24, getTotalTensorValues(&[_]u32{ 2, 3, 4 }));
}
