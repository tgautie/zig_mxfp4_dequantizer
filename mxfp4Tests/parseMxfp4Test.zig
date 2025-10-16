const std = @import("std");

pub fn main() void {
    const block_length = 32;
    const block_buf: [block_length / 2]u8 = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };

    const fp4_vec = u8_to_fp4_vec(block_length, block_buf);
    std.debug.print("{any}\n", .{fp4_vec});

    const f32_vec = fp4_to_f32_vec(block_length, fp4_vec);
    std.debug.print("{any}\n", .{f32_vec});

    const scale: u8 = 0b11010101;
    const scale_vec = e8m0_to_splat_f32_vec(block_length, scale);
    std.debug.print("{any}\n", .{scale_vec});
}

const fp4_to_f32_decode_table = [_]f32{
    0.0,  0.5,  1.0,  1.5,
    2.0,  3.0,  4.0,  6.0,
    0.0,  -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
};

fn u8_to_fp4_vec(comptime block_length: usize, block_buf: [block_length / 2]u8) @Vector(block_length, u4) {
    // Each byte produces two 4-bit nibbles
    var nibbles: [block_length]u4 = undefined;

    for (block_buf, 0..) |b, i| {
        nibbles[i * 2] = @intCast(b >> 4); // upper nibble
        nibbles[i * 2 + 1] = @intCast(b & 0xF); // lower nibble
    }

    const fp4: @Vector(block_length, u4) = nibbles;
    return fp4;
}

fn fp4_to_f32_vec(comptime block_length: usize, fp4: @Vector(block_length, u4)) @Vector(block_length, f32) {
    var result: @Vector(block_length, f32) = undefined;
    inline for (0..block_length) |i| {
        result[i] = fp4_to_f32_decode_table[fp4[i]];
    }
    return result;
}

pub fn e8m0_to_splat_f32_vec(comptime block_length: usize, x: u8) @Vector(block_length, f32) {
    const x_f32: f32 = @floatFromInt(x);
    const bias: f32 = 127.0;
    const result_f32 = std.math.pow(f32, 2.0, x_f32 - bias);
    return @splat(result_f32);
}

// - model.layers.0.mlp.experts.down_proj_blocks: { 32, 32, 2, 16 } U8
// - model.layers.0.mlp.experts.down_proj_scales: { 32, 32, 2 } U8
