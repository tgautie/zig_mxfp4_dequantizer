const std = @import("std");

// Dequantizes a single MXFP4 block (16 bytes) into a vector of 32 float32 values from the u8 scale factor, using SIMD.
pub fn decodeBlock(scale: u8, block: [16]u8) [32]f32 {
    const block_vec = getBlockVec(block);
    const scale_vec = getScaleVec(scale);
    const decoded_block_vec = block_vec * scale_vec;
    const decoded_block: [32]f32 = decoded_block_vec;
    return decoded_block;
}

test decodeBlock {
    const scale: u8 = 128; // Scale factor corresponding to pow(2.0, (128 - 127)) = 2.0
    var block: [16]u8 = undefined;
    inline for (0..16) |i| {
        block[i] = 0x0B; // Block with repeating FP4 patterns corresponding to 0.0 and -1.5
    }

    const expected: [32]f32 = blk: {
        var arr: [32]f32 = undefined;
        inline for (0..16) |i| {
            arr[i * 2] = 0.0 * 2.0;
            arr[i * 2 + 1] = -1.5 * 2.0;
        }
        break :blk arr;
    };

    const actual = decodeBlock(scale, block);
    inline for (0..32) |i| {
        try std.testing.expectEqual(expected[i], actual[i]);
    }
}

// This table provides a mapping from the 4-bit "fp4" block values to their decoded float32 representations
// The quantization scheme is as follows:
// - The first bit is the sign bit S
// - The next 2 bits are the exponent E
// - The last bit is the mantissa M
// The decoded float32 value is obtained as follows:
// - If E>0, the value is  (-1)^S * 2^(E - 1) * (1+M/2)
// - If E=0 (subnormal case), the value is (-1)^S * (M/2)
const fp4_to_f32 = [16]f32{
    0.0,  0.5,  1.0,  1.5,
    2.0,  3.0,  4.0,  6.0,
    0.0,  -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
};

fn getBlockVec(block: [16]u8) @Vector(32, f32) {
    var result: @Vector(32, f32) = undefined;
    inline for (0..16) |i| {
        const b = block[i];
        result[i * 2] = fp4_to_f32[b >> 4];
        result[i * 2 + 1] = fp4_to_f32[b & 0x0F];
    }
    return result;
}

fn getScaleVec(x: u8) @Vector(32, f32) {
    const bias: i32 = 127;
    const exponent = @as(i32, x);
    const biased_exponent = @as(f32, @floatFromInt(exponent - bias));
    const scale_f32 = std.math.pow(f32, 2.0, biased_exponent);
    const scale_vec: @Vector(32, f32) = @splat(scale_f32);
    return scale_vec;
}
