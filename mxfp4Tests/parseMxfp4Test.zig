const std = @import("std");

pub fn main() void {
    const example_bit = 0b11001010;
    const block_buf: [16]u8 = [_]u8{ example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit, example_bit };

    // Each byte produces two 4-bit nibbles
    var nibbles: [32]u4 = undefined;

    for (block_buf, 0..) |b, i| {
        nibbles[i * 2] = @intCast(b >> 4); // upper nibble
        nibbles[i * 2 + 1] = @intCast(b & 0xF); // lower nibble
    }

    const vec: @Vector(32, u4) = nibbles;
    std.debug.print("{b}\n", .{vec});
}

// - model.layers.0.mlp.experts.down_proj_blocks: { 32, 32, 2, 16 } U8
// - model.layers.0.mlp.experts.down_proj_scales: { 32, 32, 2 } U8
