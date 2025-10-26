const std = @import("std");
const haystack_buf = @embedFile("test.txt");

pub fn main() !void {
    // Simple indexOf implementation
    const needle = 'o';
    const index_from_scratch = indexOf(haystack_buf, needle);
    std.debug.print("Index of 'o': {any}\n", .{index_from_scratch});

    const index_from_std = std.mem.indexOfScalar(u8, haystack_buf, needle);
    std.debug.print("Index of 'o': {any}\n", .{index_from_std});

    // Vector tests

    const vector_len = 8;
    const vector_needles: @Vector(vector_len, u8) = @splat('o');
    std.debug.print("Vector of 'o': {any}\n", .{vector_needles});

    const haystack = "Hello Jo";
    const vector_haystack: @Vector(vector_len, u8) = haystack.*;
    std.debug.print("Vector of 'Hello Jo': {any}\n", .{vector_haystack});

    const matches = vector_haystack == vector_needles;
    std.debug.print("Matches: {any}\n", .{matches});

    const index = std.simd.firstTrue(matches);
    std.debug.print("Index of 'o' (SIMD test, short string): {any}\n", .{index});

    // SIMD indexOf implementation
    const index_from_simd = firstIndexOfSimd(haystack_buf, needle);
    std.debug.print("Index of 'o' (SIMD): {any}\n", .{index_from_simd});
}

fn indexOf(haystack: []const u8, needle: u8) ?usize {
    for (haystack, 0..) |c, i| {
        if (c == needle) return i;
    }
    return null;
}

fn firstIndexOfSimd(haystack: []const u8, needle: u8) ?usize {
    const vector_len = 8;
    const vector_needles: @Vector(vector_len, u8) = @splat(@as(u8, needle));
    const indexes = std.simd.iota(u8, vector_len);
    const nulls: @Vector(vector_len, u8) = @splat(@as(u8, 255));

    var pos: usize = 0;
    var left = haystack.len;
    while (left > 0) {
        if (left < vector_len) {
            return std.mem.indexOfScalarPos(u8, haystack, pos, needle);
        }

        const h: @Vector(vector_len, u8) = haystack[pos..][0..vector_len].*;
        const matches = h == vector_needles;

        if (@reduce(.Or, matches)) {
            const result = @select(u8, matches, indexes, nulls);

            return @reduce(.Min, result) + pos;
        }

        pos += vector_len;
        left -= vector_len;
    }
    return null;
}
