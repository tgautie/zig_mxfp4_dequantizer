const std = @import("std");

pub fn main() void {
    // Multiplying f32 vectors
    const a: @Vector(4, f32) = .{ 1.0, 2.0, 3.0, 4.0 };
    const b: @Vector(4, f32) = .{ 5.0, 6.0, 7.0, 8.0 };
    const result = multiply(f32, a, b);
    std.debug.print("Result: {any}\n", .{result});

    // Multiplying u8 vectors
    const a_u8: @Vector(4, u8) = .{ 1, 2, 3, 4 };
    const b_u8: @Vector(4, u8) = .{ 5, 6, 7, 8 };
    const result_u8 = multiply(u8, a_u8, b_u8);
    std.debug.print("Result: {any}\n", .{result_u8});
}

fn multiply(comptime T: type, a: @Vector(4, T), b: @Vector(4, T)) @Vector(4, T) {
    return a * b;
}
