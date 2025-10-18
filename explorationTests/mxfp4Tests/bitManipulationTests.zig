const std = @import("std");

pub fn main() void {
    std.debug.print("{b}\n", .{0xFF << 4});
    std.debug.print("{b}\n", .{0xFF >> 4});
    std.debug.print("{b}\n", .{0xFF & 0xF0});
    std.debug.print("{b}\n", .{0xFF | 0x0F});
    std.debug.print("{b}\n", .{0xFF ^ 0x0F});
    std.debug.print("{b}\n", .{~@as(u8, 0x0F)});
}
