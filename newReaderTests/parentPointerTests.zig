const std = @import("std");

const Struct = struct {
    foo: u8,
    field: u8,
};

pub fn main() void {
    const instance = Struct{
        .foo = 0x12,
        .field = 0x12,
    };
    // Get a pointer to the field of an instance of Struct
    const field_ptr = &instance.field;
    std.debug.print("field_ptr_int: {}\n", .{field_ptr});
    // Convert the pointer to an integer so that we can manipulate it
    const field_ptr_int = @intFromPtr(field_ptr);
    std.debug.print("field_ptr_int: {}\n", .{field_ptr_int});
    // Get the byte offset of the field from the start of its struct
    const field_offset = @offsetOf(Struct, "field");
    std.debug.print("field_offset: {}\n", .{field_offset});
    // Subtract the offset to get a pointer to the start of the 'parent' struct
    const parent_ptr_int = field_ptr_int - field_offset;
    std.debug.print("parent_ptr_int: {}\n", .{parent_ptr_int});
    // Convert the integer to a pointer to the 'parent' struct
    const parent_ptr: *Struct = @ptrFromInt(parent_ptr_int);
    std.debug.print("parent_ptr: {}\n", .{parent_ptr});
    std.debug.assert(parent_ptr == &instance);
}
