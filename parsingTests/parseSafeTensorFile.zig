const std = @import("std");
const input = @embedFile("ones.safetensors");

const Field = struct {
    dtype: []const u8,
    shape: []usize,
    data_offsets: []usize,
};

const TestData = struct {
    attention: Field,
    embedding: Field,
};

pub fn main() !void {
    const header_size = std.mem.readInt(u64, input[0..8], .little);
    std.debug.print("Header size: {}\n", .{header_size});
    const header_buf = input[8 .. 8 + header_size];
    std.debug.print("Header: {s}\n", .{header_buf});

    const allocator = std.heap.page_allocator;
    const parsed_json = try std.json.parseFromSlice(TestData, allocator, header_buf, .{});
    defer parsed_json.deinit();

    const header_json = parsed_json.value;
    std.debug.print("attention dtype = {s}\n", .{header_json.attention.dtype});
    std.debug.print("attention shape = {any}\n", .{header_json.attention.shape});
    std.debug.print("attention offsets = {any}\n", .{header_json.attention.data_offsets});
    std.debug.print("embedding dtype = {s}\n", .{header_json.embedding.dtype});
    std.debug.print("embedding shape = {any}\n", .{header_json.embedding.shape});
    std.debug.print("embedding offsets = {any}\n", .{header_json.embedding.data_offsets});

    const attention_offsets = header_json.attention.data_offsets;
    const embedding_offsets = header_json.embedding.data_offsets;

    const attention_buf = input[8 + header_size + attention_offsets[0] .. 8 + header_size + attention_offsets[1]];
    const attention_values = std.mem.bytesAsSlice(f32, attention_buf);
    const embedding_buf = input[8 + header_size + embedding_offsets[0] .. 8 + header_size + embedding_offsets[1]];
    const embedding_values = std.mem.bytesAsSlice(f32, embedding_buf);

    std.debug.print("attention values = {any}\n", .{attention_values});
    std.debug.print("embedding values = {any}\n", .{embedding_values});
}
