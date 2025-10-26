# Zig MXFP4 Dequantizer

Zig library for streaming dequantized MXFP4 tensors from Huggingface's
safetensors file format.

## Main overview

- On initialization, the module parses the provided safetensors file's header
  and identifies the MXFP4 tensors,
- The dequantization is done on the fly in streaming fashion,
- The dequantization uses SIMD instructions,
- The output provides a `reader`for each MXFP4 tensor, following the modern
  `std.Io.Reader` interface from Zig `0.15.1`.

## Quick Start

See `example.zig` for a basic usage example with a provided test file.

## Core components

- **`tensorReaders.zig`**: Main entry point, initializes the module and provides
  the full set of tensor readers for a given safetensors file,
- **`tensorReader.zig`**: Streaming MXFP4 tensor reader implementation,
- **`dequantization.zig`**: Core MXFP4 dequantization logic with SIMD
  instructions,
- **`safetensors.zig`**: SafeTensors file format parser,
- **`mxfp4Config.zig`**: MXFP4 tensor configuration extraction.

## MXFP4 details

MXFP4 (Microscaling FP4) is a floating point format that uses 4.25 bits to
encode tensor values.

The format consists of:

- blocks of 32 FP4 values,
- U8 scale factors that are shared by all values in a given block.

The bit layout is the following:

- S1E2M1 for the block values,
- S0E8M0 for the scale values.

[Official OCP MXFP4 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

## Safetensors format details

Safetensors is a Huggingface format that serializes tensor values in the
following way:

- First 8 bytes: size of the header in u64
- Header: metadata of all tensors with value offsets
- Rest of the file: raw tensor values

[Safetensors format details](https://huggingface.co/docs/safetensors/index)

## Open questions

- How to profile this system for performance?
- Is there a more efficient way to load the fp4 values into the SIMD vectors,
  i.e. with some vectorized table lookup? Right now this is done in a for loop
- Are there memory-handling subtleties that can be improved?
