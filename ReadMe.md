# Zig MXFP4 Dequantizer

Zig library for dequantizing MXFP4 tensors from SafeTensors files.

## Main features

- All MXFP4 tensors in the file are identified and dequantized,
- The decompression is done on the fly in streaming fashion,
- The MFP4 decoding is done using SIMD instructions,
- The library exposes the modern `std.Io.Reader` interface from Zig `0.15.1`.

## Quick Start

See `example.zig` for a basic usage example with a provided test file.

## Core components

- **`tensorReaders.zig`**: Main entry point, initializes and provides the full
  list of readers for a given file,
- **`tensorReader.zig`**: Streaming tensor reader implementation,
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

## References

Here are the external links to the related specs:

- [OCP MXFP4 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Safetensors format details](https://huggingface.co/docs/safetensors/index)

## Open questions

- How to profile for performance?
- Is there a more efficient way to handle load the fp4 values into the SIMD
  vectors?
- Are there memory-handling subtleties that can be improved?
