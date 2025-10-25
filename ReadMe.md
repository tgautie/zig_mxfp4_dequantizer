# Zig MXFP4 Dequantizer

Zig library for dequantizing MXFP4 tensors from SafeTensors files.

## Main features

- The decompression is done on the fly in streaming fashion
- The decoding is done using SIMD instructions
- The library exposes the modern `std.Io.Reader` interface from Zig `0.15.1`.

## Quick Start

See `example.zig` for a basic usage example with a provided test safetensors
file.

## Core components

- **`tensorReaders.zig`**: Main entry point, which initializes and provides the
  full list of readers for a given file
- **`tensorReader.zig`**: Streaming tensor reader implementation
- **`dequantization.zig`**: Core MXFP4 dequantization logic with SIMD
  instructions
- **`safetensors.zig`**: SafeTensors file format parser
- **`mxfp4Config.zig`**: MXFP4 tensor configuration extraction

## MXFP4 details

MXFP4 (Microscaling Format Point 4) is a floating point format that uses 4.25
bits to encode each value, by Each block consists of 16 bytes representing 32
fp4 values, along with a shared scale factor per block.

## References

Here are the external links to the related specs:

- [OCP MXFP4 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Safetensors format details](https://huggingface.co/docs/safetensors/index)
