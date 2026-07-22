---
name: pitch-design
description: Pitch-based row stride model for TensorView — replaces multi-dimensional strides (2026-07-22)
metadata: 
  node_type: memory
  type: project
  originSessionId: 5974f579-b7c9-444a-a957-4b2b9fe094ad
  modified: 2026-07-22T13:08:58.009Z
---

# Pitch Design

TensorView uses a pitch-based memory model (like OpenCV `cv::Mat::step[0]`) instead of per-dimension strides.

**Why:** Simpler, matches onnxruntime/nn_compute patterns, and only the innermost "row" dimension typically has alignment padding. Multi-dimensional strides are overkill for our use case.

**How to apply:** Use `TensorView::pitch()` (bytes) and `TensorView::row_stride_elems()` (elements) for row-to-row navigation. Use `TensorView::stride_elems(dim)` for arbitrary axis stride computation.

## Model

- `pitch`: byte distance between consecutive rows, where a "row" is the innermost dimension.
- Elements within a row are always contiguous (byte stride = `elem_size`).
- Higher dimensions (C, N, D, ...) are densely packed — stride = inner_dim_size × inner_stride.

For NCHW float32: `pitch >= W * 4`. For contiguous data: `pitch == W * 4`.
For NCHWC8 uint8: `pitch >= W * 8`. With 32-byte alignment: `pitch = align_up(W*8, 32)`.

## API

| Method | Returns | Notes |
|---|---|---|
| `pitch()` | `int64_t` | Row pitch in bytes |
| `row_stride_elems()` | `int64_t` | `pitch / elem_size` (elements per row, ≥ last_dim) |
| `stride_elems(dim)` | `int64_t` | Element stride for any dimension (accounts for pitch) |

## Construction

```cpp
// Dense (contiguous): pitch = last_dim * elem_size
TensorView tv(shape, dtype, data);

// Explicit pitch (for aligned/padded data)
TensorView tv(shape, dtype, data, pitch);
```

## Reference

- nn_compute `TensorView::pitch` (`d:\vscode\nn_compute\include\nn_compute\common\data_info.hpp`)
- OpenCV `cv::Mat::step[0]`
