#pragma once
/// @file bench_knobs.hpp
/// @brief Experiment-only switches for the test and benchmark binaries.
///
/// A bench stores to these atomics directly and the decision point in the
/// backend does one relaxed load. Nothing here is an application-facing tuning
/// API: the values are process-wide, they default to the automatic route, and
/// the arms they select are only the ones a bench compares. The header lives
/// under src/ so it is not part of the installed interface.

#include <atomic>

namespace nnops::detail {

/// LinearAttention route: 0 automatic, -1 force the per-token loop, 1 force the
/// chunked (WY) route.
inline std::atomic<int> linear_attention_route{0};

/// Spatial pooling: 0 automatic (the straight-line 3x3 route when it is
/// eligible), 1 force the rolled window walk.
inline std::atomic<int> pooling_3x3_roll{0};

/// CUDA GEMM split-K: -1 automatic, 0 disable, N > 1 force N splits (clamped by
/// the same rules as the auto choice).
inline std::atomic<int> cuda_splitk{-1};

/// CUDA recurrent bidirectional: 0 runs the two directions concurrently on a
/// second stream, 1 keeps them on the caller's stream in sequence. A bench
/// compares the arms; production leaves the default.
inline std::atomic<int> cuda_rnn_serial{0};

/// CUDA f16 narrow-m preference: -1 automatic, 0 keeps the tiled engine first,
/// 1 hands a product of at most sixteen rows (A row-major, B `[k][n]`) to the
/// FFMA streaming loop before the tiled engine is tried. The automatic rule
/// does that only when the tiled grid would leave the machine part empty.
inline std::atomic<int> cuda_narrow_m{-1};

/// CUDA narrow-m loop variant: 0 is the production pick (f16 keeps 32 B rows in
/// flight, f32 16), 1 the 4-column / 16-row group, 2 the 2-column / 16-row one,
/// 3 the 2-column / 32-row one. A bench compares them.
inline std::atomic<int> cuda_narrow_var{0};

/// CUDA Conv2D's 1x1 route: -1 automatic, 0 keep the implicit-GEMM conv kernel,
/// 1 hand the unit-stride 1x1 to the dense GEMM.
inline std::atomic<int> cuda_conv_pw{-1};

/// CUDA Conv2D's 3x3 route: -1 automatic (the band-staged kernel wherever its
/// shape gate holds), 0 keep the tap-by-tap gather, 1 force the staged band
/// (and fall back to the gather when the gate declines).
inline std::atomic<int> cuda_conv_direct{-1};

/// CUDA Conv2D's 1x1 filter pose, on the dense-GEMM route: -1 automatic (the
/// transposed copy where the short-K window holds), 0 keep the HWIO filter as
/// the engine's n-major B, 1 force the transposed pose.
inline std::atomic<int> cuda_conv_pw_pose{-1};

/// CUDA Conv2D f16 warp tile: 0 is the production 32x16 tile (eight warps, three
/// blocks a SM), 1 the 16x16 one (sixteen warps, two blocks) — the same CTA of
/// output split over twice the warps, which halves the registers an accumulator
/// fragment costs.
inline std::atomic<int> cuda_conv_tile{0};

/// CPU int8: 1 hides NEON i8mm (SMMLA) so the SDOT route is taken. The feature
/// scan runs once, so set this before the first CpuFeatures::get().
inline std::atomic<int> disable_i8mm{0};

/// CPU SME: 1 hides the SME GEMM route so the NEON routes are taken. Unlike the
/// feature scan this is read on every availability query, so a bench or a test
/// can flip it between calls to compare the two engines on one binary.
inline std::atomic<int> disable_sme{0};

}  // namespace nnops::detail
