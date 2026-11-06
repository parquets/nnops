/// Unit tests for the llama.cpp Q4_1 block decoder (`pack_q4`).
///
/// Blocks are built in llama.cpp's layout — 16 nibble bytes, value j in the low
/// nibble of byte j and value j + 16 in the high — then `pack_q4` must recover
/// `d * q + m` for every lane. The cases cover every quant value 0..15 in both
/// nibble halves, f32 and f16 destinations, a partial trailing block, multiple
/// blocks with different d/m, and an unaligned source pointer.

#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_X86_64 / _AARCH64
#include "common/test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/pack_q4_1.hpp"
#endif

using namespace nnops;

namespace {

/// A block whose logical 4-bit quants are known, plus the f16-rounded d/m.
struct BuiltBlock {
    q4_1_block blk{};
    float d = 0.0f;  ///< d after its f16 round-trip
    float m = 0.0f;  ///< m after its f16 round-trip
};

/// Encode 32 quants (0..15) into one block in llama.cpp's nibble order.
BuiltBlock make_block(const int q[32], float d, float m) {
    BuiltBlock b;
    b.d = half_to_float(float_to_half(d));
    b.m = half_to_float(float_to_half(m));
    b.blk.d = backend::cpu::half_to_bits(float_to_half(d));
    b.blk.m = backend::cpu::half_to_bits(float_to_half(m));

    for (int j = 0; j < 16; ++j) {
        const int q0 = q[j];
        const int q1 = q[j + 16];
        b.blk.qs[j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
    return b;
}

/// llama.cpp's quantize_row_q4_1_ref, transliterated, for whole blocks only.
std::vector<uint8_t> reference_quantize(const float* x, int64_t k) {
    const int nb = static_cast<int>(k / q4_1_block_elems);
    std::vector<uint8_t> out(static_cast<size_t>(nb) * q4_1_block_bytes);

    for (int i = 0; i < nb; ++i) {
        float vmin = 1e30f;
        float vmax = -1e30f;
        for (int j = 0; j < q4_1_block_elems; ++j) {
            const float v = x[i * q4_1_block_elems + j];
            if (v < vmin) { vmin = v; }
            if (v > vmax) { vmax = v; }
        }
        const float d = (vmax - vmin) / 15.0f;
        const float id = d ? 1.0f / d : 0.0f;

        uint8_t* blk = out.data() + static_cast<size_t>(i) * q4_1_block_bytes;
        const uint16_t d16 = backend::cpu::half_to_bits(float_to_half(d));
        const uint16_t m16 = backend::cpu::half_to_bits(float_to_half(vmin));
        std::memcpy(blk + 0, &d16, sizeof(d16));
        std::memcpy(blk + 2, &m16, sizeof(m16));

        for (int j = 0; j < q4_1_block_elems / 2; ++j) {
            const float x0 = (x[i * q4_1_block_elems + j] - vmin) * id;
            const float x1 = (x[i * q4_1_block_elems + q4_1_block_elems / 2 + j] - vmin) * id;
            const int q0 = std::min(15, static_cast<int>(x0 + 0.5f));
            const int q1 = std::min(15, static_cast<int>(x1 + 0.5f));
            blk[4 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
    return out;
}

/// llama.cpp's dequantize_row_q4_1, transliterated as an independent check of the nibble order.
void reference_decode(const q4_1_block& blk, float* y) {
    const float d = half_to_float(backend::cpu::half_from_bits(blk.d));
    const float m = half_to_float(backend::cpu::half_from_bits(blk.m));

    for (int j = 0; j < 16; ++j) {
        const float x0 = static_cast<float>(blk.qs[j] & 0x0F);
        const float x1 = static_cast<float>(blk.qs[j] >> 4);

        y[j]      = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

/// Encode @p nblocks blocks and return the expected logical values alongside.
std::vector<uint8_t> build_blocks(int nblocks, std::vector<float>& expected) {
    std::vector<uint8_t> bytes(static_cast<size_t>(nblocks) * q4_1_block_bytes);
    expected.clear();
    expected.reserve(static_cast<size_t>(nblocks) * q4_1_block_elems);

    for (int b = 0; b < nblocks; ++b) {
        int q[q4_1_block_elems];
        for (int j = 0; j < q4_1_block_elems; ++j) {
            // Every 4-bit value, both nibble halves.
            q[j] = (j + b * 3) & 0x0F;
        }
        q[0] = 0;
        q[1] = 15;

        // d/m chosen to survive the f16 round-trip exactly (powers of two and
        // multiples of 1/16 in range), so the expectations are exact in f32.
        const float d = 0.25f * static_cast<float>(b + 1);
        const float m = -1.5f + static_cast<float>(b);

        const BuiltBlock bb = make_block(q, d, m);
        for (int j = 0; j < q4_1_block_elems; ++j) {
            expected.push_back(bb.d * static_cast<float>(q[j]) + bb.m);
        }
        std::memcpy(bytes.data() + static_cast<size_t>(b) * q4_1_block_bytes,
                    &bb.blk, q4_1_block_bytes);
    }
    return bytes;
}

/// A row-major matrix of Q4_1 blocks plus its decoded f32 values.
struct Q4Matrix {
    std::vector<uint8_t> bytes;
    std::vector<float>   decoded;  ///< rows × cols, row-major
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
};

Q4Matrix build_q4_matrix(int64_t rows, int64_t cols) {
    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.blocks_per_row = (cols + q4_1_block_elems - 1) / q4_1_block_elems;
    m.bytes.assign(static_cast<size_t>(rows * m.blocks_per_row
                                       * static_cast<int64_t>(q4_1_block_bytes)), 0);
    m.decoded.assign(static_cast<size_t>(rows * cols), 0.0f);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < m.blocks_per_row; ++b) {
            int q[q4_1_block_elems];
            for (int j = 0; j < q4_1_block_elems; ++j) {
                q[j] = static_cast<int>((r * 13 + b * 7 + j * 5) & 0x0F);
            }
            // Exact in f16 so the decoded values are exact in f32 and f16.
            const float d = 0.25f * static_cast<float>(1 + (r + b) % 4);
            const float mmin = -2.0f + 0.5f * static_cast<float>((r + 2 * b) % 5);

            const BuiltBlock bb = make_block(q, d, mmin);
            std::memcpy(m.bytes.data()
                            + static_cast<size_t>((r * m.blocks_per_row + b)
                                                  * static_cast<int64_t>(q4_1_block_bytes)),
                        &bb.blk, q4_1_block_bytes);

            const int64_t base = r * cols + b * q4_1_block_elems;
            const int n = static_cast<int>(
                std::min<int64_t>(q4_1_block_elems, cols - b * q4_1_block_elems));
            for (int j = 0; j < n; ++j) {
                m.decoded[static_cast<size_t>(base + j)] =
                    bb.d * static_cast<float>(q[j]) + bb.m;
            }
        }
    }
    return m;
}

}  // namespace

NNOPS_TEST(q4_1_block_layout) {
    NNOPS_EXPECT_EQ(static_cast<int>(sizeof(q4_1_block)), 20);
    NNOPS_EXPECT_EQ(q4_1_block_elems, 32);
    NNOPS_EXPECT_EQ(static_cast<int>(q4_1_block_bytes), 20);
}

NNOPS_TEST(q4_1_pack_single_block_f32) {
    int q[q4_1_block_elems];
    for (int j = 0; j < q4_1_block_elems; ++j) {
        q[j] = j & 0x0F;  // every quant value, both nibble halves
    }
    const BuiltBlock bb = make_block(q, 0.25f, -1.5f);

    float out[q4_1_block_elems];
    pack_q4(&bb.blk, q4_1_block_elems, out);

    for (int j = 0; j < q4_1_block_elems; ++j) {
        const float want = bb.d * static_cast<float>(q[j]) + bb.m;
        NNOPS_EXPECT_EQ(out[j], want);
    }
}

NNOPS_TEST(q4_1_pack_matches_llama_reference) {
    std::vector<float> expected;
    const std::vector<uint8_t> bytes = build_blocks(4, expected);

    float out[4 * q4_1_block_elems];
    pack_q4(bytes.data(), 4 * q4_1_block_elems, out);

    for (size_t i = 0; i < expected.size(); ++i) {
        NNOPS_EXPECT_EQ(out[i], expected[i]);
    }

    for (int b = 0; b < 4; ++b) {
        q4_1_block blk;
        std::memcpy(&blk, bytes.data() + static_cast<size_t>(b) * q4_1_block_bytes,
                    sizeof(blk));
        float ref[q4_1_block_elems];
        reference_decode(blk, ref);
        for (int j = 0; j < q4_1_block_elems; ++j) {
            NNOPS_EXPECT_EQ(out[b * q4_1_block_elems + j], ref[j]);
        }
    }
}

NNOPS_TEST(q4_1_pack_f16) {
    std::vector<float> expected;
    const std::vector<uint8_t> bytes = build_blocks(3, expected);

    std::vector<half> out(expected.size());
    pack_q4(bytes.data(), static_cast<int64_t>(expected.size()), out.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        // f16 output rounds once at the end; with these exact f32 values the
        // round-trip is deterministic, so compare against the same conversion.
        NNOPS_EXPECT_EQ(half_to_float(out[i]), half_to_float(float_to_half(expected[i])));
    }
}

NNOPS_TEST(q4_1_pack_partial_tail) {
    std::vector<float> expected;
    const std::vector<uint8_t> bytes = build_blocks(3, expected);

    // 2 blocks + 7 values of the third: the tail block is decoded only up to
    // count, and only 3 blocks are present (not 4).
    for (int count : {1, 7, 31, 32, 33, 63, 64, 71}) {
        std::vector<float> out(static_cast<size_t>(count), -123.0f);
        pack_q4(bytes.data(), count, out.data());
        for (int i = 0; i < count; ++i) {
            NNOPS_EXPECT_EQ(out[static_cast<size_t>(i)], expected[static_cast<size_t>(i)]);
        }
    }
}

NNOPS_TEST(q4_1_pack_zero_count) {
    const uint8_t dummy[20] = {};
    float out[1] = {42.0f};
    pack_q4(dummy, 0, out);
    NNOPS_EXPECT_EQ(out[0], 42.0f);  // untouched

    std::vector<float> none;  // no destination work for zero values
    pack_q4(dummy, -5, none.data());
}

NNOPS_TEST(q4_1_pack_unaligned_source) {
    std::vector<float> expected;
    const std::vector<uint8_t> bytes = build_blocks(2, expected);

    // A block stream need not be aligned: pack_q4 reads through memcpy.
    std::vector<uint8_t> shifted(bytes.size() + 1);
    std::memcpy(shifted.data() + 1, bytes.data(), bytes.size());

    std::vector<float> out(expected.size());
    pack_q4(shifted.data() + 1, static_cast<int64_t>(expected.size()), out.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        NNOPS_EXPECT_EQ(out[i], expected[i]);
    }
}

// ============================================================
// quantize_q4 — llama.cpp's quantize_row_q4_1_ref
// ============================================================

/// Values in [-1, 1) at full f32 spread, so d lands mid-range and no block is
/// degenerate unless a case says so.
std::vector<float> random_values(int64_t count, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(count));
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / 8388608.0f - 1.0f;
    }
    return v;
}

NNOPS_TEST(q4_1_quantize_matches_llama_reference) {
    for (int64_t count : {32, 64, 96, 320, 4096}) {
        const std::vector<float> x = random_values(count, 12345u + count);
        std::vector<uint8_t> got(static_cast<size_t>(count / q4_1_block_elems)
                                 * q4_1_block_bytes);
        quantize_q4(x.data(), count, got.data());
        const std::vector<uint8_t> want = reference_quantize(x.data(), count);
        NNOPS_EXPECT_EQ(got.size(), want.size());
        NNOPS_EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size()), 0);
    }
}

NNOPS_TEST(q4_1_quantize_roundtrip) {
    const int64_t count = 32 * 17;
    const std::vector<float> x = random_values(count, 99u);
    std::vector<uint8_t> blocks(static_cast<size_t>(count / 32) * q4_1_block_bytes);
    quantize_q4(x.data(), count, blocks.data());

    std::vector<float> back(static_cast<size_t>(count));
    pack_q4(blocks.data(), count, back.data());

    // Every value lands within half a quant step, with the stored d (already
    // rounded to f16) as the step and a sliver for the d/m f16 rounding.
    for (int64_t b = 0; b < count / 32; ++b) {
        q4_1_block blk;
        std::memcpy(&blk, blocks.data() + static_cast<size_t>(b) * q4_1_block_bytes,
                    sizeof(blk));
        const float d = half_to_float(half_from_bits(blk.d));
        for (int j = 0; j < 32; ++j) {
            const int64_t i = b * 32 + j;
            const float err = std::fabs(back[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]);
            NNOPS_EXPECT_TRUE(err <= 0.5f * d + 1e-3f);
        }
    }
}

NNOPS_TEST(q4_1_quantize_constant_block) {
    // All values equal: d collapses to 0, m carries the value, every quant is 0,
    // and the round-trip is exact up to m's f16 rounding.
    std::vector<float> x(64, 0.375f);  // exact in f16
    std::vector<uint8_t> blocks(2 * q4_1_block_bytes);
    quantize_q4(x.data(), static_cast<int64_t>(x.size()), blocks.data());

    for (int b = 0; b < 2; ++b) {
        q4_1_block blk;
        std::memcpy(&blk, blocks.data() + static_cast<size_t>(b) * q4_1_block_bytes,
                    sizeof(blk));
        NNOPS_EXPECT_EQ(half_to_float(half_from_bits(blk.d)), 0.0f);
        NNOPS_EXPECT_EQ(half_to_float(half_from_bits(blk.m)), 0.375f);
        for (int j = 0; j < 16; ++j) {
            NNOPS_EXPECT_EQ(static_cast<int>(blk.qs[j]), 0);
        }
    }

    std::vector<float> back(64);
    pack_q4(blocks.data(), 64, back.data());
    for (float v : back) {
        NNOPS_EXPECT_EQ(v, 0.375f);
    }
}

NNOPS_TEST(q4_1_quantize_partial_tail) {
    const int64_t count = 37;  // one whole block + 5 values
    const std::vector<float> x = random_values(count, 7u);
    std::vector<uint8_t> blocks(2 * q4_1_block_bytes, 0xAA);
    quantize_q4(x.data(), count, blocks.data());

    // The tail's d and m come from its own five values, not the first block's.
    q4_1_block blk;
    std::memcpy(&blk, blocks.data() + q4_1_block_bytes, sizeof(blk));
    float lo = x[32], hi = x[32];
    for (int j = 1; j < 5; ++j) {
        const float v = x[static_cast<size_t>(32 + j)];
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
    }
    NNOPS_EXPECT_EQ(half_to_float(half_from_bits(blk.m)),
                    half_to_float(float_to_half(lo)));
    const float d = half_to_float(half_from_bits(blk.d));
    NNOPS_EXPECT_TRUE(std::fabs(d - half_to_float(float_to_half((hi - lo) / 15.0f))) <= 1e-6f);
    // Values 5..15 and every high nibble are unwritten, so the decoder reads
    // them as 0 and never looks at them.
    for (int j = 5; j < 16; ++j) {
        NNOPS_EXPECT_EQ(static_cast<int>(blk.qs[j]), 0);
    }

    std::vector<float> back(static_cast<size_t>(count));
    pack_q4(blocks.data(), count, back.data());

    q4_1_block first;
    std::memcpy(&first, blocks.data(), sizeof(first));
    const float d0 = half_to_float(half_from_bits(first.d));
    for (int64_t i = 0; i < count; ++i) {
        const float step = (i < 32) ? d0 : d;
        const float err = std::fabs(back[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]);
        NNOPS_EXPECT_TRUE(err <= 0.5f * step + 1e-3f);
    }
}

NNOPS_TEST(q4_1_quantize_f16_source) {
    // A f16 source quantizes to the same bytes as the f32 tensor it upconverts
    // to: the encoder does all its arithmetic in f32.
    const int64_t count = 128;
    const std::vector<float> x = random_values(count, 31u);
    std::vector<half> xh(static_cast<size_t>(count));
    std::vector<float> xf(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        xh[static_cast<size_t>(i)] = float_to_half(x[static_cast<size_t>(i)]);
        xf[static_cast<size_t>(i)] = half_to_float(xh[static_cast<size_t>(i)]);
    }

    std::vector<uint8_t> from_f32(static_cast<size_t>(count / 32) * q4_1_block_bytes);
    std::vector<uint8_t> from_f16(from_f32.size());
    quantize_q4(xf.data(), count, from_f32.data());
    quantize_q4(xh.data(), count, from_f16.data());
    NNOPS_EXPECT_EQ(std::memcmp(from_f32.data(), from_f16.data(), from_f32.size()), 0);
}

NNOPS_TEST(q4_1_quantize_unaligned_dest) {
    const int64_t count = 96;
    const std::vector<float> x = random_values(count, 5u);
    std::vector<uint8_t> aligned(3 * q4_1_block_bytes);
    std::vector<uint8_t> shifted(aligned.size() + 1, 0x5A);

    quantize_q4(x.data(), count, aligned.data());
    quantize_q4(x.data(), count, shifted.data() + 1);

    NNOPS_EXPECT_EQ(std::memcmp(shifted.data() + 1, aligned.data(), aligned.size()), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(shifted[0]), 0x5A);  // the pad byte is untouched
}

NNOPS_TEST(q4_1_unpack_block_helper) {
    int q[q4_1_block_elems];
    for (int j = 0; j < q4_1_block_elems; ++j) { q[j] = 15 - (j & 0x0F); }
    const BuiltBlock bb = make_block(q, 0.5f, 2.0f);

    float via_helper[q4_1_block_elems];
    q4_1_unpack_block(&bb.blk, via_helper);

    float via_pack[q4_1_block_elems];
    pack_q4(&bb.blk, q4_1_block_elems, via_pack);

    for (int j = 0; j < q4_1_block_elems; ++j) {
        NNOPS_EXPECT_EQ(via_helper[j], via_pack[j]);
        NNOPS_EXPECT_EQ(via_helper[j], bb.d * static_cast<float>(q[j]) + bb.m);
    }
}

// ============================================================
// x86_64 SIMD kernels
// ============================================================
#if defined(NNOPS_ARCH_X86_64)

using nnops::backend::cpu::x86_64::pack_q4_1_rhs;
using nnops::backend::cpu::x86_64::q4_1_nr_max;

NNOPS_TEST(q4_1_x86_pack_matches_scalar) {
    std::vector<float> expected;
    const std::vector<uint8_t> bytes = build_blocks(3, expected);

    for (int64_t count : {int64_t{1}, int64_t{7}, int64_t{31}, int64_t{32},
                          int64_t{33}, int64_t{63}, int64_t{64}, int64_t{71},
                          int64_t{96}}) {
        std::vector<float> simd(static_cast<size_t>(count), -999.0f);
        nnops::backend::cpu::x86_64::pack_q4(bytes.data(), count, simd.data());

        for (int64_t i = 0; i < count; ++i) {
            NNOPS_EXPECT_EQ(simd[static_cast<size_t>(i)],
                            expected[static_cast<size_t>(i)]);
        }
    }

    // f16 destination, including a partial last block.
    std::vector<half> simd16(static_cast<size_t>(70));
    nnops::backend::cpu::x86_64::pack_q4(bytes.data(), 70, simd16.data());
    for (int i = 0; i < 70; ++i) {
        NNOPS_EXPECT_EQ(half_to_float(simd16[static_cast<size_t>(i)]),
                        half_to_float(float_to_half(expected[static_cast<size_t>(i)])));
    }
}

/// Verify one packed-B tile: every panel lane for every k, including the
/// zero-filled lanes past a tail panel's live width.
template <class T>
void check_q4_1_panels(const Q4Matrix& m, bool trans,
                       int n_start, int nc, int k_start, int kc) {
    const int npanels = (nc + q4_1_nr_max - 1) / q4_1_nr_max;
    const int ldd = q4_1_nr_max * kc;  // contiguous panels are enough here
    std::vector<T> dst(static_cast<size_t>(npanels * ldd), T{});

    pack_q4_1_rhs<T>(trans, n_start, nc, k_start, kc,
                     dst.data(), ldd, m.bytes.data(), m.blocks_per_row, 1.0f);

    for (int p = 0; p < npanels; ++p) {
        const int n0 = n_start + p * q4_1_nr_max;
        const int width = std::min(q4_1_nr_max, nc - p * q4_1_nr_max);
        for (int k = 0; k < kc; ++k) {
            const T* row = dst.data() + static_cast<int64_t>(p * ldd + k * q4_1_nr_max);
            for (int j = 0; j < q4_1_nr_max; ++j) {
                float want = 0.0f;
                if (j < width) {
                    const int64_t kk = k_start + k;
                    const int64_t nn = n0 + j;
                    const int64_t idx = trans ? (nn * m.cols + kk)
                                              : (kk * m.cols + nn);
                    want = m.decoded[static_cast<size_t>(idx)];
                }
                if constexpr (std::is_same_v<T, float>) {
                    NNOPS_EXPECT_EQ(row[j], want);
                } else {
                    NNOPS_EXPECT_EQ(half_to_float(row[j]), half_to_float(float_to_half(want)));
                }
            }
        }
    }
}

NNOPS_TEST(q4_1_x86_rhs_panels) {
    // K = 70 (not block-aligned), N = 96. The matrix builder always stores
    // [physical rows][logical columns] with whole-block row padding, so the
    // transposed case is simply the [N][K] orientation of the same stream.
    constexpr int64_t K = 70;
    constexpr int64_t N = 96;

    const Q4Matrix b_kn = build_q4_matrix(K, N);  // B phys [K][N]
    const Q4Matrix b_nk = build_q4_matrix(N, K);  // B phys [N][K]

    struct Case { int n_start, nc, k_start, kc; };
    const Case cases[] = {
        {0, 96, 0, 70},   // full tile
        {5, 37, 3, 61},   // misaligned origin, K tail
        {16, 32, 32, 38}, // block-aligned n, K tail
        {33, 17, 1, 31},  // panel straddles blocks at both ends
        {1, 1, 0, 1},     // single value
        {48, 9, 64, 6},   // tail panel, short k
        {80, 16, 0, 64},  // panel starting inside the last block
    };

    for (const Case& c : cases) {
        check_q4_1_panels<float>(b_kn, false, c.n_start, c.nc, c.k_start, c.kc);
        check_q4_1_panels<float>(b_nk, true,  c.n_start, c.nc, c.k_start, c.kc);
        check_q4_1_panels<half>(b_kn, false, c.n_start, c.nc, c.k_start, c.kc);
        check_q4_1_panels<half>(b_nk, true,  c.n_start, c.nc, c.k_start, c.kc);
    }
}

#endif  // NNOPS_ARCH_X86_64
