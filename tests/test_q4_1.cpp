/// Unit tests for the llama.cpp Q4_1 block decoder (`pack_q4`).
///
/// The tests build blocks the same way llama.cpp's quantize_row_q4_1_ref does
/// (4-bit quants → 16 nibble bytes, no mask), then check that pack_q4 recovers
/// `d * q + m` for every lane, including:
///   - the low/high split of a block (values j and j + 16 share byte j),
///   - every quant value 0..15 in both nibble halves,
///   - f32 and f16 destinations,
///   - a partial trailing block and multiple blocks with different d/m,
///   - an unaligned source pointer.

#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_X86_64 / _AARCH64
#include "common/test_harness.hpp"

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

/// Encode 32 quants (0..15) into one block, exactly as llama.cpp does.
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

/// llama.cpp's dequantize_row_q4_1, transliterated: an independent check of
/// the nibble order (value j from the low nibble, j + 16 from the high one).
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

/// Fill a byte buffer with @p nblocks encoded blocks, returning the expected
/// logical values alongside so a test can compare element by element.
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

/// A row-major matrix of Q4_1 blocks plus its logical f32 values.
///
/// Rows are padded to whole blocks, exactly like data_type_row_bytes() and the
/// dense TensorView pitch: `blocks_per_row * 20` bytes per row.
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

    // And the transliterated ggml loop agrees block by block.
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
