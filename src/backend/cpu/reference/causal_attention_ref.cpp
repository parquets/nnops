/// @file causal_attention_ref.cpp
/// @brief Scalar CPU reference implementation of CausalAttention with KV-cache.
///
/// Handles:
///   - Decode (Sq=1) and chunk prefill (Sq>=1)
///   - Contiguous and block-level (PagedAttention) KV-cache
///   - Multi-dtype cache: f32, f16, bf16, s8, u8
///   - PerBlock, PerTensor, PerChannel quantization for int8/uint8 cache

#include "nnops/ops/causal_attention.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/bf16.hpp"
#include "nnops/detail/half.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace nnops::backend::cpu::reference {

namespace {

// ============================================================
// Cache type conversion helpers
// ============================================================

/// Load a single element from cache storage and convert to float32.
inline float cache_load_f32(const void* cache_ptr, int64_t offset,
                            DataType cache_dtype,
                            const QuantParams& qp, int64_t qp_idx)
{
    switch (cache_dtype) {
    case DataType::f32: {
        return static_cast<const float*>(cache_ptr)[offset];
    }
    case DataType::f16: {
        uint16_t bits = static_cast<const uint16_t*>(cache_ptr)[offset];
        return half_to_float(half(bits));
    }
    case DataType::bf16: {
        uint16_t v = static_cast<const uint16_t*>(cache_ptr)[offset];
        return bf16_to_float(v);
    }
    case DataType::s8: {
        int8_t iv = static_cast<const int8_t*>(cache_ptr)[offset];
        float scale = (qp.granularity == QuantGranularity::PerTensor) ? qp.scale : qp.scale_data[qp_idx];
        int32_t zp  = (qp.granularity == QuantGranularity::PerTensor) ? qp.zero_point : qp.zero_point_data[qp_idx];
        return (static_cast<float>(iv) - static_cast<float>(zp)) * scale;
    }
    case DataType::u8: {
        uint8_t iv = static_cast<const uint8_t*>(cache_ptr)[offset];
        float scale = (qp.granularity == QuantGranularity::PerTensor) ? qp.scale : qp.scale_data[qp_idx];
        int32_t zp  = (qp.granularity == QuantGranularity::PerTensor) ? qp.zero_point : qp.zero_point_data[qp_idx];
        return (static_cast<float>(iv) - static_cast<float>(zp)) * scale;
    }
    default:
        return 0.0f;
    }
}

/// Write a float32 value to cache storage with appropriate conversion.
inline void cache_store_f32(void* cache_ptr, int64_t offset, float val,
                            DataType cache_dtype,
                            const QuantParams& qp, int64_t qp_idx)
{
    switch (cache_dtype) {
    case DataType::f32:
        static_cast<float*>(cache_ptr)[offset] = val;
        break;
    case DataType::f16: {
        static_cast<uint16_t*>(cache_ptr)[offset] = half_to_bits(float_to_half(val));
        break;
    }
    case DataType::bf16: {
        static_cast<uint16_t*>(cache_ptr)[offset] = float_to_bf16(val);
        break;
    }
    case DataType::s8: {
        float scale = (qp.granularity == QuantGranularity::PerTensor) ? qp.scale : qp.scale_data[qp_idx];
        int32_t zp  = (qp.granularity == QuantGranularity::PerTensor) ? qp.zero_point : qp.zero_point_data[qp_idx];
        float iv = std::round(val / scale) + static_cast<float>(zp);
        static_cast<int8_t*>(cache_ptr)[offset] =
            static_cast<int8_t>(std::clamp(iv, -128.0f, 127.0f));
        break;
    }
    case DataType::u8: {
        float scale = (qp.granularity == QuantGranularity::PerTensor) ? qp.scale : qp.scale_data[qp_idx];
        int32_t zp  = (qp.granularity == QuantGranularity::PerTensor) ? qp.zero_point : qp.zero_point_data[qp_idx];
        float iv = std::round(val / scale) + static_cast<float>(zp);
        static_cast<uint8_t*>(cache_ptr)[offset] =
            static_cast<uint8_t>(std::clamp(iv, 0.0f, 255.0f));
        break;
    }
    default:
        break;
    }
}

/// Get the quantization parameter index for a physical cache position.
/// PerBlock:  index = phys_block
/// PerChannel: index = channel (head index)
/// PerToken:   index = logical_position (row)
/// PerTensor:  index = 0 (unused)
inline int64_t qp_index(const QuantParams& qp, int64_t phys_block,
                        int64_t head, int64_t logical_pos)
{
    switch (qp.granularity) {
    case QuantGranularity::PerBlock:   return phys_block;
    case QuantGranularity::PerChannel: return head;
    case QuantGranularity::PerToken:   return logical_pos;
    default:                           return 0;
    }
}

// ============================================================
// Cache addressing (contiguous vs block mode)
// ============================================================

/// Resolve a logical cache position to a physical offset within the cache buffer.
/// Continuous mode (block_size == 0): phys_offset = b * HC * max_len * D + h * max_len * D + pos * D + d
/// Block mode (block_size > 0):     phys_block → row within block
inline int64_t cache_offset(int64_t b, int64_t h, int64_t logical_pos, int64_t d,
                            const TensorView& cache,
                            int64_t block_size,
                            const int32_t* block_table, int64_t max_blocks,
                            int64_t& out_phys_block)
{
    if (block_size == 0) {
        // Contiguous: [B, H, max_len, D]
        out_phys_block = 0;  // unused
        int64_t H_dim = cache.shape(1);
        int64_t max_len = cache.shape(2);
        int64_t D_dim = cache.shape(3);
        int64_t row_stride = cache.row_stride_elems();
        // Per-batch offset
        int64_t b_off = b * H_dim * max_len * D_dim;
        // Within batch: h * max_len * row_stride + pos * row_stride + d
        return b_off + h * max_len * D_dim + logical_pos * D_dim + d;
    } else {
        // Block mode: [num_blocks, H, block_size, D]
        int64_t logical_block = logical_pos / block_size;
        int64_t offset_in_block = logical_pos % block_size;
        out_phys_block = block_table[b * max_blocks + logical_block];
        int64_t H_dim = cache.shape(1);
        int64_t bs = cache.shape(2);   // block_size
        int64_t D_dim = cache.shape(3);
        return out_phys_block * H_dim * bs * D_dim
             + h * bs * D_dim
             + offset_in_block * D_dim
             + d;
    }
}

/// Compute the base offset for a full row in the cache (used for reading
/// entire rows of D elements efficiently).
inline int64_t cache_row_offset(int64_t b, int64_t h, int64_t logical_pos,
                                const TensorView& cache,
                                int64_t block_size,
                                const int32_t* block_table, int64_t max_blocks,
                                int64_t& out_phys_block)
{
    return cache_offset(b, h, logical_pos, 0, cache, block_size,
                        block_table, max_blocks, out_phys_block);
}

// ============================================================
// Element load/store helpers for compute tensors (Q, K_new, V_new, output)
// ============================================================

inline float load_f32(const void* ptr, int64_t offset, DataType dtype) {
    switch (dtype) {
    case DataType::f32: return static_cast<const float*>(ptr)[offset];
    case DataType::f16: return half_to_float(half(static_cast<const uint16_t*>(ptr)[offset]));
    default: return 0.0f;
    }
}

inline void store_f32(void* ptr, int64_t offset, float val, DataType dtype) {
    float* dst = static_cast<float*>(ptr);
    if (dtype == DataType::f32) {
        dst[offset] = val;
    } else {
        // f16 compute dtype
        uint16_t* dst16 = static_cast<uint16_t*>(ptr);
        dst16[offset] = half_to_bits(float_to_half(val));
    }
}

/// Softmax over a row, in-place, with max-subtraction for numerical stability.
inline void softmax_row(float* row, int64_t n, float inv_T) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        max_val = std::max(max_val, row[i]);
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        row[i] = std::exp((row[i] - max_val) * inv_T);
        sum += row[i];
    }

    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < n; ++i) {
        row[i] *= inv_sum;
    }
}

}  // anonymous namespace

// ============================================================
// Main kernel
// ============================================================

void causal_attention_ref(const CausalAttentionAttributes& attrs,
                           std::span<TensorView> outputs,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* workspace)
{
    const auto& Q      = inputs[0];
    const auto& K_new  = inputs[1];
    const auto& V_new  = inputs[2];
    int64_t cache_pos  = *inputs[3].ptr<int64_t>();
    int64_t cache_len  = *inputs[4].ptr<int64_t>();

    auto& output  = outputs[0];
    auto& K_cache = outputs[1];
    auto& V_cache = outputs[2];

    const int64_t B  = Q.shape(0);
    const int64_t H  = Q.shape(1);
    const int64_t Sq = Q.shape(2);
    const int64_t D  = Q.shape(3);
    const int64_t Sk = cache_len + Sq;

    const DataType comp_dtype  = Q.data_type();
    const DataType cache_dtype = K_cache.data_type();
    const auto& k_qp = K_cache.quant_params();
    const auto& v_qp = V_cache.quant_params();

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;
    const float inv_T = (attrs.temperature > 0.0f)
        ? (1.0f / attrs.temperature) : 1.0f;

    // Block table (only used in block mode)
    const bool block_mode = (attrs.block_size > 0);
    const int32_t* block_table = nullptr;
    int64_t max_blocks = 0;
    if (block_mode) {
        NNOPS_ASSERT(inputs.size() >= 6);
        block_table = inputs[5].ptr<int32_t>();
        max_blocks = inputs[5].shape(1);
    }

    // Raw data pointers
    const void* q_ptr    = Q.ptr<void>();
    const void* kn_ptr   = K_new.ptr<void>();
    const void* vn_ptr   = V_new.ptr<void>();
    void*       o_ptr    = output.ptr<void>();
    void*       kc_ptr   = K_cache.ptr<void>();
    void*       vc_ptr   = V_cache.ptr<void>();

    // Stride helpers for compute tensors
    const int64_t q_row_stride  = Q.row_stride_elems();
    const int64_t kn_row_stride = K_new.row_stride_elems();
    const int64_t vn_row_stride = V_new.row_stride_elems();
    const int64_t o_row_stride  = output.row_stride_elems();

    /// Get offset into a compute tensor for element (b, h, s, d).
    /// Layout [B, H, Sq, D]: row_index = (b * H + h) * Sq + s
    auto comp_offset = [&](const TensorView& t, int64_t b, int64_t h,
                           int64_t s, int64_t d, int64_t row_stride) -> int64_t {
        return ((b * H + h) * t.shape(2) + s) * row_stride + d;
    };

    // Per-(batch, head) work item
    auto process_head = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;

        float* scores = static_cast<float*>(workspace) + idx * Sk;

        // ---- Step 1-2: Write K_new / V_new into cache ----
        for (int64_t s = 0; s < Sq; ++s) {
            int64_t dst_pos = cache_pos + s;
            int64_t phys_block = 0;
            int64_t row_off = cache_row_offset(b, h, dst_pos, K_cache,
                                                attrs.block_size, block_table,
                                                max_blocks, phys_block);
            int64_t k_qp_idx = qp_index(k_qp, phys_block, h, dst_pos);
            int64_t v_qp_idx = qp_index(v_qp, phys_block, h, dst_pos);

            for (int64_t d = 0; d < D; ++d) {
                float kv = load_f32(kn_ptr, comp_offset(K_new, b, h, s, d, kn_row_stride), comp_dtype);
                float vv = load_f32(vn_ptr, comp_offset(V_new, b, h, s, d, vn_row_stride), comp_dtype);
                cache_store_f32(kc_ptr, row_off + d, kv, cache_dtype, k_qp, k_qp_idx);
                cache_store_f32(vc_ptr, row_off + d, vv, cache_dtype, v_qp, v_qp_idx);
            }
        }

        // ---- Step 3: Compute scores ----
        for (int64_t i = 0; i < Sq; ++i) {
            // Past cache keys (no causal mask)
            for (int64_t j = 0; j < cache_len; ++j) {
                float dot = 0.0f;
                int64_t phys_block = 0;
                int64_t k_row_off = cache_row_offset(b, h, j, K_cache,
                                                      attrs.block_size, block_table,
                                                      max_blocks, phys_block);
                int64_t k_qp_idx = qp_index(k_qp, phys_block, h, j);

                for (int64_t d = 0; d < D; ++d) {
                    float qv = load_f32(q_ptr, comp_offset(Q, b, h, i, d, q_row_stride), comp_dtype);
                    float kv = cache_load_f32(kc_ptr, k_row_off + d, cache_dtype, k_qp, k_qp_idx);
                    dot += qv * kv;
                }
                scores[i * Sk + j] = dot * scale;
            }

            // Current chunk keys (causal mask: j <= i)
            for (int64_t j = 0; j < Sq; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float qv = load_f32(q_ptr, comp_offset(Q, b, h, i, d, q_row_stride), comp_dtype);
                    float kv = load_f32(kn_ptr, comp_offset(K_new, b, h, j, d, kn_row_stride), comp_dtype);
                    dot += qv * kv;
                }
                dot *= scale;
                scores[i * Sk + cache_len + j] = (j <= i) ? dot
                    : -std::numeric_limits<float>::infinity();
            }
        }

        // ---- Step 4: Softmax (per query row) ----
        for (int64_t i = 0; i < Sq; ++i) {
            softmax_row(scores + i * Sk, Sk, inv_T);
        }

        // ---- Step 5: Weighted V sum ----
        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t d = 0; d < D; ++d) {
                float sum = 0.0f;

                // Past cache values
                for (int64_t j = 0; j < cache_len; ++j) {
                    float s = scores[i * Sk + j];
                    if (std::isinf(s) || std::isnan(s)) {
                        continue;
                    }
                    int64_t phys_block = 0;
                    int64_t v_row_off = cache_row_offset(b, h, j, V_cache,
                                                          attrs.block_size, block_table,
                                                          max_blocks, phys_block);
                    int64_t v_qp_idx = qp_index(v_qp, phys_block, h, j);
                    sum += s * cache_load_f32(vc_ptr, v_row_off + d, cache_dtype, v_qp, v_qp_idx);
                }

                // Current chunk values
                for (int64_t j = 0; j < Sq; ++j) {
                    float s = scores[i * Sk + cache_len + j];
                    if (std::isinf(s) || std::isnan(s)) {
                        continue;
                    }
                    sum += s * load_f32(vn_ptr, comp_offset(V_new, b, h, j, d, vn_row_stride), comp_dtype);
                }

                store_f32(o_ptr, comp_offset(output, b, h, i, d, o_row_stride),
                          sum, comp_dtype);
            }
        }
    };

    // Parallel dispatch over B * H heads
    const int64_t total = B * H;
    ctx.cpu.run(0, total,
            [&](int64_t idx) { process_head(idx); });
}

}  // namespace nnops::backend::cpu::reference
