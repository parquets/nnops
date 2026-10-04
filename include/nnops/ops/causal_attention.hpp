#pragma once
/// @file causal_attention.hpp
/// @brief CausalAttention operator — causal self-attention with KV-cache.
///
/// Designed for autoregressive LLM decoding with external KV-cache storage.
/// Supports chunk prefill (Sq >= 1), block-level KV-cache (PagedAttention),
/// and multi-dtype cache (f32/f16/bf16/s8/u8).
///
/// The operator is stateless: all cache memory is owned by the caller.
/// Intermediate scores are allocated internally from the CPU memory pool.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the CausalAttention operator.
///
/// Static configuration set at operator creation time. Dynamic per-step
/// values (cache_position, cache_length) are passed as tensor inputs.
struct CausalAttentionAttributes {
    /// Number of attention heads.
    int64_t num_heads = 8;

    /// Maximum KV-cache sequence length (total capacity).
    /// Used for bounds validation.
    /// In block mode: max_cache_seq_len = num_blocks * block_size.
    int64_t max_cache_seq_len = 0;

    /// Maximum chunk size (Sq dimension of Q/K_new/V_new).
    /// Controls peak scratch memory. Set to 1 for decode-only,
    /// or to the desired chunk size (e.g. 512) for chunk prefill support.
    int64_t max_chunk_size = 1;

    /// KV-cache block size for PagedAttention-style block management.
    /// 0 = contiguous mode (cache layout: [B, H, max_len, D]).
    /// >0 = block mode (cache layout: [num_blocks, H, block_size, D]).
    int64_t block_size = 0;

    /// Scaling factor for QK^T before softmax.
    /// If 0, auto-computed as 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Softmax temperature (T > 0). T=1.0 is standard softmax.
    float temperature = 1.0f;

    /// Number of key-value head groups for GQA (Grouped Query Attention).
    /// When 0 (default), equals num_heads (standard MHA).
    /// When < num_heads, KV heads are shared across query head groups.
    int64_t num_group = 0;

    /// Data type for KV-cache storage (f32/f16/bf16/s8/u8).
    DataType kv_cache_dtype = DataType::f32;

    /// If true, apply QK normalization (QK-norm) before softmax.
    bool qk_norm = false;
};

/// Causal self-attention with KV-cache (class-based API).
///
/// Computes one step of autoregressive attention:
///   1. Writes K_new / V_new into the KV-cache at cache_position
///   2. Computes Q @ K_cache^T (over past + current chunk)
///   3. Applies causal mask within the current chunk
///   4. Softmax + temperature scaling
///   5. Weighted sum over V_cache → output
///
/// Inputs (5-6):
///   inputs[0] = Q              [B, H, Sq, D]       query
///   inputs[1] = K_new          [B, H, Sq, D]       new key(s)
///   inputs[2] = V_new          [B, H, Sq, D]       new value(s)
///   inputs[3] = cache_position [1] int64           cache write start position
///   inputs[4] = cache_length   [1] int64           valid past cache entries
///   inputs[5] = block_table    [B, max_blocks] int32  (only when block_size > 0)
///
/// Outputs (3):
///   outputs[0] = output   [B, H, Sq, D]            attention result
///   outputs[1] = K_cache  [B,H,max_len,D] or [num_blocks,H,block_size,D]
///   outputs[2] = V_cache  same as K_cache
///
/// Scratch is pooled internally by the CPU kernel (getWorkspaceSize returns 0).
class CausalAttention : public OpBase {
public:
    static std::unique_ptr<CausalAttention> create(
        const CausalAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<CausalAttention> create(Backend backend = Backend::CPU) {
        return create(CausalAttentionAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::CausalAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const CausalAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in causal_attention.cpp (Pimpl pattern)
    ~CausalAttention();

private:
    CausalAttention(const CausalAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    CausalAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
