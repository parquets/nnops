#pragma once
/// @file attention_common.hpp
/// @brief Types the attention operators share.
///
/// GroupQueryAttention — which covers ordinary MHA as its H_kv == H case — and
/// MultiLatentAttention differ in what they project and how many key/value
/// heads they carry, not in how a head is laid out. Ordinary MHA is not a
/// separate operator: it is the degenerate grouped-query case. This header carries the one
/// choice that has to be made the same way by all of them.

#include <cstdint>

namespace nnops {

/// Storage order of the rank-4 per-head form.
///
/// The merged [B, S, H*D] form is unambiguous at rank 3. At rank 4 there are
/// two orders with the same shape signature, so the layout is named:
///
///   BHSD  [B, H, S, D]  — onnxruntime's cache-major BNSH; the default, and
///                          the one order the KV cache always uses.
///   BSHD  [B, S, H, D]  — HuggingFace's layout, where heads sit between the
///                          sequence and the head dimension.
///
/// Auto keeps the pre-existing behaviour: rank 3 is merged, rank 4 is BHSD.
enum class AttentionHeadLayout : int64_t {
    Auto = 0,
    BSHD = 1,
};

} // namespace nnops
