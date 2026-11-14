/// @file linear_attention.cpp
/// @brief LinearAttention operator dispatch.
///
/// Gated delta-rule linear attention with an optional causal conv1d over the
/// packed Q | K | V channels. Inputs are Q, K, V, Gate [, conv_weights [, beta]],
/// f32 or f16. The key/key-query width is Dk and the value width Dv; they are
/// read from the tensors, so the plain Dk == Dv form needs no extra attributes.
/// num_kv_heads (H_k) defaults to num_heads; K is [B, H_k, S, Dk], V owns the
/// value heads and width [B, H_v, S, Dv], and Q is indexed by the value head
/// (H_q % H_v == 0).
///
/// The operator has two outputs, so the kernel takes the whole span: the
/// attention output [B, H_q, S, Dv] and the recurrent State [B, H_v, Dk, Dv],
/// which the kernel carries across the sequence. Optional tails arrive as empty
/// views with rank 0 -- absent, not malformed -- and conv_weights come shaped
/// [(H_q + H_k) * Dk + H_v * Dv, 1, K].

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

namespace nnops {

namespace backend::cpu {
    Status linear_attention_cpu(const LinearAttentionAttributes& attrs,
                              std::span<TensorView> outputs,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    Status linear_attention_cuda(const LinearAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}
#endif

struct LinearAttention::Impl {
    using KernelFn = Status (*)(const LinearAttentionAttributes&,
                              std::span<TensorView>,
                              std::span<const TensorView>,
                              const ComputeContext&,
                              void*);
    KernelFn kernel_fn = nullptr;
};
LinearAttention::~LinearAttention() = default;

namespace {
auto resolve_linear_attention_kernel(Backend backend) -> LinearAttention::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::linear_attention_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::linear_attention_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}
}  // anonymous namespace

NNOPS_OP_CREATE(LinearAttention, LinearAttentionAttributes)

LinearAttention::LinearAttention(const LinearAttentionAttributes& attrs,
                                   Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_linear_attention_kernel(backend);
}

std::vector<TensorDesc> LinearAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    NNOPS_ASSERT(inputs.size() >= 4);
    const auto& Q = inputs[0];
    const int64_t B  = Q.dims[0];
    const int64_t Dk = Q.dims[3];

    // V owns the value head count and width — the state's columns. Fall back to
    // the legacy attribute-derived geometry when V's descriptor is unavailable.
    const bool has_v = inputs.size() > 2 && inputs[2].rank == 4;
    const int64_t Hv = has_v ? inputs[2].dims[1]
                             : (attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads
                                                        : attrs_.num_heads);
    const int64_t Dv = has_v ? inputs[2].dims[3] : attrs_.head_dim;

    // The output keeps Q's batch, heads and sequence, at the value width.
    TensorDesc out_desc  = Q;
    out_desc.dims[3]     = Dv;
    TensorDesc state_desc;
    state_desc.rank   = 4;
    state_desc.dims   = {B, Hv, Dk, Dv};
    state_desc.dtype  = Q.dtype;
    state_desc.layout = Q.layout;

    return {out_desc, state_desc};
}

Status LinearAttention::validate(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    // Q, K, V, Gate [, conv_weights [, beta]]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 4, 6, 2));

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, dt)) {
        return Status::unsupported;
    }
    for (size_t i = 1; i < 4; ++i) {
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    if (outputs[0].dtype != dt || outputs[1].dtype != dt) {
        return Status::invalid_argument;
    }

    if (inputs[0].rank != 4) { return Status::invalid_argument; }
    const int64_t b  = inputs[0].dims[0];
    const int64_t hq = inputs[0].dims[1];
    const int64_t s  = inputs[0].dims[2];
    const int64_t dk = inputs[0].dims[3];
    const int64_t hk = attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads : hq;
    if (hq <= 0 || hk <= 0) { return Status::invalid_argument; }

    // K carries the key heads at the state's row width.
    {
        const auto& t = inputs[1];
        if (t.rank != 4 || t.dims[0] != b || t.dims[1] != hk ||
            t.dims[2] != s || t.dims[3] != dk) {
            return Status::invalid_argument;
        }
    }

    // V carries its own head count and width; the value head owns the state, so
    // those are the state's and the output's columns.
    const auto& v = inputs[2];
    if (v.rank != 4 || v.dims[0] != b || v.dims[2] != s) {
        return Status::invalid_argument;
    }
    const int64_t hv = v.dims[1];
    const int64_t dv = v.dims[3];
    if (hv <= 0 || dv <= 0 || hv % hk != 0) { return Status::invalid_argument; }
    // The query heads are indexed by the value head, so they must be a multiple
    // of H_v; the key-head-to-value-head repeat is the caller's.
    if (hq % hv != 0) { return Status::invalid_argument; }

    // Gate decays one state row per key channel, so it is per value head and Dk
    // wide; beta is one scalar per value head and token.
    {
        const auto& g = inputs[3];
        if (g.rank != 4 || g.dims[0] != b || g.dims[1] != hv ||
            g.dims[2] != s || g.dims[3] != dk) {
            return Status::invalid_argument;
        }
    }
    if (outputs[0].rank != 4 || outputs[0].dims[0] != b || outputs[0].dims[1] != hq ||
        outputs[0].dims[2] != s || outputs[0].dims[3] != dv) {
        return Status::invalid_argument;
    }
    if (outputs[1].rank != 4 || outputs[1].dims[0] != b ||
        outputs[1].dims[1] != hv || outputs[1].dims[2] != dk ||
        outputs[1].dims[3] != dv) {
        return Status::invalid_argument;
    }

    if (inputs.size() > 4 && inputs[4].rank != 0) {   // [(H_q + H_k)*Dk + H_v*Dv, 1, K]
        const auto& w = inputs[4];
        if (w.dtype != dt || w.rank != 3 ||
            w.dims[0] != (hq + hk) * dk + hv * dv || w.dims[1] != 1 ||
            w.dims[2] != attrs_.conv_kernel_size) {
            return Status::invalid_argument;
        }
    }
    if (inputs.size() > 5 && inputs[5].rank != 0) {   // beta: [B, H_v, S]
        const auto& beta = inputs[5];
        if (beta.dtype != dt || beta.rank != 3 || beta.dims[0] != b ||
            beta.dims[1] != hv || beta.dims[2] != s) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status LinearAttention::compute(std::span<TensorView> outputs,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 4, 6, 2));
    for (size_t i = 0; i < 4; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty() || outputs[1].is_empty()) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs, inputs, ctx, workspace);
}

}  // namespace nnops
