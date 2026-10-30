/// @file linear_attention.cpp
/// @brief LinearAttention operator dispatch.
///
/// Gated delta-rule linear attention with an optional causal conv1d over the
/// packed Q | K | V channels. Inputs are Q, K, V, Gate [, conv_weights [, beta]],
/// f32 or f16; num_kv_heads defaults to num_heads, and under GQA (H % H_kv == 0)
/// K, V and Gate take the kv-head geometry [B, H_kv, S, D], beta [B, H_kv, S],
/// while the output stays [B, H, S, D].
///
/// The operator has two outputs, so the kernel takes the whole span: the
/// attention output and the recurrent State [B, H_kv, D, D], which the kernel
/// carries across the sequence. Optional tails arrive as empty views with rank
/// 0 -- absent, not malformed -- and conv_weights come shaped
/// [(H + 2*H_kv) * D, 1, K].

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
    int64_t B  = Q.dims[0];
    int64_t H_kv = attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads : attrs_.num_heads;
    int64_t D  = attrs_.head_dim;

    TensorDesc out_desc  = Q;
    TensorDesc state_desc;
    state_desc.rank   = 4;
    state_desc.dims   = {B, H_kv, D, D};
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
    if (!detail::dtype_in<DataType::f32, DataType::f16>(dt)) {
        return Status::unsupported;
    }
    for (size_t i = 1; i < 4; ++i) {
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    if (outputs[0].dtype != dt || outputs[1].dtype != dt) {
        return Status::invalid_argument;
    }

    if (inputs[0].rank != 4) { return Status::invalid_argument; }
    const int64_t b = inputs[0].dims[0];
    const int64_t h = inputs[0].dims[1];
    const int64_t s = inputs[0].dims[2];
    const int64_t d = inputs[0].dims[3];
    const int64_t h_kv = attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads : h;
    if (h_kv <= 0 || h % h_kv != 0) { return Status::invalid_argument; }

    for (size_t i = 1; i <= 3; ++i) {
        const auto& t = inputs[i];
        if (t.rank != 4 || t.dims[0] != b || t.dims[1] != h_kv ||
            t.dims[2] != s || t.dims[3] != d) {
            return Status::invalid_argument;
        }
    }
    if (outputs[0].rank != 4 || outputs[0].dims[0] != b || outputs[0].dims[1] != h ||
        outputs[0].dims[2] != s || outputs[0].dims[3] != d) {
        return Status::invalid_argument;
    }
    if (outputs[1].rank != 4 || outputs[1].dims[0] != b ||
        outputs[1].dims[1] != h_kv || outputs[1].dims[2] != d ||
        outputs[1].dims[3] != d) {
        return Status::invalid_argument;
    }

    if (inputs.size() > 4 && inputs[4].rank != 0) {   // [(H + 2*H_kv) * D, 1, K]
        const auto& w = inputs[4];
        if (w.dtype != dt || w.rank != 3 ||
            w.dims[0] != (h + 2 * h_kv) * d || w.dims[1] != 1 ||
            w.dims[2] != attrs_.conv_kernel_size) {
            return Status::invalid_argument;
        }
    }
    if (inputs.size() > 5 && inputs[5].rank != 0) {   // beta: [B, H_kv, S]
        const auto& beta = inputs[5];
        if (beta.dtype != dt || beta.rank != 3 || beta.dims[0] != b ||
            beta.dims[1] != h_kv || beta.dims[2] != s) {
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
