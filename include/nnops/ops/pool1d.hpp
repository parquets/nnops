#pragma once
/// @file pool1d.hpp
/// @brief Pool1D — max/average pooling over one spatial axis, channel-first or
///        channel-last.
///
/// The 2D pooling kernel with a 1-wide second spatial axis: [N, C, L] is lifted
/// to [N, C, 1, L] and Pooling is called with a window of {1, K}. The lift keeps
/// the caller's pitch and layout, so the window walk, the auto-pad resolution
/// and the channel-last strip split are the 2D ones.
///
/// The output's layout follows the input's. Channel-first input supports
/// f32/f16; quantized (s8/u8) pooling is channel-last only, where the lane is
/// the channel — the same rule and the same reason as Pooling.
///
/// global pools the whole signal to [N, C, 1]. Adaptive (`output_size > 0`)
/// walks PyTorch's floor/ceil windows over the L axis; a channel-first adaptive
/// walk splits over (batch, channel) while a channel-last one splits over
/// output rows, which for a 1-wide width is one task per batch item — so
/// adaptive pooling stays sequential on a channel-last tensor at N = 1.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class Pool1DType : uint8_t {
    Max,
    Average,
};

struct Pool1DAttributes {
    Pool1DType type = Pool1DType::Max;

    int64_t kernel_shape = 2;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;

    /// If true, exclude padded positions from the average (Average only).
    bool exclude_pad = false;

    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;

    /// Global pooling: the output length is 1; the window fields are not
    /// consulted. `type` still selects max vs mean.
    bool global = false;

    /// Adaptive pooling: the output length, 0 means the fixed window fields
    /// decide. The window of output index o spans input
    /// [floor(o·I/O), ceil((o+1)·I/O)).
    int64_t output_size = 0;

    bool adaptive() const noexcept { return output_size > 0; }
};

/// [N, C, IL] → [N, C, OL]; global → [N, C, 1].
class Pool1D : public OpBase {
public:
    static std::unique_ptr<Pool1D> create(const Pool1DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Pool1D> create(Backend backend = Backend::CPU) {
        return create(Pool1DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Pool1D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const Pool1DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in pool1d.cpp
    ~Pool1D();

private:
    Pool1D(const Pool1DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Pool1DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
