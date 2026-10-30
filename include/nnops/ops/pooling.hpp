#pragma once
/// @file pooling.hpp
/// @brief Pooling operator — 2D/3D spatial pooling, planar (NCHW/NCDHW) or
///        channel-last (NHWC/NDHWC).
///
/// The output's layout follows the input's. Planar input supports f32/f16;
/// quantized (s8/u8) pooling is channel-last only — its kernel lanes the
/// channel, where a planar plane lanes output width. Per-axis arrays
/// are [D, H, W]; a rank-4 (2D) input reads only the H/W entries, a rank-5
/// (3D) input all three.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class PoolingType : uint8_t {
    Max,
    Average,
};

struct PoolingAttributes {
    PoolingType type = PoolingType::Max;

    /// [KD, KH, KW]
    std::array<int64_t, 3> kernel_shape = {2, 2, 2};

    /// [SD, SH, SW]
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// [PD, PH, PW]
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// [DD, DH, DW]
    std::array<int64_t, 3> dilation = {1, 1, 1};

    /// If true, exclude padded positions from the average (Average only).
    bool exclude_pad = false;

    /// Auto-padding mode.
    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;

    /// Global pooling: every output spatial axis is 1; the window fields
    /// (kernel_shape/stride/padding/dilation/auto_pad) are not consulted.
    /// `type` still selects max vs mean.
    bool global = false;

    /// 2 for a rank-4 input, 3 for rank-5.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// 2D: [N, C, IH, IW] → [N, C, OH, OW]; 3D adds a D axis; global → [N, C, 1, ...].
class Pooling : public OpBase {
public:
    static std::unique_ptr<Pooling> create(const PoolingAttributes& attrs,
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Pooling> create(Backend backend = Backend::CPU) {
        return create(PoolingAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Pooling; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const PoolingAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in pooling.cpp
    ~Pooling();

private:
    Pooling(const PoolingAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PoolingAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
