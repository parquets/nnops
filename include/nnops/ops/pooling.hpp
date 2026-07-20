#pragma once
/// @file pooling.hpp
/// @brief Pooling operator — 2D/3D spatial pooling with NCHW/NCDHW layout.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported pooling types.
enum class PoolingType : uint8_t {
    Max,                 ///< Maximum value in window
    Average,             ///< Average of values in window (including padding)
    AverageExcludePad,   ///< Average excluding padded positions
    Lp,                  ///< Lp-norm pooling
};

/// Attributes for the Pooling operator (supports both 2D and 3D).
///
/// For 2D pooling (NCHW input, rank=4): only the last 2 elements of each array
/// are used (H, W dimensions). The first element (D) is ignored.
///
/// For 3D pooling (NCDHW input, rank=5): all 3 elements are used (D, H, W).
struct PoolingAttributes {
    PoolingType type = PoolingType::Max;

    /// Kernel shape: [KD, KH, KW]. For 2D, only KH/KW are used.
    std::array<int64_t, 3> kernel_shape = {2, 2, 2};

    /// Stride: [SD, SH, SW]. For 2D, only SH/SW are used.
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// Padding: [PD, PH, PW]. For 2D, only PH/PW are used.
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// Dilation: [DD, DH, DW]. For 2D, only DH/DW are used.
    std::array<int64_t, 3> dilation = {1, 1, 1};

    int64_t p_norm = 2;  ///< p value for Lp pooling

    /// Auto-padding mode.
    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// Returns the spatial rank (2 or 3) inferred from the input tensor rank.
    /// For a 4D input (NCHW), returns 2. For a 5D input (NCDHW), returns 3.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// Pooling operator (class-based API).
///
/// 2D: Input [N, C, IH, IW] → Output [N, C, OH, OW]
/// 3D: Input [N, C, ID, IH, IW] → Output [N, C, OD, OH, OW]
class Pooling : public OpBase {
public:
    /// Create a Pooling operator for the specified backend.
    static std::unique_ptr<Pooling> create(const PoolingAttributes& attrs,
                                           Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Pooling> create(Backend backend = Backend::CPU) {
        return create(PoolingAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Pooling; }
    Backend getBackend() const override { return backend_; }

    const PoolingAttributes& attributes() const noexcept { return attrs_; }

private:
    Pooling(const PoolingAttributes& attrs, Backend backend);
    PoolingAttributes attrs_;
    Backend backend_;
};

/// Functional pooling.
void pooling(const TensorView& input,
             const TensorView& output,
             const PoolingAttributes& attrs,
             const ComputeContext& ctx = {});

}  // namespace nnops
