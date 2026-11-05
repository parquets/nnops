#pragma once
/// @file scatter.hpp
/// @brief ScatterElements (ONNX Scatter) and ScatterND — write updates into a
///        copy of data at indices.
///
/// Both start from a copy of `data`, so an index that is visited twice keeps the
/// last write under Reduction::None (concurrent duplicates are unordered) and
/// accumulates under Add/Mul. Indices are s64/s32 and may be negative,
/// wrapping from the end of the axis (ScatterElements) or of each axis the
/// tuple addresses (ScatterND). Data may be f32/f16/s8/u8/s32/s64. Channel-first.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class ScatterReduction : uint8_t {
    None,
    Add,
    Mul,
};

/// ScatterElements: indices and updates share one shape; each writes
/// out[.., idx[..], ..] along `axis`.
struct ScatterElementsAttributes {
    int64_t axis = 0;
    ScatterReduction reduction = ScatterReduction::None;
};

class ScatterElements : public OpBase {
public:
    static std::unique_ptr<ScatterElements> create(
        const ScatterElementsAttributes& attrs, Backend backend = Backend::CPU);
    static std::unique_ptr<ScatterElements> create(Backend backend = Backend::CPU) {
        return create(ScatterElementsAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::ScatterElements; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    struct Impl;
    ~ScatterElements();

private:
    ScatterElements(const ScatterElementsAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ScatterElementsAttributes attrs_;
    Backend backend_;
};

/// ScatterND: indices' last axis is a tuple of K, updates address the slice
/// data[tuple..., :]; updates = indices[:-1] + data[K:].
struct ScatterNDAttributes {
    ScatterReduction reduction = ScatterReduction::None;
};

class ScatterND : public OpBase {
public:
    static std::unique_ptr<ScatterND> create(const ScatterNDAttributes& attrs,
                                             Backend backend = Backend::CPU);
    static std::unique_ptr<ScatterND> create(Backend backend = Backend::CPU) {
        return create(ScatterNDAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::ScatterND; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    struct Impl;
    ~ScatterND();

private:
    ScatterND(const ScatterNDAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ScatterNDAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
