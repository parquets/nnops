#pragma once
/// @file gather.hpp
/// @brief Gather family — Gather (one axis, any index shape), GatherElements
///        (one element per index, indices and data the same rank) and GatherND
///        (an index tuple into the leading axes).
///
/// Indices are s64 or s32 and may be negative, wrapping from the end of the
/// gathered axis. Data may be f32/f16/s8/u8/s32/s64 — a gather moves elements
/// without touching them. Planar order.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Gather: output = data[..., indices[...], ...] along `axis`; the index shape
/// replaces the gathered axis, so out.rank = data.rank - 1 + indices.rank.
struct GatherAttributes {
    int64_t axis = 0;  ///< negative values wrap
};

class Gather : public OpBase {
public:
    static std::unique_ptr<Gather> create(const GatherAttributes& attrs,
                                          Backend backend = Backend::CPU);
    static std::unique_ptr<Gather> create(Backend backend = Backend::CPU) {
        return create(GatherAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Gather; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~Gather();

private:
    Gather(const GatherAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    GatherAttributes attrs_;
    Backend backend_;
};

/// GatherElements: same rank on both sides; the output takes the indices' shape
/// and out[i] = data[i with idx[i] swapped in at `axis`].
struct GatherElementsAttributes {
    int64_t axis = 0;
};

class GatherElements : public OpBase {
public:
    static std::unique_ptr<GatherElements> create(
        const GatherElementsAttributes& attrs, Backend backend = Backend::CPU);
    static std::unique_ptr<GatherElements> create(Backend backend = Backend::CPU) {
        return create(GatherElementsAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::GatherElements; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~GatherElements();

private:
    GatherElements(const GatherElementsAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    GatherElementsAttributes attrs_;
    Backend backend_;
};

/// GatherND: indices' last axis is a tuple of length K read into data's axes
/// [batch_dims, batch_dims+K); out = indices[:-1] + data[batch_dims+K:].
struct GatherNDAttributes {
    int64_t batch_dims = 0;
};

class GatherND : public OpBase {
public:
    static std::unique_ptr<GatherND> create(const GatherNDAttributes& attrs,
                                            Backend backend = Backend::CPU);
    static std::unique_ptr<GatherND> create(Backend backend = Backend::CPU) {
        return create(GatherNDAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::GatherND; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~GatherND();

private:
    GatherND(const GatherNDAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    GatherNDAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
