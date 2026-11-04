#pragma once
/// @file tile.hpp
/// @brief Tile and Expand — the two ways to repeat a tensor.
///
/// Tile repeats each axis `repeats[d]` times, so out.dims = in.dims * repeats.
/// Expand broadcasts the input to a target shape under NumPy rules: the target
/// is right-aligned, a size-1 (or missing) input axis repeats, and every other
/// axis must match the target exactly. Both are element copies. Planar.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

struct TileAttributes {
    /// One repeat count per input axis, in order.
    std::vector<int64_t> repeats;
};

class Tile : public OpBase {
public:
    static std::unique_ptr<Tile> create(const TileAttributes& attrs,
                                        Backend backend = Backend::CPU);
    static std::unique_ptr<Tile> create(Backend backend = Backend::CPU) {
        return create(TileAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Tile; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~Tile();

private:
    Tile(const TileAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    TileAttributes attrs_;
    Backend backend_;
};

struct ExpandAttributes {
    /// Target shape; a negative entry means "same as the input axis there".
    std::vector<int64_t> shape;
};

class Expand : public OpBase {
public:
    static std::unique_ptr<Expand> create(const ExpandAttributes& attrs,
                                          Backend backend = Backend::CPU);
    static std::unique_ptr<Expand> create(Backend backend = Backend::CPU) {
        return create(ExpandAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Expand; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~Expand();

private:
    Expand(const ExpandAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ExpandAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
