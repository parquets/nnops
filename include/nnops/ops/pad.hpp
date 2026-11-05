#pragma once
/// @file pad.hpp
/// @brief Pad — pad every axis with a constant or a border/reflection/wrap.
///
/// `pads` is [begin_0 … begin_{R-1}, end_0 … end_{R-1}] in elements, one pair
/// per axis; a negative value crops. Modes follow ONNX: Constant fills with
/// `constant_value`, Edge repeats the border, Reflect mirrors without repeating
/// the border (each pad must stay below its axis length), Wrap tiles the axis.
///
/// Pads ride as an attribute, not an input: getOutputTensorDesc() reads
/// descriptors only, so a pads *tensor* could not size the output. A graph
/// front-end resolves a constant pads input into the attribute. Channel-first order.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

struct PadAttributes {
    enum class Mode : uint8_t {
        Constant,
        Edge,
        Reflect,
        Wrap,
    };

    /// [b0..b(R-1), e0..e(R-1)]; negative crops.
    std::vector<int64_t> pads;

    Mode mode = Mode::Constant;

    /// Fill for Constant mode; int dtypes round it.
    float constant_value = 0.0f;
};

class Pad : public OpBase {
public:
    static std::unique_ptr<Pad> create(const PadAttributes& attrs,
                                       Backend backend = Backend::CPU);

    static std::unique_ptr<Pad> create(Backend backend = Backend::CPU) {
        return create(PadAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Pad; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    const PadAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Pad();

private:
    Pad(const PadAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PadAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
