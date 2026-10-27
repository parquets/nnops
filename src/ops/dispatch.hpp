#pragma once
/// @file dispatch.hpp
/// @brief The per-op backend kernel table.
///
/// Every operator repeats the same things: forward declarations of its backend
/// entry points, an Impl holding the resolved function pointer, the factory,
/// and a switch on Backend. These macros emit that. Ops whose kernels differ in
/// signature or whose constructor does more (Conv2D picks by `algorithm`,
/// DequantizeLinear rewrites `output_dtype`, topk/attention write several
/// outputs) build from the pieces (NNOPS_KERNEL_IMPL, NNOPS_OP_CREATE) instead.
///
/// An op declares only the backends it has kernels for and passes `nullptr`
/// for the rest; NNOPS_IF_* drops the name in builds where that backend is off.

#include <memory>
#include <span>

#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/assert.hpp"

#if defined(NNOPS_HAS_CUDA)
#define NNOPS_IF_CUDA(x) x
#else
#define NNOPS_IF_CUDA(x) nullptr
#endif
#if defined(NNOPS_HAS_VULKAN)
#define NNOPS_IF_VULKAN(x) x
#else
#define NNOPS_IF_VULKAN(x) nullptr
#endif

/// Default kernel signature: one output view, N input views.
#define NNOPS_KERNEL_ARGS(Attrs)                                       \
    (const Attrs&, TensorView&, std::span<const TensorView>,           \
     const ComputeContext&, void*)

#define NNOPS_DECLARE_CPU(Attrs, fn)                                   \
    namespace backend::cpu { void fn NNOPS_KERNEL_ARGS(Attrs); }
#define NNOPS_DECLARE_CUDA(Attrs, fn)                                  \
    namespace backend::cuda { void fn NNOPS_KERNEL_ARGS(Attrs); }
#define NNOPS_DECLARE_VULKAN(Attrs, fn)                                \
    namespace backend::vulkan { void fn NNOPS_KERNEL_ARGS(Attrs); }

/// The op's Impl (one function pointer) and its out-of-line destructor.
#define NNOPS_KERNEL_IMPL(Op, Attrs)                                   \
    struct Op::Impl {                                                  \
        using KernelFn = void (*) NNOPS_KERNEL_ARGS(Attrs);            \
        KernelFn kernel_fn = nullptr;                                  \
    };                                                                 \
    Op::~Op() = default;

/// The factory is the same everywhere.
#define NNOPS_OP_CREATE(Op, Attrs)                                     \
    std::unique_ptr<Op> Op::create(const Attrs& attrs, Backend backend) \
    {                                                                  \
        return std::unique_ptr<Op>(new Op(attrs, backend));            \
    }

/// Impl, destructor, factory and constructor — the shape every op shares.
#define NNOPS_OP_BOILERPLATE(Op, Attrs, cpu, cuda, vulkan)             \
    NNOPS_KERNEL_IMPL(Op, Attrs)                                       \
    NNOPS_OP_CREATE(Op, Attrs)                                         \
    Op::Op(const Attrs& attrs, Backend backend)                        \
        : impl_(std::make_unique<Impl>()), attrs_(attrs),               \
          backend_(backend)                                            \
    {                                                                  \
        impl_->kernel_fn = NNOPS_RESOLVE(Op, Op::Impl::KernelFn,       \
                                         backend, cpu, cuda, vulkan);  \
    }

#define NNOPS_PICK_KERNEL(Fn, be, cpu, cuda, vulkan)                   \
    ((be) == Backend::CPU    ? static_cast<Fn>(cpu)    :               \
     (be) == Backend::CUDA   ? static_cast<Fn>(cuda)   :               \
     (be) == Backend::Vulkan ? static_cast<Fn>(vulkan) : nullptr)

/// Resolve one backend's kernel, failing loudly when the backend has none.
#define NNOPS_RESOLVE(Op, Fn, be, cpu, cuda, vulkan)                   \
    [] (Backend b) -> Fn {                                             \
        const Fn fn = NNOPS_PICK_KERNEL(Fn, b, cpu, cuda, vulkan);     \
        NNOPS_CHECK_MSG(fn != nullptr, #Op ": backend has no kernel"); \
        return fn;                                                     \
    }(be)
