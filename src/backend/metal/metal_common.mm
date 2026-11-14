/// @file metal_common.mm
/// @brief The Metal runtime behind the unary and eltwise kernels.
///
/// The MSL below is compiled at runtime (`newLibraryWithSource:`) the first time
/// a Metal op runs. An application that wants AOT can `xcrun metal` the same
/// source into a .metallib and swap `ensure_ready()` for newLibraryWithURL.
///
/// Pipelines are cached per function name; the op code and the add_to flag ride
/// a constant `Params` block at buffer(8), so a dtype needs one pipeline no
/// matter how many operations use it. Element types are f32 and f16; the f16
/// kernels do their math in fp32 and narrow on store, like the Vulkan pair.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_common.hpp"

#include "nnops/core/compute_context.hpp"

#include <cstdio>

namespace nnops::backend::metal {
namespace {

NSString* const kSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    int op;
    int add_to;
    uint n;
};

static inline float unary_op(float x, int op) {
    switch (op) {
        case 0:  return exp(x);
        case 1:  return log(x);
        case 2:  return sin(x);
        case 3:  return cos(x);
        case 4:  return tan(x);
        case 5:  return tanh(x);
        case 6:  return fabs(x);
        case 7:  return -x;
        case 8:  return sqrt(x);
        case 10: return rint(x);
        case 11: return ceil(x);
        case 12: return floor(x);
        case 13: return 1.0f / x;
        case 14: return sign(x);
        case 15: return exp2(x);
        case 16: return log2(x);
        case 17: return log10(x);
        case 18: return asin(x);
        case 19: return acos(x);
        case 20: return atan(x);
        case 21: return sinh(x);
        case 22: return cosh(x);
        case 23: return asinh(x);
        case 24: return acosh(x);
        case 25: return atanh(x);
    }
    return x;
}

kernel void unary_f32(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant Params& p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) { return; }
    const float y = unary_op(in[i], p.op);
    out[i] = p.add_to ? out[i] + y : y;
}

kernel void unary_f16(device const half* in [[buffer(0)]],
                      device half* out [[buffer(1)]],
                      constant Params& p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) { return; }
    const float y = unary_op(float(in[i]), p.op);
    out[i] = p.add_to ? half(float(out[i]) + y) : half(y);
}

static inline float eltwise_op(float a, float b, int op) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return fmin(a, b);
        case 5: return fmax(a, b);
        case 6: return pow(a, b);
    }
    return a;
}

kernel void eltwise_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant Params& p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) { return; }
    const float y = eltwise_op(a[i], b[i], p.op);
    out[i] = p.add_to ? out[i] + y : y;
}

kernel void eltwise_f16(device const half* a [[buffer(0)]],
                        device const half* b [[buffer(1)]],
                        device half* out [[buffer(2)]],
                        constant Params& p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) { return; }
    const float y = eltwise_op(float(a[i]), float(b[i]), p.op);
    out[i] = p.add_to ? half(float(out[i]) + y) : half(y);
}
)MSL";

const char* function_name(KernelFn fn) {
    switch (fn) {
    case KernelFn::UnaryF32:   return "unary_f32";
    case KernelFn::UnaryF16:   return "unary_f16";
    case KernelFn::EltwiseF32: return "eltwise_f32";
    case KernelFn::EltwiseF16: return "eltwise_f16";
    }
    return nullptr;
}

struct State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    bool ready = false;
};

State& state() {
    static State s;
    return s;
}

bool ensure_ready() {
    State& s = state();
    if (s.ready) { return true; }
    @autoreleasepool {
        s.device = MTLCreateSystemDefaultDevice();
        if (s.device == nil) { return false; }
        NSError* err = nil;
        s.library = [s.device newLibraryWithSource:kSource options:nil error:&err];
        if (s.library == nil) {
            std::fprintf(stderr, "nnops metal: MSL compile failed: %s\n",
                         [[err description] UTF8String]);
            return false;
        }
        s.queue = [s.device newCommandQueue];
        s.pipelines = [NSMutableDictionary dictionary];
        s.ready = (s.queue != nil);
    }
    return s.ready;
}

id<MTLComputePipelineState> pipeline(KernelFn fn) {
    State& s = state();
    NSString* name = @(function_name(fn));
    id<MTLComputePipelineState> pso = s.pipelines[name];
    if (pso != nil) { return pso; }
    NSError* err = nil;
    id<MTLFunction> f = [s.library newFunctionWithName:name];
    if (f == nil) { return nil; }
    pso = [s.device newComputePipelineStateWithFunction:f error:&err];
    if (pso == nil) {
        std::fprintf(stderr, "nnops metal: pipeline %s failed: %s\n",
                     [name UTF8String], [[err description] UTF8String]);
        return nil;
    }
    s.pipelines[name] = pso;
    return pso;
}

}  // namespace

Status dispatch(KernelFn fn, int32_t op, bool add_to, const ComputeContext& ctx,
                int64_t total, int n_tensor_buffers) {
    if (total <= 0) { return Status::ok; }
    if (ctx.metal_buffers == nullptr ||
        ctx.metal_buffers_count < n_tensor_buffers) {
        return Status::unsupported;
    }
    if (!ensure_ready()) { return Status::unsupported; }
    id<MTLComputePipelineState> pso = pipeline(fn);
    if (pso == nil) { return Status::unsupported; }

    @autoreleasepool {
        const void* const* raw = static_cast<const void* const*>(ctx.metal_buffers);
        id<MTLCommandBuffer> cb = [state().queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (int i = 0; i < n_tensor_buffers; ++i) {
            [enc setBuffer:(__bridge id<MTLBuffer>)raw[i] offset:0 atIndex:i];
        }
        struct Params { int32_t op; int32_t add_to; uint32_t n; };
        Params p{op, add_to ? 1 : 0, static_cast<uint32_t>(total)};
        [enc setBytes:&p length:sizeof(p) atIndex:8];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > 256) { tg = 256; }
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted
                   ? Status::ok
                   : Status::unsupported;
    }
}

}  // namespace nnops::backend::metal
