// Metal backend demo: unary and eltwise through Backend::Metal, checked against
// the CPU backend on the same data.
//
// Buffers are MTLBuffers in shared storage; their contents() doubles as the
// TensorView data, and the same handles go to ComputeContext::metal_buffers in
// the order the kernels declare them — [input, output] for unary, [A, B,
// output] for eltwise.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "nnops/detail/half.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/unary.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

id<MTLBuffer> make_buffer(id<MTLDevice> dev, size_t bytes) {
  return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

bool close_enough(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

int run_eltwise(id<MTLDevice> dev, int64_t n) {
  std::vector<float> a(n), b(n), want(n);
  for (int64_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i % 17) * 0.25f;
    b[i] = 1.0f + static_cast<float>(i % 5);
  }

  id<MTLBuffer> ba = make_buffer(dev, n * sizeof(float));
  id<MTLBuffer> bb = make_buffer(dev, n * sizeof(float));
  id<MTLBuffer> bo = make_buffer(dev, n * sizeof(float));
  std::memcpy([ba contents], a.data(), n * sizeof(float));
  std::memcpy([bb contents], b.data(), n * sizeof(float));

  const int64_t shape[] = {n};
  TensorView A(shape, DataType::f32, [ba contents]);
  TensorView B(shape, DataType::f32, [bb contents]);
  TensorView C(shape, DataType::f32, [bo contents]);

  void* buffers[3] = {(__bridge void*)ba, (__bridge void*)bb, (__bridge void*)bo};
  ComputeContext ctx;
  ctx.metal_buffers = buffers;
  ctx.metal_buffers_count = 3;

  EltwiseAttributes attrs;
  attrs.type = EltwiseType::Add;
  auto gpu = Eltwise::create(attrs, Backend::Metal);
  TensorView ins[] = {A, B};
  TensorView outs[] = {C};
  const Status st = gpu->compute(outs, ins, ctx);
  if (st != Status::ok) {
    std::printf("eltwise add: Metal compute returned %d\n", static_cast<int>(st));
    return 1;
  }

  // CPU reference on the same inputs.
  std::vector<float> cpu_out(n);
  TensorView Cp(shape, DataType::f32, cpu_out.data());
  auto cpu = Eltwise::create(attrs, Backend::CPU);
  TensorView cpu_outs[] = {Cp};
  cpu->compute(cpu_outs, ins, ComputeContext{});

  const float* got = static_cast<const float*>([bo contents]);
  int bad = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (!close_enough(got[i], cpu_out[static_cast<size_t>(i)], 0.0f)) { ++bad; }
  }
  std::printf("eltwise f32 Add  n=%lld  max|gpu-cpu|=%s  %s\n", (long long)n,
              bad ? "nonzero" : "0", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

int run_unary(id<MTLDevice> dev, int64_t n) {
  std::vector<float> in(n);
  for (int64_t i = 0; i < n; ++i) {
    in[i] = -2.0f + 4.0f * static_cast<float>(i) / static_cast<float>(n);
  }

  id<MTLBuffer> bi = make_buffer(dev, n * sizeof(float));
  id<MTLBuffer> bo = make_buffer(dev, n * sizeof(float));
  std::memcpy([bi contents], in.data(), n * sizeof(float));

  const int64_t shape[] = {n};
  TensorView X(shape, DataType::f32, [bi contents]);
  TensorView Y(shape, DataType::f32, [bo contents]);

  void* buffers[2] = {(__bridge void*)bi, (__bridge void*)bo};
  ComputeContext ctx;
  ctx.metal_buffers = buffers;
  ctx.metal_buffers_count = 2;

  UnaryAttributes attrs;
  attrs.type = UnaryType::Exp;
  auto gpu = Unary::create(attrs, Backend::Metal);
  TensorView ins[] = {X};
  TensorView outs[] = {Y};
  const Status st = gpu->compute(outs, ins, ctx);
  if (st != Status::ok) {
    std::printf("unary exp: Metal compute returned %d\n", static_cast<int>(st));
    return 1;
  }

  std::vector<float> cpu_out(n);
  TensorView Yp(shape, DataType::f32, cpu_out.data());
  auto cpu = Unary::create(attrs, Backend::CPU);
  TensorView cpu_outs[] = {Yp};
  cpu->compute(cpu_outs, ins, ComputeContext{});

  const float* got = static_cast<const float*>([bo contents]);
  float worst = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    worst = std::fmax(worst, std::fabs(got[i] - cpu_out[static_cast<size_t>(i)]));
  }
  std::printf("unary   f32 Exp  n=%lld  max|gpu-cpu|=%.3g  %s\n", (long long)n,
              worst, worst < 1e-5f ? "PASS" : "FAIL");
  return worst < 1e-5f ? 0 : 1;
}

int run_unary_f16(id<MTLDevice> dev, int64_t n) {
  std::vector<half> in(n);
  for (int64_t i = 0; i < n; ++i) {
    in[i] = static_cast<half>(-1.0f + 2.0f * static_cast<float>(i) /
                                           static_cast<float>(n));
  }

  id<MTLBuffer> bi = make_buffer(dev, n * sizeof(half));
  id<MTLBuffer> bo = make_buffer(dev, n * sizeof(half));
  std::memcpy([bi contents], in.data(), n * sizeof(half));

  const int64_t shape[] = {n};
  TensorView X(shape, DataType::f16, [bi contents]);
  TensorView Y(shape, DataType::f16, [bo contents]);

  void* buffers[2] = {(__bridge void*)bi, (__bridge void*)bo};
  ComputeContext ctx;
  ctx.metal_buffers = buffers;
  ctx.metal_buffers_count = 2;

  UnaryAttributes attrs;
  attrs.type = UnaryType::Tanh;
  auto gpu = Unary::create(attrs, Backend::Metal);
  TensorView ins[] = {X};
  TensorView outs[] = {Y};
  if (gpu->compute(outs, ins, ctx) != Status::ok) {
    std::printf("unary f16 tanh: Metal compute failed\n");
    return 1;
  }

  std::vector<half> cpu_out(n);
  TensorView Yp(shape, DataType::f16, cpu_out.data());
  auto cpu = Unary::create(attrs, Backend::CPU);
  TensorView cpu_outs[] = {Yp};
  cpu->compute(cpu_outs, ins, ComputeContext{});

  const half* got = static_cast<const half*>([bo contents]);
  float worst = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    worst = std::fmax(worst,
                      std::fabs(static_cast<float>(got[i]) -
                                static_cast<float>(cpu_out[static_cast<size_t>(i)])));
  }
  std::printf("unary   f16 Tanh n=%lld  max|gpu-cpu|=%.3g  %s\n", (long long)n,
              worst, worst < 5e-3f ? "PASS" : "FAIL");
  return worst < 5e-3f ? 0 : 1;
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
      std::printf("no Metal device\n");
      return 1;
    }
    std::printf("Metal device: %s\n", [[dev name] UTF8String]);
    int rc = 0;
    rc |= run_eltwise(dev, 1 << 16);
    rc |= run_unary(dev, 1 << 16);
    rc |= run_unary_f16(dev, 1 << 16);
    std::printf("%s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
  }
}
