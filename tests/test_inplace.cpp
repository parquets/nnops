/// @file test_inplace.cpp
/// @brief The in-place contract: OpBase::supports_inplace() and the aliased
///        calls it promises.
///
/// An aliased call is run four ways — out-of-place and in-place, serial and on
/// four threads — and all four outputs must be bit-identical. Bit-exactness is
/// the point: the aliased call is the same computation, so nothing but the
/// storage should differ.
///
/// The query is also checked against geometry it must refuse (another extent,
/// dtype or layout), against an operator instance whose backend kernel it does
/// not cover, and against an operator that never opted in.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/cast.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/rope.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/unary.hpp"
#include "nnops/ops/where.hpp"
#include "common/test_harness.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

/// A four-worker pool over the operator's own hooks.
ComputeContext par_ctx(int nthreads = 4)
{
    ComputeContext ctx;
    const int nt = nthreads;
    ctx.cpu.num_threads = [nt] { return nt; };
    ctx.cpu.parallel_for = [nt](int64_t begin, int64_t end, const ParallelForBody& body) {
        const int64_t n = end - begin;
        if (n <= 1 || nt <= 1) {
            for (int64_t i = begin; i < end; ++i) {
                body(i);
            }
            return;
        }
        const int64_t per = (n + nt - 1) / nt;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nt - 1));
        for (int t = 1; t < nt; ++t) {
            const int64_t lo = begin + t * per;
            const int64_t hi = std::min(lo + per, end);
            if (lo >= hi) {
                break;
            }
            workers.emplace_back([&body, lo, hi] {
                for (int64_t i = lo; i < hi; ++i) {
                    body(i);
                }
            });
        }
        const int64_t hi0 = std::min(begin + per, end);
        for (int64_t i = begin; i < hi0; ++i) {
            body(i);
        }
        for (auto& w : workers) {
            w.join();
        }
    };
    return ctx;
}

/// Values in [-1, 1) at full f32 spread.
std::vector<float> ramp(int64_t n, uint32_t seed)
{
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / 8388608.0f - 1.0f;
    }
    return v;
}

using RunFn = std::function<Status(const ComputeContext&, TensorView&)>;

/// The four arms: out-of-place serial, in-place serial, out-of-place on four
/// threads, in-place on four threads. `reset()` restores every buffer the call
/// may write (the aliased input storage, and an add_to case's separate output).
void expect_alias_equal(const char* what, const RunFn& run, const std::function<void()>& reset,
                        TensorView& out_sep, TensorView& out_alias, size_t out_bytes)
{
    (void)what;
    reset();
    NNOPS_EXPECT_TRUE(is_ok(run(ComputeContext{}, out_sep)));
    std::vector<std::byte> ref(out_bytes);
    std::memcpy(ref.data(), out_sep.ptr<void>(), out_bytes);

    reset();
    NNOPS_EXPECT_TRUE(is_ok(run(ComputeContext{}, out_alias)));
    NNOPS_EXPECT_TRUE(std::memcmp(ref.data(), out_alias.ptr<void>(), out_bytes) == 0);

    reset();
    NNOPS_EXPECT_TRUE(is_ok(run(par_ctx(4), out_sep)));
    NNOPS_EXPECT_TRUE(std::memcmp(ref.data(), out_sep.ptr<void>(), out_bytes) == 0);

    reset();
    NNOPS_EXPECT_TRUE(is_ok(run(par_ctx(4), out_alias)));
    NNOPS_EXPECT_TRUE(std::memcmp(ref.data(), out_alias.ptr<void>(), out_bytes) == 0);
}

std::span<const int64_t> shape_of(const int64_t* dims, int64_t rank)
{
    return std::span<const int64_t>(dims, static_cast<size_t>(rank));
}

} // namespace

// ============================================================
// Element-wise: the aliased store lands on an element already read
// ============================================================

NNOPS_TEST(inplace_activation)
{
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 7u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc in_d[] = {x.desc()};
    const TensorDesc out_d[] = {out_sep.desc()};
    NNOPS_EXPECT_TRUE(op->supports_inplace(in_d, out_d));

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("activation", run, [&] { xbuf = x0; }, out_sep, out_alias,
                       n * sizeof(float));
}

NNOPS_TEST(inplace_activation_add_to)
{
    // add_to reads the destination before storing, so the aliased call is the
    // residual-in-place form: x = relu(x) + x.
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 11u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep = x0;

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    ActivationAttributes attrs;
    attrs.type = ActivationType::Silu;
    attrs.add_to = true;
    auto op = Activation::create(attrs, Backend::CPU);

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("activation/add_to", run,
                       [&] {
                           xbuf = x0;
                           sep = x0;
                       },
                       out_sep, out_alias, n * sizeof(float));
}

NNOPS_TEST(inplace_unary)
{
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 13u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    auto op = Unary::create(attrs, Backend::CPU);

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("unary", run, [&] { xbuf = x0; }, out_sep, out_alias, n * sizeof(float));
}

NNOPS_TEST(inplace_clamp)
{
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 17u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    ClampAttributes attrs;
    attrs.min_val = -0.5f;
    attrs.max_val = 0.5f;
    auto op = Clamp::create(attrs, Backend::CPU);

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("clamp", run, [&] { xbuf = x0; }, out_sep, out_alias, n * sizeof(float));
}

NNOPS_TEST(inplace_eltwise)
{
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> a0 = ramp(n, 19u);
    const std::vector<float> b0 = ramp(n, 23u);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    auto op = Eltwise::create(attrs, Backend::CPU);

    // out == a
    {
        std::vector<float> abuf = a0;
        std::vector<float> sep(n, 0.0f);
        std::vector<float> bbuf = b0;
        TensorView a(shape_of(dims, 2), DataType::f32, abuf.data());
        TensorView b(shape_of(dims, 2), DataType::f32, bbuf.data());
        TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
        TensorView out_alias(shape_of(dims, 2), DataType::f32, abuf.data());

        const auto run = [&](const ComputeContext& ctx, TensorView& o) {
            const TensorView ins[] = {a, b};
            return op->compute(o, ins, ctx);
        };
        expect_alias_equal("eltwise/a", run, [&] { abuf = a0; }, out_sep, out_alias,
                           n * sizeof(float));
    }

    // out == b
    {
        std::vector<float> abuf = a0;
        std::vector<float> bbuf = b0;
        std::vector<float> sep(n, 0.0f);
        TensorView a(shape_of(dims, 2), DataType::f32, abuf.data());
        TensorView b(shape_of(dims, 2), DataType::f32, bbuf.data());
        TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
        TensorView out_alias(shape_of(dims, 2), DataType::f32, bbuf.data());

        const auto run = [&](const ComputeContext& ctx, TensorView& o) {
            const TensorView ins[] = {a, b};
            return op->compute(o, ins, ctx);
        };
        expect_alias_equal("eltwise/b", run, [&] { bbuf = b0; }, out_sep, out_alias,
                           n * sizeof(float));
    }
}

NNOPS_TEST(inplace_where)
{
    // out == x: the selected branch is read at the element being written.
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    std::vector<uint8_t> cbuf(n);
    for (int64_t i = 0; i < n; ++i) {
        cbuf[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 7 + 3) % 3 == 0 ? 0 : 1);
    }
    const std::vector<float> x0 = ramp(n, 29u);
    std::vector<float> xbuf = x0;
    std::vector<float> ybuf = ramp(n, 31u);
    std::vector<float> sep(n, 0.0f);

    TensorView c(shape_of(dims, 2), DataType::u8, cbuf.data());
    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView y(shape_of(dims, 2), DataType::f32, ybuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    auto op = Where::create(WhereAttributes{}, Backend::CPU);
    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {c, x, y};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("where", run, [&] { xbuf = x0; }, out_sep, out_alias, n * sizeof(float));
}

// ============================================================
// Row-wise: the row is fully read (stats included) before it is written
// ============================================================

NNOPS_TEST(inplace_softmax)
{
    const int64_t dims[] = {8, 24};
    constexpr int64_t n = 8 * 24;
    const std::vector<float> x0 = ramp(n, 37u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("softmax", run, [&] { xbuf = x0; }, out_sep, out_alias, n * sizeof(float));
}

NNOPS_TEST(inplace_norm)
{
    const int64_t dims[] = {8, 24};
    constexpr int64_t n = 8 * 24;
    const std::vector<float> x0 = ramp(n, 41u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);
    std::vector<float> scale = ramp(24, 43u);
    std::vector<float> bias = ramp(24, 47u);

    const int64_t row[] = {24};
    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView s(shape_of(row, 1), DataType::f32, scale.data());
    TensorView b(shape_of(row, 1), DataType::f32, bias.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    attrs.axis = -1;
    auto op = Norm::create(attrs, Backend::CPU);

    const TensorDesc in_d[] = {x.desc(), s.desc(), b.desc()};
    const TensorDesc out_d[] = {out_sep.desc()};
    NNOPS_EXPECT_TRUE(op->supports_inplace(in_d, out_d));

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x, s, b};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("norm/layernorm", run, [&] { xbuf = x0; }, out_sep, out_alias,
                       n * sizeof(float));
}

NNOPS_TEST(inplace_cumsum)
{
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 53u);

    for (bool reverse : {false, true}) {
        std::vector<float> xbuf = x0;
        std::vector<float> sep(n, 0.0f);
        TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
        TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
        TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

        CumSumAttributes attrs;
        attrs.axis = 1;
        attrs.reverse = reverse;
        auto op = CumSum::create(attrs, Backend::CPU);

        const auto run = [&](const ComputeContext& ctx, TensorView& o) {
            const TensorView ins[] = {x};
            return op->compute(o, ins, ctx);
        };
        expect_alias_equal("cumsum", run, [&] { xbuf = x0; }, out_sep, out_alias,
                           n * sizeof(float));
    }
}

NNOPS_TEST(inplace_rope)
{
    // Rank 3: [batch, seq, head_dim]. Both pair layouts read both halves of a
    // pair before storing either.
    const int64_t dims[] = {2, 4, 8};
    constexpr int64_t n = 2 * 4 * 8;
    const std::vector<float> x0 = ramp(n, 59u);

    for (bool interleaved : {true, false}) {
        std::vector<float> xbuf = x0;
        std::vector<float> sep(n, 0.0f);
        TensorView x(shape_of(dims, 3), DataType::f32, xbuf.data());
        TensorView out_sep(shape_of(dims, 3), DataType::f32, sep.data());
        TensorView out_alias(shape_of(dims, 3), DataType::f32, xbuf.data());

        RoPEAttributes attrs;
        attrs.interleaved = interleaved;
        auto op = RoPE::create(attrs, Backend::CPU);

        const auto run = [&](const ComputeContext& ctx, TensorView& o) {
            const TensorView ins[] = {x};
            return op->compute(o, ins, ctx);
        };
        expect_alias_equal("rope", run, [&] { xbuf = x0; }, out_sep, out_alias,
                           n * sizeof(float));
    }
}

NNOPS_TEST(inplace_cast)
{
    // Only the same-dtype cast can alias: the descriptor carries the width.
    const int64_t dims[] = {4, 16};
    constexpr int64_t n = 64;
    const std::vector<float> x0 = ramp(n, 61u);
    std::vector<float> xbuf = x0;
    std::vector<float> sep(n, 0.0f);

    TensorView x(shape_of(dims, 2), DataType::f32, xbuf.data());
    TensorView out_sep(shape_of(dims, 2), DataType::f32, sep.data());
    TensorView out_alias(shape_of(dims, 2), DataType::f32, xbuf.data());

    CastAttributes attrs;
    attrs.to = DataType::f32;
    auto op = Cast::create(attrs, Backend::CPU);

    const auto run = [&](const ComputeContext& ctx, TensorView& o) {
        const TensorView ins[] = {x};
        return op->compute(o, ins, ctx);
    };
    expect_alias_equal("cast", run, [&] { xbuf = x0; }, out_sep, out_alias, n * sizeof(float));
}

// ============================================================
// The query itself
// ============================================================

NNOPS_TEST(supports_inplace_geometry)
{
    const int64_t dims[] = {4, 16};
    const int64_t other[] = {4, 8};
    std::vector<float> buf(64, 0.5f);
    std::vector<float> buf2(32, 0.5f);
    std::vector<uint16_t> buf16(64, 0);

    TensorView a(shape_of(dims, 2), DataType::f32, buf.data());
    TensorView b(shape_of(other, 2), DataType::f32, buf2.data());
    TensorView c(shape_of(dims, 2), DataType::f16, buf16.data());
    TensorView l(shape_of(dims, 2), DataType::f32, buf.data(), TensorLayout::NHWC);

    auto op = Activation::create(Backend::CPU);
    const TensorDesc ad[] = {a.desc()};
    const TensorDesc bd[] = {b.desc()};
    const TensorDesc cd[] = {c.desc()};
    const TensorDesc ld[] = {l.desc()};

    NNOPS_EXPECT_TRUE(op->supports_inplace(ad, ad));
    NNOPS_EXPECT_TRUE(!op->supports_inplace(ad, bd)); // extent
    NNOPS_EXPECT_TRUE(!op->supports_inplace(ad, cd)); // dtype
    NNOPS_EXPECT_TRUE(!op->supports_inplace(ad, ld)); // layout

    // A backend whose kernel has not been verified never opts in.
    auto cuda = Activation::create(Backend::CUDA);
    NNOPS_EXPECT_TRUE(!cuda->supports_inplace(ad, ad));

    // An operator that did not opt in answers false by default.
    auto mm = MatMul::create(MatMulAttributes{}, Backend::CPU);
    const TensorDesc mm_in[] = {a.desc(), a.desc()};
    NNOPS_EXPECT_TRUE(!mm->supports_inplace(mm_in, ad));
}
