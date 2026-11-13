/// @file test_bf16.cpp
/// @brief bf16 dtype support across the CPU operators: an f32-vs-bf16 oracle
///        sweep.
///
/// For every op the same call is run twice: once with every fp tensor in f32
/// (the oracle), once with the same values stored as bf16. Both runs draw from
/// one seeded generator, so the bf16 run differs only by the rounding of its
/// inputs plus whatever the kernel does. The outputs are compared with a
/// per-family relative tolerance, sized for bf16's 8-bit mantissa and — on the
/// GEMM-family ops — for the kc-blocked accumulation through C, which is the
/// same drift the f16 kernels already carry (see kBF16AccumTol).
///
/// The sweep exists because validate() admits bf16 wherever it admits f16
/// (dtype_in), so an op whose kernel never learned bf16 would either assert or
/// silently take an f16 path. MatMul has its own numeric suite in test_matmul;
/// this file covers the rest of the operator surface, including both Conv2D
/// layouts, the transpose/depthwise variants, conv1d/conv3d, the normalization
/// and softmax kernels, embedding, pooling, the elementwise set and the
/// movement ops.
///
/// On x86-64 a bf16 call must be refused rather than mis-run; the final test
/// asserts that.

#include "nnops/core/data_type.hpp"
#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/activation.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/conv3d.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/embed.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/linear_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"
#include "nnops/ops/moe.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/pool1d.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/rope.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/transpose_conv3d.hpp"
#include "nnops/ops/unary.hpp"
#include "nnops/ops/where.hpp"

#include "common/random_tensor.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NNOPS_ARCH_AARCH64)
#include "backend/cpu/aarch64/mma_bf16.hpp"
#endif

using namespace nnops;
using nnops::backend::cpu::bfloat16;
using nnops::test::bf16_to_f32;
using nnops::test::f32_to_bf16;

namespace {

/// One operator input: its shape, layout, and how its values are drawn.
struct InSpec {
    std::vector<int64_t> shape;
    TensorLayout layout = TensorLayout::NCHW;
    bool index = false;      ///< s32 index input (gather / embed); dtype-independent
    int64_t index_mod = 1;   ///< indices land in [0, index_mod)
    bool cond = false;       ///< u8 0/1 mask input (where)
    float lo = -1.0f;        ///< fp value range
    float hi = 1.0f;
    /// When > 0, values snap to multiples of `step`. A step a power of two is
    /// exact in bf16, so a rank-ordered output (topk) cannot reorder between
    /// the f32 and bf16 runs over near-equal neighbours.
    float step = 0.0f;
    bool empty = false;      ///< an absent optional, passed as a default view
};

using OpFactory = std::function<std::unique_ptr<OpBase>()>;

constexpr size_t kAlign = 64;

void* align64(void* p) noexcept
{
    const auto a = reinterpret_cast<uintptr_t>(p);
    return reinterpret_cast<void*>((a + kAlign - 1) & ~static_cast<uintptr_t>(kAlign - 1));
}

/// One output of a run, in the output's storage order and as f32 values.
struct RunOut {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

/// Run `make()` over `ins` with every fp tensor stored as `dt`; index inputs
/// stay s32 in both runs.
std::vector<RunOut> run_op(const OpFactory& make, std::span<const InSpec> ins,
                           DataType dt, uint64_t seed)
{
    auto op = make();

    // All storage first, then every view: a TensorView holds a raw pointer, so
    // nothing it points at may move once the views exist.
    std::vector<std::vector<float>> f32_st;
    std::vector<std::vector<bfloat16>> bf_st;
    std::vector<std::vector<int32_t>> idx_st;
    std::vector<std::vector<uint8_t>> cond_st;
    f32_st.resize(ins.size());
    bf_st.resize(ins.size());
    idx_st.resize(ins.size());
    cond_st.resize(ins.size());
    nnops::test::XorShift128 rng(seed);
    for (size_t i = 0; i < ins.size(); ++i) {
        if (ins[i].empty) { continue; }
        int64_t n = 1;
        for (int64_t d : ins[i].shape) { n *= d; }
        if (ins[i].index) {
            idx_st[i].resize(static_cast<size_t>(n));
            for (int64_t j = 0; j < n; ++j) {
                idx_st[i][static_cast<size_t>(j)] = static_cast<int32_t>(
                    rng.next_u64() % static_cast<uint64_t>(ins[i].index_mod));
            }
        } else if (ins[i].cond) {
            cond_st[i].resize(static_cast<size_t>(n));
            for (int64_t j = 0; j < n; ++j) {
                cond_st[i][static_cast<size_t>(j)] =
                    static_cast<uint8_t>(rng.next_u64() & 1u);
            }
        } else {
            f32_st[i].resize(static_cast<size_t>(n));
            rng.fill_float(f32_st[i].data(), n, ins[i].lo, ins[i].hi);
            if (ins[i].step > 0.0f) {
                for (float& v : f32_st[i]) {
                    v = std::round(v / ins[i].step) * ins[i].step;
                }
            }
            if (dt != DataType::f32) { bf_st[i] = f32_to_bf16(f32_st[i]); }
        }
    }

    std::vector<TensorView> views;
    views.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto shape = std::span<const int64_t>(ins[i].shape);
        if (ins[i].empty) {
            views.emplace_back();
        } else if (ins[i].index) {
            views.emplace_back(shape, DataType::s32, idx_st[i].data(), ins[i].layout);
        } else if (ins[i].cond) {
            views.emplace_back(shape, DataType::u8, cond_st[i].data(), ins[i].layout);
        } else if (dt == DataType::f32) {
            views.emplace_back(shape, DataType::f32, f32_st[i].data(), ins[i].layout);
        } else {
            views.emplace_back(shape, DataType::bf16, bf_st[i].data(), ins[i].layout);
        }
    }

    std::vector<TensorDesc> descs;
    descs.reserve(views.size());
    for (const auto& v : views) { descs.push_back(v.desc()); }
    const auto odescs = op->getOutputTensorDesc(descs);

    // Outputs are allocated over-aligned: the kernels store with 16-byte vector
    // writes, and a std::vector's buffer is only max_align_t.
    std::vector<std::vector<uint8_t>> obuf(odescs.size());
    std::vector<TensorView> outs;
    outs.reserve(odescs.size());
    for (size_t i = 0; i < odescs.size(); ++i) {
        const int64_t bytes = odescs[i].numel() *
                              static_cast<int64_t>(data_type_size(odescs[i].dtype));
        obuf[i].resize(static_cast<size_t>(bytes) + kAlign);
        const auto shape = std::span<const int64_t>(odescs[i].dims.data(),
                                                    static_cast<size_t>(odescs[i].rank));
        outs.emplace_back(shape, odescs[i].dtype, align64(obuf[i].data()),
                          odescs[i].layout);
    }

    std::vector<uint8_t> wsbuf(op->getWorkspaceSize(descs, odescs) + kAlign);
    void* ws = wsbuf.size() > kAlign ? align64(wsbuf.data()) : nullptr;

    const Status st = op->compute(outs, views, ComputeContext{}, ws);
    if (!is_ok(st)) {
        throw std::runtime_error(std::string("compute returned status ")
                                 + std::to_string(static_cast<int>(st)));
    }

    std::vector<RunOut> res;
    res.reserve(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        RunOut r;
        r.shape.assign(odescs[i].dims.begin(),
                       odescs[i].dims.begin() + odescs[i].rank);
        const int64_t n = odescs[i].numel();
        r.values.resize(static_cast<size_t>(n));
        const void* src = outs[i].ptr<const void>();
        switch (odescs[i].dtype) {
        case DataType::f32:
            std::memcpy(r.values.data(), src, static_cast<size_t>(n) * sizeof(float));
            break;
        case DataType::bf16: {
            std::vector<bfloat16> tmp(static_cast<size_t>(n));
            std::memcpy(tmp.data(), src, static_cast<size_t>(n) * sizeof(bfloat16));
            r.values = bf16_to_f32(tmp);
            break;
        }
        case DataType::f16: {
            std::vector<backend::cpu::half> tmp(static_cast<size_t>(n));
            std::memcpy(tmp.data(), src, static_cast<size_t>(n) * sizeof(backend::cpu::half));
            r.values = nnops::test::f16_to_f32(tmp);
            break;
        }
        case DataType::s32:
            for (int64_t j = 0; j < n; ++j) {
                r.values[static_cast<size_t>(j)] =
                    static_cast<float>(outs[i].ptr<int32_t>()[j]);
            }
            break;
        case DataType::s64:
            for (int64_t j = 0; j < n; ++j) {
                r.values[static_cast<size_t>(j)] =
                    static_cast<float>(outs[i].ptr<int64_t>()[j]);
            }
            break;
        default:
            throw std::runtime_error("run_op: unexpected output dtype");
        }
        res.push_back(std::move(r));
    }
    return res;
}

/// Run the f32 oracle and the bf16 call, and throw with the worst mismatch if
/// any element is off by more than `tol` relative to its scale.
void check_case(const char* name, const OpFactory& make, std::vector<InSpec> ins,
                float tol, uint64_t seed = 12345)
{
    std::vector<RunOut> ref, got;
    try {
        ref = run_op(make, ins, DataType::f32, seed);
        got = run_op(make, ins, DataType::bf16, seed);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string(name) + ": " + e.what());
    }

    if (ref.size() != got.size()) {
        throw std::runtime_error(std::string(name) + ": output count differs");
    }
    for (size_t o = 0; o < ref.size(); ++o) {
        if (ref[o].values.size() != got[o].values.size()) {
            throw std::runtime_error(std::string(name) + ": output " +
                                     std::to_string(o) + " size differs");
        }
        float worst = 0.0f;
        size_t worst_i = 0;
        for (size_t i = 0; i < ref[o].values.size(); ++i) {
            const float r = ref[o].values[i];
            const float g = got[o].values[i];
            const float scale = std::max(1.0f, std::max(std::abs(r), std::abs(g)));
            const float rel = std::abs(g - r) / scale;
            if (rel > worst) { worst = rel; worst_i = i; }
        }
        if (!(worst <= tol)) {
            char msg[320];
            std::snprintf(msg, sizeof(msg),
                          "%s: output %zu element %zu: bf16 %g vs f32 %g "
                          "(rel %.4g > tol %.4g)",
                          name, o, worst_i, got[o].values[worst_i],
                          ref[o].values[worst_i], worst, tol);
            throw std::runtime_error(msg);
        }
    }
}

// ---- elementwise / reduction ------------------------------------------------

void run_elementwise()
{
    // One activation, one unary, one eltwise, one clamp, one reduce.
    for (auto at : {ActivationType::Relu, ActivationType::Gelu,
                    ActivationType::Silu, ActivationType::Sigmoid}) {
        ActivationAttributes a;
        a.type = at;
        check_case("activation", [a] { return Activation::create(a, Backend::CPU); },
                   {{{2, 3, 4, 5}}}, 2e-2f);
    }
    for (auto ut : {UnaryType::Exp, UnaryType::Tanh, UnaryType::Neg}) {
        UnaryAttributes a;
        a.type = ut;
        check_case("unary", [a] { return Unary::create(a, Backend::CPU); },
                   {{{3, 5, 7}}}, 2e-2f);
    }
    for (auto et : {EltwiseType::Add, EltwiseType::Mul}) {
        EltwiseAttributes a;
        a.type = et;
        check_case("eltwise", [a] { return Eltwise::create(a, Backend::CPU); },
                   {{{2, 3, 8}}, {{2, 3, 8}}}, 2e-2f);
    }
    ClampAttributes c;
    c.min_val = -0.5f;
    c.max_val = 0.5f;
    check_case("clamp", [c] { return Clamp::create(c, Backend::CPU); },
               {{{2, 16}}}, 2e-2f);

    for (auto rt : {ReduceType::Sum, ReduceType::Mean, ReduceType::Max}) {
        ReduceAttributes a;
        a.type = rt;
        a.axes = {1};
        check_case("reduce", [a] { return Reduce::create(a, Backend::CPU); },
                   {{{2, 4, 8}}}, 5e-2f);
    }
}

// ---- softmax / norm ---------------------------------------------------------

void run_normalization()
{
    SoftmaxAttributes sa;
    sa.axis = -1;
    check_case("softmax", [sa] { return Softmax::create(sa, Backend::CPU); },
               {{{2, 5, 7}}}, 3e-2f);

    {
        NormAttributes a;
        a.type = NormType::LayerNorm;
        a.axis = -1;
        check_case("layernorm", [a] { return Norm::create(a, Backend::CPU); },
                   {{{2, 8, 16}}, {{16}}, {{16}}}, 3e-2f);
    }
    {
        NormAttributes a;
        a.type = NormType::RMSNorm;
        a.axis = -1;
        check_case("rmsnorm", [a] { return Norm::create(a, Backend::CPU); },
                   {{{2, 8, 16}}, {{16}}}, 3e-2f);
    }
    {
        NormAttributes a;
        a.type = NormType::GroupNorm;
        a.num_groups = 4;
        check_case("groupnorm", [a] { return Norm::create(a, Backend::CPU); },
                   {{{2, 8, 4, 4}}, {{8}}, {{8}}}, 3e-2f);
    }
}

// ---- conv family ------------------------------------------------------------

void run_conv()
{
    {
        Conv2DAttributes a;
        a.kernel_size = {3, 3};
        a.padding = {1, 1};
        check_case("conv2d_cf_3x3",
                   [a] { return Conv2D::create(a, Backend::CPU); },
                   {{{1, 4, 8, 8}}, {{6, 4, 3, 3}}}, 5e-2f);
    }
    {
        // kdim = 3·9 = 27: the pack's k groups past a 4-multiple, the case the
        // BFMMLA panel stride has to agree with the packer on.
        Conv2DAttributes a;
        a.kernel_size = {3, 3};
        check_case("conv2d_cf_odd_k",
                   [a] { return Conv2D::create(a, Backend::CPU); },
                   {{{1, 3, 9, 9}}, {{5, 3, 3, 3}}}, 5e-2f);
    }
    {
        Conv2DAttributes a;
        a.kernel_size = {1, 1};
        a.epilogue.type = EpilogueActivateType::Relu;
        check_case("conv2d_cf_1x1_relu",
                   [a] { return Conv2D::create(a, Backend::CPU); },
                   {{{1, 4, 6, 6}}, {{6, 4, 1, 1}}, {{6}}}, 5e-2f);
    }
    {
        Conv2DAttributes a;
        a.kernel_size = {3, 3};
        a.padding = {1, 1};
        check_case("conv2d_cl_3x3",
                   [a] { return Conv2D::create(a, Backend::CPU); },
                   {{{1, 8, 8, 4}}, {{3, 3, 4, 6}}}, 5e-2f);
    }
    {
        Conv2DAttributes a;
        a.kernel_size = {1, 1};
        check_case("conv2d_cl_1x1",
                   [a] { return Conv2D::create(a, Backend::CPU); },
                   {{{1, 6, 6, 4}}, {{1, 1, 4, 6}}, {{6}}}, 5e-2f);
    }
    {
        DepthwiseConvAttributes a;
        a.kernel_size = {1, 3, 3};
        a.padding = {0, 1, 1};
        check_case("depthwise",
                   [a] { return DepthwiseConv::create(a, Backend::CPU); },
                   {{{1, 4, 8, 8}}, {{4, 1, 3, 3}}}, 5e-2f);
    }
    {
        DepthwiseTransposeConv2DAttributes a;
        a.kernel_size = {3, 3};
        a.stride = {2, 2};
        a.padding = {1, 1};
        a.output_padding = {1, 1};
        check_case("depthwise_transpose",
                   [a] { return DepthwiseTransposeConv2D::create(a, Backend::CPU); },
                   {{{1, 4, 6, 6}}, {{4, 1, 3, 3}}}, 5e-2f);
    }
    {
        TransposeConv2DAttributes a;
        a.kernel_size = {3, 3};
        a.stride = {2, 2};
        a.padding = {1, 1};
        a.output_padding = {1, 1};
        check_case("transpose_conv2d",
                   [a] { return TransposeConv2D::create(a, Backend::CPU); },
                   {{{1, 3, 6, 6}}, {{3, 4, 3, 3}}}, 5e-2f);
    }
    {
        Conv1DAttributes a;
        a.kernel_size = 3;
        a.padding = 1;
        check_case("conv1d",
                   [a] { return Conv1D::create(a, Backend::CPU); },
                   {{{1, 4, 20}}, {{6, 4, 3}}}, 5e-2f);
    }
    {
        Conv3DAttributes a;
        a.kernel_size = {2, 2, 2};
        check_case("conv3d",
                   [a] { return Conv3D::create(a, Backend::CPU); },
                   {{{1, 2, 5, 5, 5}}, {{3, 2, 2, 2, 2}}}, 5e-2f);
    }
}

// ---- matmul family ----------------------------------------------------------

void run_matmul_family()
{
    {
        // [M, K] · [N, K]ᵀ: the plain Linear.
        LinearAttributes a;
        check_case("linear", [a] { return Linear::create(a, Backend::CPU); },
                   {{{4, 16}}, {{8, 16}}, {{8}}}, 1e-1f);
    }
    {
        // The gated (SwiGLU) form: one GEMM over [2N, K], then the combine.
        LinearAttributes a;
        a.gated = true;
        check_case("linear_gated", [a] { return Linear::create(a, Backend::CPU); },
                   {{{4, 16}}, {{8, 16}}, {{8}}}, 1e-1f);
    }
    {
        EmbedAttributes a;
        check_case("embed", [a] { return Embed::create(a, Backend::CPU); },
                   {{{10, 8}}, {{3, 4}, TensorLayout::NCHW, true, 10}}, 2e-2f);
    }
    {
        // One expert, top-1, identity activation: the MoE router + expert GEMM
        // path without needing the gather order to match a reference.
        MoEAttributes a;
        a.k = 1;
        a.activation = MoEActivation::Identity;
        check_case("moe", [a] { return MoE::create(a, Backend::CPU); },
                   {{{4, 8}}, {{4, 2}}, {{2, 8, 8}},
                    {.empty = true}, {{2, 8, 8}}, {.empty = true}}, 1e-1f);
    }
}

// ---- attention --------------------------------------------------------------

void run_attention()
{
    {
        // [B, S, H·D] merged layout, H = Hkv = 4, no mask, no causal.
        MultiHeadAttentionAttributes a;
        a.num_heads = 4;
        check_case("multi_head_attention",
                   [a] { return MultiHeadAttention::create(a, Backend::CPU); },
                   {{{2, 4, 32}}, {{2, 6, 32}}, {{2, 6, 32}}}, 1e-1f);
    }
    {
        // Grouped heads (H = 4, Hkv = 2) with the causal mask: the decode and
        // standard routes share this dispatch.
        GroupQueryAttentionAttributes a;
        a.num_heads = 4;
        a.kv_num_heads = 2;
        a.causal = true;
        check_case("group_query_attention",
                   [a] { return GroupQueryAttention::create(a, Backend::CPU); },
                   {{{2, 4, 32}}, {{2, 6, 16}}, {{2, 6, 16}}}, 1e-1f);
    }
    {
        // Decode route: one query row and no mask, so both products are GEMV.
        MultiHeadAttentionAttributes a;
        a.num_heads = 4;
        check_case("multi_head_attention_decode",
                   [a] { return MultiHeadAttention::create(a, Backend::CPU); },
                   {{{2, 1, 32}}, {{2, 32, 32}}, {{2, 32, 32}}}, 1e-1f);
    }
    {
        // Flash route: Sq·Sk past kFlashMinScores (1 << 16), so the streaming
        // kernel replaces the standard one.
        MultiHeadAttentionAttributes a;
        a.num_heads = 2;
        check_case("multi_head_attention_flash",
                   [a] { return MultiHeadAttention::create(a, Backend::CPU); },
                   {{{1, 300, 16}}, {{1, 384, 16}}, {{1, 384, 16}}}, 1e-1f);
    }
    {
        // Recurrent form: [B, H, S, D] in, output plus the [B, Hkv, D, D] state.
        LinearAttentionAttributes a;
        a.num_heads = 2;
        a.num_kv_heads = 1;
        a.head_dim = 8;
        check_case("linear_attention",
                   [a] { return LinearAttention::create(a, Backend::CPU); },
                   {{{1, 2, 4, 8}}, {{1, 1, 4, 8}}, {{1, 1, 4, 8}}, {{1, 1, 4, 8}}},
                   1e-1f);
    }
    {
        // The absorbed latent form shares the attention kernels, so it must
        // carry bf16 too. q width = H·Dkv, kv width = Hkv·Dkv.
        MultiLatentAttentionAttributes a;
        a.num_heads = 2;
        a.kv_num_heads = 1;
        a.kv_lora_rank = 12;
        a.qk_rope_head_dim = 4;
        a.qk_nope_head_dim = 8;
        a.v_head_dim = 8;
        a.causal = false;
        check_case("multi_latent_attention",
                   [a] { return MultiLatentAttention::create(a, Backend::CPU); },
                   {{{1, 4, 32}}, {{1, 6, 16}}}, 1e-1f);
    }
}

// ---- pooling / movement -----------------------------------------------------

void run_pooling_and_movement()
{
    for (auto pt : {PoolingType::Max, PoolingType::Average}) {
        PoolingAttributes a;
        a.type = pt;
        a.kernel_shape = {1, 3, 3};
        a.stride = {1, 2, 2};
        a.padding = {0, 1, 1};
        check_case("pooling", [a] { return Pooling::create(a, Backend::CPU); },
                   {{{1, 4, 8, 8}}}, 3e-2f);
    }
    {
        ConcatAttributes a;
        a.axis = 1;
        check_case("concat", [a] { return Concat::create(a, Backend::CPU); },
                   {{{2, 3, 4}}, {{2, 5, 4}}}, 2e-2f);
    }
    {
        TileAttributes a;
        a.repeats = {2, 1};
        check_case("tile", [a] { return Tile::create(a, Backend::CPU); },
                   {{{2, 3}}}, 2e-2f);
    }
    {
        PermuteAttributes a;
        a.perm = {2, 0, 1};
        check_case("permute", [a] { return Permute::create(a, Backend::CPU); },
                   {{{2, 3, 4}}}, 2e-2f);
    }
    {
        SliceAttributes a;
        a.starts = {1};
        a.ends = {7};
        a.axes = {1};
        check_case("slice", [a] { return Slice::create(a, Backend::CPU); },
                   {{{3, 8}}}, 2e-2f);
    }
    {
        PadAttributes a;
        a.mode = PadAttributes::Mode::Constant;
        a.pads = {0, 1, 1, 0};
        a.constant_value = 0.5f;
        check_case("pad", [a] { return Pad::create(a, Backend::CPU); },
                   {{{2, 3}}}, 2e-2f);
    }
    {
        GatherAttributes a;
        a.axis = 0;
        check_case("gather", [a] { return Gather::create(a, Backend::CPU); },
                   {{{4, 5}}, {{6, 2}, TensorLayout::NCHW, true, 4}}, 2e-2f);
    }
    {
        auto op = [] {
            return LayoutConvert::create(TensorLayout::NHWC);
        };
        check_case("layout_convert", op, {{{2, 3, 4, 5}}}, 5e-3f);
    }
}

}  // namespace

// ---- the remaining float operators -----------------------------------------

void run_misc_ops()
{
    for (bool rev : {false, true}) {
        for (bool excl : {false, true}) {
            CumSumAttributes a;
            a.axis = 1;
            a.reverse = rev;
            a.exclusive = excl;
            check_case("cumsum", [a] { return CumSum::create(a, Backend::CPU); },
                       {{{2, 6}}}, 2e-2f);
        }
    }
    {
        SplitAttributes a;
        a.axis = 1;
        a.split = {2, 6};
        check_case("split", [a] { return Split::create(a, Backend::CPU); },
                   {{{3, 8}}}, 2e-2f);
    }
    {
        // Values on a 1/8 grid: exactly representable in bf16, so the top-k
        // ordering is the same in both runs.
        TopKAttributes a;
        a.k = 3;
        a.axis = -1;
        check_case("topk", [a] { return TopK::create(a, Backend::CPU); },
                   {{{.shape = {4, 6}, .lo = 0.0f, .hi = 1.0f, .step = 0.125f}}},
                   2e-2f);
    }
    {
        // Resize is channel-last: [N, H, W, C] in, [N, OH, OW, C] out.
        for (auto mode : {ResizeMode::Nearest, ResizeMode::Linear}) {
            ResizeAttributes a;
            a.mode = mode;
            a.output_size = {0, 5, 7};
            check_case("resize", [a] { return Resize::create(a, Backend::CPU); },
                       {{{.shape = {1, 4, 5, 3}, .layout = TensorLayout::NHWC}}},
                       3e-2f);
        }
    }
    {
        // [B, S, H, R]: adjacent pairs rotate, each row one position further.
        RoPEAttributes a;
        a.base = 500.0f;
        a.interleaved = true;
        check_case("rope", [a] { return RoPE::create(a, Backend::CPU); },
                   {{{2, 3, 2, 8}}}, 5e-2f);
    }
    {
        Pool1DAttributes a;
        a.kernel_shape = 3;
        a.stride = 2;
        a.padding = 1;
        check_case("pool1d", [a] { return Pool1D::create(a, Backend::CPU); },
                   {{{1, 4, 9}}}, 3e-2f);
    }
    {
        TransposeConv3DAttributes a;
        a.kernel_size = {2, 2, 2};
        check_case("transpose_conv3d",
                   [a] { return TransposeConv3D::create(a, Backend::CPU); },
                   {{{1, 2, 3, 3, 3}}, {{2, 3, 2, 2, 2}}}, 5e-2f);
    }
    {
        // cond is u8; the two value operands decide the output dtype.
        WhereAttributes a;
        std::vector<InSpec> where_ins(3);
        where_ins[0].shape = {2, 3};
        where_ins[0].cond = true;
        where_ins[1].shape = {2, 3};
        where_ins[2].shape = {2, 3};
        check_case("where", [a] { return Where::create(a, Backend::CPU); },
                   where_ins, 2e-2f);
    }
}

NNOPS_TEST(bf16_elementwise) { run_elementwise(); }
NNOPS_TEST(bf16_normalization) { run_normalization(); }
NNOPS_TEST(bf16_conv) { run_conv(); }
NNOPS_TEST(bf16_matmul_family) { run_matmul_family(); }
NNOPS_TEST(bf16_attention) { run_attention(); }
NNOPS_TEST(bf16_pooling_and_movement) { run_pooling_and_movement(); }
NNOPS_TEST(bf16_misc_ops) { run_misc_ops(); }

#if defined(NNOPS_ARCH_AARCH64)
namespace {

using backend::cpu::bfloat16;
namespace mma = backend::cpu::aarch64;

/// One mrxncxkc tile through the three B forms the MMA entries accept, against
/// a naive f32 GEMM: packed panels, those same panels through the direct entry,
/// and a caller's own [K][N] matrix read in place (`ldb > 0`, the in-place 1x1
/// conv route). The shapes under it straddle a panel edge (16 columns) and a
/// k-group edge (4), which is where the tail paths live.
template <bool Zero, int Seed>
void check_mma_tile(int mr, int nc, int kc, float lo, float hi) {
    const int ldc = nc + 3;
    std::mt19937 rng(static_cast<unsigned>(mr * 977 + nc * 31 + kc));
    std::uniform_real_distribution<float> rnd(-1.0f, 1.0f);

    std::vector<bfloat16> a(static_cast<size_t>(mr) * kc);
    std::vector<bfloat16> bmat(static_cast<size_t>(kc) * nc);
    std::vector<bfloat16> c0(static_cast<size_t>(mr) * ldc);
    std::vector<bfloat16> bias(static_cast<size_t>(std::max(mr, nc)) + 8);
    for (auto& v : a) { v = bfloat16(rnd(rng)); }
    for (auto& v : bmat) { v = bfloat16(rnd(rng)); }
    for (auto& v : c0) { v = bfloat16(rnd(rng)); }
    for (auto& v : bias) { v = bfloat16(rnd(rng)); }

    std::vector<float> want(static_cast<size_t>(mr) * nc);
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nc; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < kc; ++k) {
                acc += static_cast<float>(a[static_cast<size_t>(i) * kc + k]) *
                       static_cast<float>(bmat[static_cast<size_t>(k) * nc + j]);
            }
            if (Seed == 1) { acc += static_cast<float>(bias[i]); }
            else if (Seed == 2) { acc += static_cast<float>(bias[j]); }
            if constexpr (!Zero) {
                acc += static_cast<float>(c0[static_cast<size_t>(i) * ldc + j]);
            }
            want[static_cast<size_t>(i) * nc + j] = std::min(std::max(acc, lo), hi);
        }
    }

    // Packed B: ceil(nc / 16) panels, each one bf16_b_panel_stride(kc) long.
    const int panels = (nc + 15) / 16;
    std::vector<bfloat16> bpack(static_cast<size_t>(panels) * mma::bf16_b_panel_stride(kc),
                                bfloat16(0.0f));
    for (int p = 0; p < panels; ++p) {
        mma::pack_rhs_bf16_copy(
            bpack.data() + static_cast<size_t>(p) * mma::bf16_b_panel_stride(kc),
            bmat.data() + 16 * p, nc, kc, 1.0f, std::min(16, nc - 16 * p));
    }

    const int rung = (mr > 4) ? 0 : (mr > 1 ? 1 : 2);
    const auto expect = [&](const std::vector<bfloat16>& c) {
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nc; ++j) {
                const float got = static_cast<float>(c[static_cast<size_t>(i) * ldc + j]);
                const float ref = want[static_cast<size_t>(i) * nc + j];
                NNOPS_EXPECT_NEAR(got, ref, 3e-2f * (1.0f + std::fabs(ref)));
            }
        }
    };

    const auto pack_fn = mma::mma_pack_fns_bf16<Zero, Seed>[rung];
    const auto direct_fn = mma::mma_direct_fns_bf16<Zero, Seed>[rung];
    {
        std::vector<bfloat16> c = c0;
        pack_fn(mr, nc, kc, c.data(), ldc, a.data(), bpack.data(), lo, hi, bias.data());
        expect(c);
    }
    {
        std::vector<bfloat16> c = c0;
        direct_fn(mr, nc, kc, c.data(), ldc, a.data(), kc, bpack.data(), /*ldb=*/-1,
                  lo, hi, bias.data());
        expect(c);
    }
    {
        std::vector<bfloat16> c = c0;
        direct_fn(mr, nc, kc, c.data(), ldc, a.data(), kc, bmat.data(), /*ldb=*/nc,
                  lo, hi, bias.data());
        expect(c);
    }
}

}  // namespace

NNOPS_TEST(bf16_mma_routes) {
    // mr straddles the rungs ({8,4,1}), nc the 16-column panel, kc the 4-value
    // k-group; every entry in the seed × zero matrix runs on each shape.
    const int shapes[][3] = {{1, 1, 1},  {1, 33, 1}, {2, 17, 3}, {3, 15, 1},
                             {4, 16, 7},  {5, 33, 4}, {7, 31, 32}, {8, 8, 8},
                             {8, 40, 13}, {6, 47, 2}};
    for (const auto& s : shapes) {
        const int mr = s[0], nc = s[1], kc = s[2];
        check_mma_tile<false, 0>(mr, nc, kc, -1e9f, 1e9f);
        check_mma_tile<true, 0>(mr, nc, kc, -1e9f, 1e9f);
        check_mma_tile<false, 1>(mr, nc, kc, -0.5f, 0.5f);
        check_mma_tile<false, 2>(mr, nc, kc, -1e9f, 1e9f);
        check_mma_tile<true, 2>(mr, nc, kc, 0.0f, 1e9f);
    }
}
#endif

#if !defined(NNOPS_ARCH_AARCH64)
NNOPS_TEST(bf16_unsupported_on_x86) {
    // bf16 has no x86 kernel: validate() must refuse it rather than dispatch.
    LinearAttributes a;
    auto op = Linear::create(a, Backend::CPU);
    std::vector<float> x(4 * 16), w(8 * 16), b(8);
    const int64_t xs_shape[] = {4, 16};
    const int64_t ws_shape[] = {8, 16};
    const int64_t bs_shape[] = {8};
    TensorView xv(xs_shape, DataType::bf16, x.data());
    TensorView wv(ws_shape, DataType::bf16, w.data());
    TensorView bv(bs_shape, DataType::bf16, b.data());
    const TensorDesc ins[] = {xv.desc(), wv.desc(), bv.desc()};
    const auto od = op->getOutputTensorDesc(ins);
    NNOPS_EXPECT_EQ(op->validate(ins, od), Status::unsupported);

    // An operator that lists no dtype in validate() is caught by the shared
    // view check instead, so compute() refuses rather than dispatching into a
    // kernel that has no bf16 case.
    TopKAttributes ta;
    ta.k = 2;
    auto topk = TopK::create(ta, Backend::CPU);
    std::vector<float> t(8);
    const int64_t t_shape[] = {8};
    TensorView tv(t_shape, DataType::bf16, t.data());
    const TensorDesc tin[] = {tv.desc()};
    const auto tout = topk->getOutputTensorDesc(tin);
    NNOPS_EXPECT_EQ(tout.size(), size_t{2});
    std::vector<float> tv_buf(static_cast<size_t>(tout[0].numel()));
    std::vector<int32_t> ti_buf(static_cast<size_t>(tout[1].numel()));
    const int64_t out_shape[] = {2};
    TensorView ov(out_shape, tout[0].dtype, tv_buf.data());
    TensorView oi(out_shape, tout[1].dtype, ti_buf.data());
    TensorView outs2[] = {ov, oi};
    const TensorView ins2[] = {tv};
    NNOPS_EXPECT_EQ(topk->compute(outs2, ins2), Status::unsupported);
}
#endif
