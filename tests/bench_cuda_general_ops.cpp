/// @file bench_cuda_general_ops.cpp
/// @brief CUDA general-operator throughput — the non-conv/attention/recurrent
///        family.
///
/// Each row is one operator call timed with CUDA events (median of the samples
/// after a warm-up), reported with the bytes it touches, so an elementwise row
/// reads as a bandwidth and a sampling row as a rate. The shapes are the ones a
/// graph actually feeds these ops: a token batch for cast/clamp/where, a ViT
/// patch merge for the pixel shuffle, an embedding table, a QDQ pair, a
/// detection crop, a prefill RoPE row and a detector's anchor set for NMS.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/nnops.hpp"
#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/cast.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/pool1d.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/scatter.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/unary.hpp"

#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

inline std::vector<int64_t> sh(std::initializer_list<int64_t> l)
{
    return std::vector<int64_t>(l);
}

/// A dense host tensor of an integer dtype (the shared maker fills floats).
HostTensor int_t(const std::vector<int64_t>& s, DataType dt, uint64_t seed, int64_t lo,
                 int64_t hi)
{
    HostTensor t;
    const TensorView probe(s, dt, nullptr);
    t.bytes.assign(probe.storage_bytes(), std::byte{0});
    XorShift128 rng(seed);
    const int64_t n = probe.numel();
    if (dt == DataType::s32) {
        auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<int32_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else if (dt == DataType::s8) {
        auto* p = reinterpret_cast<int8_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<int8_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else if (dt == DataType::u8) {
        auto* p = reinterpret_cast<uint8_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<uint8_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else {
        auto* p = reinterpret_cast<int64_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1));
        }
    }
    t.view = TensorView(s, dt, t.bytes.data());
    return t;
}

/// One call timed with events; `launch` submits it on the default stream.
template <class F> double time_ms(F&& launch) { return cuda_median_ms(launch, 3, 7, 10); }

void row(const char* tag, const char* shape, double ms, double bytes)
{
    std::printf("  %-22s %-38s %9.4f ms %9.1f GB/s\n", tag, shape, ms,
                ms > 0.0 ? bytes / (ms * 1e6) : 0.0);
    std::fflush(stdout);
}

/// Run a single-output op over uploaded inputs and time it.
template <class OpT, class AttrT>
void bench_op(const char* tag, const char* shape, const AttrT& attrs,
              std::initializer_list<const HostTensor*> ins, double bytes)
{
    std::vector<TensorDesc> descs;
    std::vector<TensorView> views;
    for (const HostTensor* t : ins) {
        descs.push_back(t->view.desc());
        views.push_back(t->view);
    }
    auto op = OpT::create(attrs, Backend::CUDA);
    const auto od = op->getOutputTensorDesc(descs);
    HostTensor zero = make_host_zeroed(od[0].dims, od[0].dtype, false, od[0].layout);
    std::vector<DeviceTensor> dins;
    for (const HostTensor* t : ins) {
        dins.push_back(to_device(*t));
    }
    std::vector<TensorView> dv;
    for (DeviceTensor& d : dins) {
        dv.push_back(d.view);
    }
    DeviceTensor dout = alloc_like(zero.view, zero);
    TensorView out = dout.view;
    auto launch = [&] { cuda_launch(*op, out, dv); };
    row(tag, shape, time_ms(launch), bytes);
}

/// Multi-output variant (TopK, Split): one zeroed buffer per descriptor.
template <class OpT, class AttrT>
void bench_op_multi(const char* tag, const char* shape, const AttrT& attrs,
                    std::initializer_list<const HostTensor*> ins, double bytes)
{
    std::vector<TensorDesc> descs;
    for (const HostTensor* t : ins) {
        descs.push_back(t->view.desc());
    }
    auto op = OpT::create(attrs, Backend::CUDA);
    const auto od = op->getOutputTensorDesc(descs);

    std::vector<HostTensor> zeros;
    zeros.reserve(od.size());
    for (const auto& d : od) {
        zeros.push_back(make_host_zeroed(d.dims, d.dtype, false, d.layout));
    }
    std::vector<DeviceTensor> douts;
    douts.reserve(zeros.size());
    for (HostTensor& z : zeros) {
        douts.push_back(alloc_like(z.view, z));
    }
    std::vector<DeviceTensor> dins;
    dins.reserve(ins.size());
    for (const HostTensor* t : ins) {
        dins.push_back(to_device(*t));
    }
    std::vector<TensorView> views;
    for (DeviceTensor& d : dins) {
        views.push_back(d.view);
    }
    std::vector<TensorView> ov;
    for (DeviceTensor& d : douts) {
        ov.push_back(d.view);
    }
    auto launch = [&] {
        const Status st = op->compute(ov, views, cuda_context(), nullptr);
        if (st != Status::ok) {
            throw std::runtime_error(std::string(tag) + ": " + to_string(st));
        }
    };
    row(tag, shape, time_ms(launch), bytes);
}

} // anonymous namespace

NNOPS_BENCH(cuda_general_elementwise) {
    print_cuda_header("CUDA cast / clamp / compare / where");

    const int64_t n = 4 << 20;
    HostTensor x = make_host_random(sh({n}), DataType::f32, false, 11, -1.0f, 1.0f);
    {
        CastAttributes to_f16;
        to_f16.to = DataType::f16;
        bench_op<Cast, CastAttributes>("cast f32->f16", "4M", to_f16, {&x}, n * 6.0);
        CastAttributes to_s8;
        to_s8.to = DataType::s8;
        bench_op<Cast, CastAttributes>("cast f32->s8", "4M", to_s8, {&x}, n * 5.0);
    }
    {
        ClampAttributes a;
        a.min_val = -0.5f;
        a.max_val = 0.5f;
        bench_op<Clamp, ClampAttributes>("clamp f32", "4M", a, {&x}, n * 8.0);
    }
    {
        HostTensor cond = int_t(sh({n}), DataType::u8, 21, 0, 1);
        HostTensor y = make_host_random(sh({n}), DataType::f32, false, 23, -1.0f, 1.0f);
        bench_op<Where, WhereAttributes>("where f32", "4M", WhereAttributes{}, {&cond, &x, &y},
                                         n * 13.0);
        CompareAttributes cmp;
        cmp.type = CompareType::Greater;
        bench_op<Compare, CompareAttributes>("compare f32", "4M", cmp, {&x, &y}, n * 9.0);


    }
}

NNOPS_BENCH(cuda_general_movement) {
    print_cuda_header("CUDA flatten / pixel shuffle / embedding");

    HostTensor a = make_host_random(sh({4, 64, 64, 96}), DataType::f32, false, 31);
    bench_op<Flatten, FlattenAttributes>("flatten f32", "4x64x64x96", FlattenAttributes{}, {&a},
                                         a.bytes.size() * 2.0);

    HostTensor c = make_host_random(sh({1, 32, 32, 384}), DataType::f32, false, 37);
    c.view = TensorView(c.view.shape_span(), DataType::f32, c.bytes.data(), TensorLayout::NHWC);
    PixelShuffleAttributes ps;
    ps.block_size = 2;
    bench_op<PixelShuffle, PixelShuffleAttributes>("pixel_shuffle f32", "1x32x32x384", ps, {&c},
                                                   c.bytes.size() * 2.0);

    HostTensor w = make_host_random(sh({32000, 1024}), DataType::f16, false, 41, -0.1f, 0.1f);
    HostTensor idx = int_t(sh({1024}), DataType::s32, 43, 0, 31999);
    bench_op<Embed, EmbedAttributes>("embed f16", "1024x1024", EmbedAttributes{}, {&w, &idx},
                                     1024.0 * 1024 * 2 * 2);
}

NNOPS_BENCH(cuda_general_quant) {
    print_cuda_header("CUDA Quantize / Dequantize (per-token)");

    const int64_t rows = 1024, cols = 4096;
    HostTensor x = make_host_random(sh({rows, cols}), DataType::f16, false, 47, -1.0f, 1.0f);
    HostTensor scale = make_host_random(sh({rows}), DataType::f32, false, 53, 0.01f, 0.05f);
    HostTensor zp = int_t(sh({rows}), DataType::s8, 59, -5, 5);
    QuantLinearAttributes qa;
    qa.axis = -1;
    qa.output_dtype = DataType::s8;
    bench_op<QuantizeLinear, QuantLinearAttributes>("quantize f16->s8", "1024x4096", qa,
                                                   {&x, &scale, &zp}, rows * cols * 6.0);

    HostTensor q = int_t(sh({rows, cols}), DataType::s8, 61, -128, 127);
    QuantLinearAttributes da;
    da.axis = -1;
    da.output_dtype = DataType::f16;
    bench_op<DequantizeLinear, QuantLinearAttributes>("dequantize s8->f16", "1024x4096", da,
                                                      {&q, &scale, &zp}, rows * cols * 6.0);
}

NNOPS_BENCH(cuda_general_spatial) {
    print_cuda_header("CUDA grid_sample / roi_align / pool1d");

    {
        HostTensor x = make_host_random(sh({1, 128, 128, 64}), DataType::f16, false, 67, -1.0f, 1.0f);
        x.view = TensorView(x.view.shape_span(), DataType::f16, x.bytes.data(),
                            TensorLayout::NHWC);
        HostTensor grid = make_host_random(sh({1, 128, 128, 2}), DataType::f32, false, 71, -1.0f,
                                           1.0f);
        bench_op<GridSample, GridSampleAttributes>("grid_sample f16", "128x128x64", {},
                                                   {&x, &grid}, x.bytes.size() * 2.0);
    }
    {
        HostTensor x = make_host_random(sh({1, 64, 64, 256}), DataType::f32, false, 73, -1.0f, 1.0f);
        x.view = TensorView(x.view.shape_span(), DataType::f32, x.bytes.data(),
                            TensorLayout::NHWC);
        HostTensor rois = make_host_random(sh({256, 4}), DataType::f32, false, 79, 0.0f, 60.0f);
        RoiAlignAttributes ra;
        ra.output_height = 7;
        ra.output_width = 7;
        ra.sampling_ratio = 2;
        bench_op<RoiAlign, RoiAlignAttributes>("roi_align f32", "256 rois 7x7", ra, {&x, &rois},
                                               x.bytes.size() * 2.0);
    }
    {
        HostTensor x = make_host_random(sh({8, 4096, 512}), DataType::f32, false, 83, -1.0f, 1.0f);
        x.view = TensorView(x.view.shape_span(), DataType::f32, x.bytes.data(),
                            TensorLayout::NHWC);
        Pool1DAttributes pa;
        pa.type = Pool1DType::Average;
        pa.kernel_shape = 3;
        pa.stride = 2;
        pa.padding = 1;
        bench_op<Pool1D, Pool1DAttributes>("pool1d f32", "8x4096x512", pa, {&x},
                                           x.bytes.size() * 2.0);
    }
}

NNOPS_BENCH(cuda_general_rope_nms) {
    print_cuda_header("CUDA RoPE / NonMaxSuppression");

    {
        // [B, S, heads, D] — the last axis is the head width.
        HostTensor x = make_host_random(sh({1, 2048, 32, 128}), DataType::f16, false, 89, -1.0f, 1.0f);
        RoPEAttributes ra;
        bench_op<RoPE, RoPEAttributes>("rope f16", "S=2048 h=32 D=128", ra, {&x},
                                       x.bytes.size() * 2.0);
    }
    {
        const int64_t B = 8, N = 2000, C = 80;
        HostTensor boxes = make_host_random(sh({B, N, 4}), DataType::f32, false, 97, 0.0f, 1.0f);
        HostTensor scores = make_host_random(sh({B, C, N}), DataType::f32, false, 101, -0.5f, 0.5f);
        HostTensor max_out = int_t(sh({1}), DataType::s32, 103, 200, 200);
        HostTensor iou = make_host_random(sh({1}), DataType::f32, false, 107, 0.4f, 0.6f);
        NonMaxSuppressionAttributes na;
        na.top_k = 1000;

        std::vector<TensorDesc> descs = {boxes.view.desc(), scores.view.desc(),
                                         max_out.view.desc(), iou.view.desc()};
        auto op = NonMaxSuppression::create(na, Backend::CUDA);
        const auto od = op->getOutputTensorDesc(descs);
        HostTensor zi = make_host_zeroed(od[0].dims, od[0].dtype, false);
        HostTensor zn = make_host_zeroed(od[1].dims, od[1].dtype, false);
        DeviceTensor db = to_device(boxes);
        DeviceTensor ds = to_device(scores);
        DeviceTensor dm = to_device(max_out);
        DeviceTensor di = to_device(iou);
        DeviceTensor do1 = alloc_like(zi.view, zi);
        DeviceTensor do2 = alloc_like(zn.view, zn);
        std::vector<TensorView> dv = {db.view, ds.view, dm.view, di.view};
        std::vector<TensorView> douts = {do1.view, do2.view};
        auto launch = [&] {
            const Status st = op->compute(douts, dv, cuda_context(), nullptr);
            if (st != Status::ok) {
                throw std::runtime_error(std::string("nms: ") + to_string(st));
            }
        };
        row("nms f32", "8x80x2000 topk=1000", time_ms(launch),
            boxes.bytes.size() + scores.bytes.size());
    }
}

NNOPS_BENCH(cuda_general_reduce_select) {
    print_cuda_header("CUDA reduce / argmax / topk / cumsum / softmax");

    {
        // A transformer block's activation: reduce the channel axis of a token
        // batch, and the same shape reduced on the other axis (the strided,
        // offset-table route).
        HostTensor a = make_host_random(sh({4096, 4096}), DataType::f32, false, 111);
        ReduceAttributes ra;
        ra.axes = {1};
        bench_op<Reduce, ReduceAttributes>("reduce sum ax1 f32", "4096x4096", ra, {&a},
                                           a.bytes.size() * 1.0);

        HostTensor b = make_host_random(sh({4096, 4096}), DataType::f32, false, 113);
        ReduceAttributes ra0;
        ra0.axes = {0};
        bench_op<Reduce, ReduceAttributes>("reduce sum ax0 f32", "4096x4096", ra0, {&b},
                                           b.bytes.size() * 1.0);

        // Many tiny rows: the per-row block cost, not bandwidth, is the metric.
        HostTensor c = make_host_random(sh({262144, 16}), DataType::f16, false, 115);
        ReduceAttributes rsmall;
        rsmall.axes = {1};
        bench_op<Reduce, ReduceAttributes>("reduce sum ax1 f16", "262144x16", rsmall, {&c},
                                           c.bytes.size() * 1.0);
    }
    {
        HostTensor a = make_host_random(sh({4096, 1000}), DataType::f32, false, 117);
        ArgMinMaxAttributes aa;
        aa.axis = 1;
        bench_op<ArgMax, ArgMinMaxAttributes>("argmax ax1 f32", "4096x1000", aa, {&a},
                                              a.bytes.size() + 4096 * 8);
    }
    {
        // Decode-time sampling: one row of a 151936-entry vocabulary (the chunk
        // route), and a batch of them.
        HostTensor a = make_host_random(sh({1, 151936}), DataType::f32, false, 119);
        TopKAttributes ta;
        ta.k = 50;
        bench_op_multi<TopK, TopKAttributes>("topk k=50 f32", "1x151936", ta, {&a},
                                             a.bytes.size() * 2.0);
        TopKAttributes t1;
        t1.k = 1;
        bench_op_multi<TopK, TopKAttributes>("topk k=1 f32", "1x151936", t1, {&a},
                                             a.bytes.size() * 2.0);

        HostTensor batch = make_host_random(sh({64, 151936}), DataType::f32, false, 122);
        bench_op_multi<TopK, TopKAttributes>("topk k=1 f32", "64x151936", t1, {&batch},
                                             batch.bytes.size() * 2.0);

        // A batched row that repeats per-token work: the register-list route.
        HostTensor b = make_host_random(sh({4096, 32000}), DataType::f32, false, 121);
        TopKAttributes tk1;
        tk1.k = 1;
        bench_op_multi<TopK, TopKAttributes>("topk k=1 f32", "4096x32000", tk1, {&b},
                                             b.bytes.size() * 2.0);
        TopKAttributes tb;
        tb.k = 10;
        bench_op_multi<TopK, TopKAttributes>("topk k=10 f32", "4096x32000", tb, {&b},
                                             b.bytes.size() * 2.0 + 4096.0 * 10 * 12);
    }
    {
        HostTensor a = make_host_random(sh({1024, 4096}), DataType::f32, false, 123);
        CumSumAttributes ca;
        ca.axis = 1;
        bench_op<CumSum, CumSumAttributes>("cumsum ax1 f32", "1024x4096", ca, {&a},
                                           a.bytes.size() * 2.0);
        CumSumAttributes ca0;
        ca0.axis = 0;
        bench_op<CumSum, CumSumAttributes>("cumsum ax0 f32", "1024x4096", ca0, {&a},
                                           a.bytes.size() * 2.0);
    }
    {
        // Softmax on the non-tail axis is the offset-table route; the tail is
        // the resident-row fast path (bench_cuda_elementwise covers it).
        HostTensor a = make_host_random(sh({64, 4096, 64}), DataType::f32, false, 127,
                                        -4.0f, 4.0f, TensorLayout::NHWC);
        SoftmaxAttributes s0;
        s0.axis = 1;
        bench_op<Softmax, SoftmaxAttributes>("softmax NHWC ax1 f32", "64x4096x64", s0, {&a},
                                             a.bytes.size() * 3.0);
    }
}

NNOPS_BENCH(cuda_general_index_move) {
    print_cuda_header("CUDA gather / scatter / pad / tile / concat / split / slice");

    {
        HostTensor w = make_host_random(sh({32000, 1024}), DataType::f16, false, 131);
        HostTensor idx = int_t(sh({2048}), DataType::s32, 133, 0, 31999);
        bench_op<Gather, GatherAttributes>("gather rows f16", "2048 of 32000x1024",
                                           GatherAttributes{}, {&w, &idx},
                                           2048.0 * 1024 * 2 * 2);

        HostTensor d = make_host_random(sh({512, 4096}), DataType::f32, false, 137);
        HostTensor gi = int_t(sh({512, 512}), DataType::s64, 139, 0, 4095);
        GatherElementsAttributes ge;
        ge.axis = 1;
        bench_op<GatherElements, GatherElementsAttributes>("gather_elements ax1", "512x512 from 512x4096",
                                                           ge, {&d, &gi},
                                                           512.0 * 512 * (8 + 4 + 4));
    }
    {
        HostTensor d = make_host_random(sh({512, 4096}), DataType::f32, false, 141);
        HostTensor si = int_t(sh({512, 512}), DataType::s64, 143, 0, 4095);
        HostTensor up = make_host_random(sh({512, 512}), DataType::f32, false, 147);
        ScatterElementsAttributes se;
        se.axis = 1;
        bench_op<ScatterElements, ScatterElementsAttributes>("scatter_elements ax1", "512x512",
                                                             se, {&d, &si, &up}, 512.0 * 512 * 16);
    }
    {
        HostTensor a = make_host_random(sh({32, 64, 64, 96}), DataType::f32, false, 149);
        PadAttributes pa;
        pa.pads = {0, 1, 1, 0, 0, 1, 1, 0};
        pa.mode = PadAttributes::Mode::Edge;
        bench_op<Pad, PadAttributes>("pad edge f32", "32x64x64x96 +1", pa, {&a},
                                     a.bytes.size() * 2.0);

        HostTensor t = make_host_random(sh({1024, 1024}), DataType::f32, false, 151);
        TileAttributes ta;
        ta.repeats = {1, 2};
        bench_op<Tile, TileAttributes>("tile f32", "1024x1024 x2", ta, {&t}, t.bytes.size() * 3.0);

        HostTensor s = make_host_random(sh({2048, 4096}), DataType::f32, false, 153);
        SliceAttributes sa;
        sa.starts = {0, 0};
        sa.ends = {2048, 2048};
        sa.axes = {0, 1};
        bench_op<Slice, SliceAttributes>("slice f32", "2048x4096 -> 2048x2048", sa, {&s},
                                         s.bytes.size() * 1.5);
    }
    {
        HostTensor a = make_host_random(sh({1024, 1024}), DataType::f32, false, 157);
        HostTensor b = make_host_random(sh({1024, 1024}), DataType::f32, false, 159);
        HostTensor c = make_host_random(sh({1024, 1024}), DataType::f32, false, 161);
        HostTensor d = make_host_random(sh({1024, 1024}), DataType::f32, false, 163);
        bench_op<Concat, ConcatAttributes>("concat f32", "4 x 1024x1024 ax0", ConcatAttributes{},
                                           {&a, &b, &c, &d}, a.bytes.size() * 8.0);

        HostTensor s = make_host_random(sh({4096, 4096}), DataType::f32, false, 167);
        SplitAttributes sp;
        sp.axis = 1;
        sp.split = {1024, 1024, 1024, 1024};
        bench_op_multi<Split, SplitAttributes>("split f32", "4096x4096 into 4", sp, {&s},
                                               s.bytes.size() * 2.0);

        HostTensor p = make_host_random(sh({2048, 2048}), DataType::f32, false, 173);
        PermuteAttributes pattr;
        pattr.perm = {1, 0};
        bench_op<Permute, PermuteAttributes>("permute f32", "2048x2048 T", pattr, {&p},
                                             p.bytes.size() * 2.0);

        HostTensor u = make_host_random(sh({4 << 20}), DataType::f32, false, 179, 0.1f, 2.0f);
        UnaryAttributes ua;
        ua.type = UnaryType::Exp;
        bench_op<Unary, UnaryAttributes>("unary exp f32", "4M", ua, {&u}, u.bytes.size() * 2.0);
        ua.type = UnaryType::Sqrt;
        bench_op<Unary, UnaryAttributes>("unary sqrt f32", "4M", ua, {&u}, u.bytes.size() * 2.0);
    }
}

#endif  // NNOPS_HAS_CUDA
