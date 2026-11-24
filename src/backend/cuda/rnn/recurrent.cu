/// @file recurrent.cu
/// @brief CUDA recurrent layer — RNN, GRU and LSTM behind one timeline.
///
/// The CPU route's decomposition survives the move to the device, because it is
/// the decomposition the problem has:
///
///   1. The input projection X·W^T has all its rows up front, so a chunk of
///      timesteps runs as ONE dense GEMM (M = chunk·B) on the tensor-core
///      engine. W is [G·H, I] row-major, which is already that kernel's k-major
///      B pose, so the projection reads the caller's weight without a pack.
///   2. H_{t-1}·R^T is serial in t by definition: per timestep it is one more
///      dense GEMM of M = B rows, again reading R in place.
///   3. The gate arithmetic is one elementwise pass over [B, gates·H]: it reads
///      the projection row and the recurrent product, applies clip and each
///      gate's activation, updates the hidden (and cell) state and writes Y.
///
/// The state buffers are dense [B, H], double-buffered and swapped once per
/// timestep, which is also how the CPU driver walks the timeline. A GRU without
/// linear_before_reset needs two passes per step — h~'s operand r ⊙ H_{t-1} does
/// not exist until r does — so its first phase leaves z and r ⊙ H_{t-1} behind
/// for the second, exactly as on the CPU. Every launch is on the caller's
/// stream, so the phase order is the stream's.
///
/// f32, f16 and bf16. An s8 weight (the CPU's W8A8 route) and a prepacked
/// PackedRecurrent weight are both Status::unsupported here: the first would
/// need the int8 MMA's per-row scaling and zero-point terms, and the second is
/// a CPU-packer blob this driver has no reader for.

#include "backend/cuda/rnn/recurrent.h"

#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/matmul/gemm_common.cuh"
#include "backend/cuda/matmul/matmul_op.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>

namespace nnops::backend::cuda {

namespace {

// =========================================================================
//  The elementwise epilogue
// =========================================================================

/// Which gates one pass covers. GruAll is the linear_before_reset GRU, whose
/// three recurrent products all take H_{t-1}; without lbr the h gate's operand
/// is r ⊙ H_{t-1}, so the step splits into GruZr then GruH.
enum StepKind {
    kRnn = 0,
    kLstm = 1,
    kGruAll = 2,
    kGruZr = 3,
    kGruH = 4,
};

/// One phase's view of a timestep. All pointers are device-resident; `proj`
/// points at this timestep's first row of the chunk buffer, already advanced by
/// the caller.
template <class T> struct StepParams {
    int64_t batch = 0;
    int64_t hidden = 0;
    int64_t gate_row = 0; ///< G·H: the row stride of `proj` and `rec`
    int64_t col_base = 0; ///< this phase's first column in the gate row
    int64_t gates = 0;    ///< gates this phase covers

    const T* proj = nullptr; ///< X_t·W^T, [batch][gate_row]
    const T* rec = nullptr;  ///< H_{t-1}·R^T (or its phase's slice)
    const T* wb = nullptr;   ///< the direction's Wb, or null
    const T* rb = nullptr;   ///< the direction's Rb, or null

    T* h_prev = nullptr; ///< [batch][hidden]
    T* h_next = nullptr;
    T* c_prev = nullptr; ///< LSTM only
    T* c_next = nullptr;
    T* zb = nullptr;  ///< GRU (lbr = 0): z, the h~ phase's blend
    T* aux = nullptr; ///< GRU (lbr = 0): r ⊙ H_{t-1}, the h~ phase's operand

    T* y = nullptr; ///< Y's row for batch 0 at (t, d), or null
    int64_t y_ld = 0;
    const int32_t* seq_len = nullptr;

    int64_t t = 0;
    float clip = 0.0f;
    int act_f = 0, act_g = 1, act_h = 1;
    float alpha_f = 0.0f, alpha_g = 0.0f, alpha_h = 0.0f;
    int input_forget = 0;
    int lbr = 0;
};

/// The five ONNX activations with a kernel; the operator rejects the rest.
/// Sigmoid, LeakyRelu and Elu follow the CPU SIMD kernels' forms exactly so the
/// two routes agree on the special points (LeakyRelu at 0, Elu at 0).
__device__ inline float rnn_act(float x, int a, float alpha) noexcept
{
    switch (a) {
    case 0: // Sigmoid
        return 1.0f / (1.0f + expf(-x));
    case 1: // Tanh
        return tanhf(x);
    case 2: // Relu
        return fmaxf(x, 0.0f);
    case 3: // LeakyRelu
        return fmaxf(x, 0.0f) + alpha * fminf(x, 0.0f);
    case 4: // Elu
        return fmaxf(x, 0.0f) + alpha * (expf(fminf(x, 0.0f)) - 1.0f);
    default:
        return x;
    }
}

/// The pre-activation bound ONNX calls `clip`; 0 disables it.
__device__ inline float rnn_clip(float x, float clip) noexcept
{
    return clip > 0.0f ? fminf(fmaxf(x, -clip), clip) : x;
}

template <class T> __device__ inline float ld(const T* p, int64_t i) noexcept
{
    return s_load(p + i);
}

template <class T, StepKind Kind> __global__ void recurrent_step_kernel(StepParams<T> s)
{
    const int64_t total = s.batch * s.hidden;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / s.hidden;
    const int64_t h = idx - b * s.hidden;
    T* const hn_row = s.h_next + b * s.hidden;
    T* const y_row = s.y != nullptr ? s.y + b * s.y_ld : nullptr;

    if (s.seq_len != nullptr && s.t >= s.seq_len[b]) {
        // Past its length a row holds its state and writes Y = 0; the state
        // buffers are swapped every step, so the copy is what preserves it.
        hn_row[h] = s.h_prev[b * s.hidden + h];
        if (s.c_next != nullptr) {
            s.c_next[b * s.hidden + h] = s.c_prev[b * s.hidden + h];
        }
        if (y_row != nullptr) {
            s_store(y_row + h, 0.0f);
        }
        return;
    }

    const int64_t row = b * s.gate_row;
    const int64_t c = s.col_base + h;
    const T* const proj = s.proj + row;
    const T* const rec = s.rec + row;

    // Gate g's pre-activation: projection + recurrent product + Wb. Rb is read
    // separately because a linear_before_reset GRU adds it to the recurrent
    // product instead.
    const auto pre = [&](int64_t g) {
        const int64_t at = c + g * s.hidden;
        return ld(proj, at) + ld(rec, g * s.hidden + h) +
               (s.wb != nullptr ? s_load(s.wb + at) : 0.0f);
    };
    const auto r_bias = [&](int64_t g) {
        return s.rb != nullptr ? s_load(s.rb + c + g * s.hidden) : 0.0f;
    };

    const float h_prev = ld(s.h_prev + b * s.hidden, h);

    if constexpr (Kind == kRnn) {
        const float v =
            rnn_act(rnn_clip(pre(0) + r_bias(0), s.clip), s.act_g, s.alpha_g);
        s_store(hn_row + h, v);
        if (y_row != nullptr) {
            s_store(y_row + h, v);
        }

    } else if constexpr (Kind == kLstm) {
        // ONNX gate order is i, o, f, c.
        const float vi = rnn_act(rnn_clip(pre(0) + r_bias(0), s.clip), s.act_f, s.alpha_f);
        const float vo = rnn_act(rnn_clip(pre(1) + r_bias(1), s.clip), s.act_f, s.alpha_f);
        const float vf =
            s.input_forget ? 1.0f - vi
                           : rnn_act(rnn_clip(pre(2) + r_bias(2), s.clip), s.act_f, s.alpha_f);
        const float vc = rnn_act(rnn_clip(pre(3) + r_bias(3), s.clip), s.act_g, s.alpha_g);
        const float c_prev = ld(s.c_prev + b * s.hidden, h);
        const float cell = vf * c_prev + vi * vc;
        s_store(s.c_next + b * s.hidden + h, cell);
        const float v = vo * rnn_act(cell, s.act_h, s.alpha_h);
        s_store(hn_row + h, v);
        if (y_row != nullptr) {
            s_store(y_row + h, v);
        }

    } else if constexpr (Kind == kGruAll) {
        // ONNX gate order is z, r, h. The h gate's recurrent term is the one r
        // scales, and linear_before_reset is why Rbh is added here rather than
        // folded into the projection.
        const float z =
            rnn_act(rnn_clip(pre(0) + r_bias(0), s.clip), s.act_f, s.alpha_f);
        const float r =
            rnn_act(rnn_clip(pre(1) + r_bias(1), s.clip), s.act_f, s.alpha_f);
        const int64_t at = c + 2 * s.hidden;
        const float ph = ld(proj, at) + (s.wb != nullptr ? s_load(s.wb + at) : 0.0f);
        const float rh = ld(rec, 2 * s.hidden + h) +
                         (s.rb != nullptr ? s_load(s.rb + at) : 0.0f);
        const float vh = rnn_act(rnn_clip(ph + r * rh, s.clip), s.act_g, s.alpha_g);
        const float v = z * h_prev + (1.0f - z) * vh;
        s_store(hn_row + h, v);
        if (y_row != nullptr) {
            s_store(y_row + h, v);
        }

    } else if constexpr (Kind == kGruZr) {
        // z and r, and the h~ phase's operand r ⊙ H_{t-1}. No state or Y write:
        // the step is only half finished.
        const float z =
            rnn_act(rnn_clip(pre(0) + r_bias(0), s.clip), s.act_f, s.alpha_f);
        const float r =
            rnn_act(rnn_clip(pre(1) + r_bias(1), s.clip), s.act_f, s.alpha_f);
        s_store(s.zb + b * s.hidden + h, z);
        s_store(s.aux + b * s.hidden + h, r * h_prev);

    } else { // kGruH
        const float vh =
            rnn_act(rnn_clip(pre(0) + r_bias(0), s.clip), s.act_g, s.alpha_g);
        const float z = ld(s.zb + b * s.hidden, h);
        const float v = z * h_prev + (1.0f - z) * vh;
        s_store(hn_row + h, v);
        if (y_row != nullptr) {
            s_store(y_row + h, v);
        }
    }
}

// =========================================================================
//  State staging
// =========================================================================

/// h or c for one direction, widened into the dense [batch, hidden] state
/// buffer. A null source is all zeros, which is what ONNX's absent initial state
/// means.
template <class T>
__global__ void init_state_kernel(T* dst, const T* src, int64_t src_ld, int64_t hidden,
                                  int64_t total)
{
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    if (src == nullptr) {
        dst[i] = float_to_elem<T>(0.0f);
        return;
    }
    const int64_t b = i / hidden;
    dst[i] = src[b * src_ld + (i - b * hidden)];
}

/// The reverse: the state buffer out to the caller's [.., batch, hidden] row,
/// zeroing a length-0 sequence per ONNX's null_state_default_value.
template <class T>
__global__ void store_state_kernel(T* dst, int64_t dst_ld, const T* src, const int32_t* seq_len,
                                   int64_t hidden, int64_t total)
{
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int64_t b = i / hidden;
    if (seq_len != nullptr && seq_len[b] == 0) {
        s_store(dst + b * dst_ld + (i - b * hidden), 0.0f);
        return;
    }
    dst[b * dst_ld + (i - b * hidden)] = src[i];
}

// =========================================================================
//  Host side
// =========================================================================

constexpr int64_t kAlign = 256;
constexpr int64_t kProjectionBudget = int64_t{8} << 20;

inline int64_t align_up(int64_t bytes) noexcept { return (bytes + kAlign - 1) / kAlign * kAlign; }

/// The scalar launch geometry for the elementwise passes.
inline int launch_blocks(int64_t total) noexcept
{
    return static_cast<int>((total + kCudaBlockSize - 1) / kCudaBlockSize);
}

/// The narrow products' split count, 0 when the automatic ladder should decide.
///
/// The loop's own rule sizes a split by K and the L2 planes and never looks at
/// N. A step's N is the gate row — H, 2·H or 4·H — and at H ≤ 512 that is a
/// handful of 128-column blocks, so the machine runs a fraction full and the
/// step costs more than its bytes. The rule here is blocks first: a split per
/// 128-column block up to 64 blocks in f32 and 128 in f16 (the 2-byte loop's
/// staging is half the bytes, so it fills deeper), rounded to a power of two
/// and capped by the k tiles' depth (a split below 128 k has nothing to
/// stream). Measured on this host at M=8, K=1024, against the automatic route:
/// f32 N=256 2.6x, 512 2.1x, 768 1.8x, 1024 1.7x, 1536 1.35x, 2048 1.19x;
/// f16 N=256 2.3x, 512 2.0x, 768 1.9x, 1024 1.8x, 1536 1.3x, 2048 1.18x.
/// f16 keeps the tiled engine above N=2048, and for a K under 1024 there too:
/// its narrower rows win once the column grid is wide.
inline int narrow_split_hint(int64_t m, int64_t k, int64_t n, DataType dt) noexcept
{
    if (m < 1 || m > 16 || n < 1 || k < 256) {
        return 0;
    }
    int64_t target;
    if (dt == DataType::f32) {
        target = 8192;
    } else if (dt == DataType::f16) {
        if (n > 2048 || (n > 1536 && k < 1024)) {
            return 0;
        }
        target = 16384;
    } else {
        return 0;
    }
    int64_t splits = (target + n - 1) / n;
    int64_t pow2 = 1;
    while (pow2 < splits) {
        pow2 <<= 1;
    }
    splits = std::min<int64_t>(std::min<int64_t>(pow2, 16), std::max<int64_t>(k / 128, 1));
    return splits >= 2 ? static_cast<int>(splits) : 0;
}

/// The second stream a bidirectional layer runs its other timeline on. One per
/// host thread, so two threads never share an event; the stream is non-blocking
/// because a blocking one would implicitly order against the legacy default
/// stream, and the two are joined with explicit events instead.
struct AuxChannel {
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t done = nullptr;
};

inline AuxChannel& aux_channel()
{
    thread_local AuxChannel ch = [] {
        AuxChannel c;
        if (cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            c.stream = nullptr;
            return c;
        }
        if (cudaEventCreateWithFlags(&c.ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&c.done, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            c.stream = nullptr;
        }
        return c;
    }();
    return ch;
}

/// One phase of one timestep: the recurrent product on the dense engine, then
/// the fused gate pass.
template <class T>
Status run_phase(const cpu::RecurrentParams& p, StepKind kind, int64_t gates, int64_t gate_row,
                 int64_t col_base, const T* a, const T* r_dir, int64_t r_ld, T* rec, int64_t batch,
                 int64_t hidden, StepParams<T> s, cudaStream_t stream)
{
    GemmParams g;
    g.a = a;
    g.b = r_dir + (col_base * r_ld);
    g.c = rec;
    g.m = static_cast<int>(batch);
    g.n = static_cast<int>(gates * hidden);
    g.k = static_cast<int>(hidden);
    g.lda = hidden;
    g.ldb = r_ld;
    g.ldc = gate_row;
    g.alpha = 1.0f;
    g.beta = 0.0f;
    g.splits_hint = narrow_split_hint(batch, hidden, gates * hidden, p.dtype);

    const Status st = dense_gemm_status(g, p.dtype, /*transpose_a=*/false, /*transpose_b=*/true,
                                        stream);
    if (!is_ok(st)) {
        return st;
    }

    s.col_base = col_base;
    s.gates = gates;
    s.rec = rec;

    const int64_t total = batch * hidden;
    const int blocks = launch_blocks(total);
    switch (kind) {
    case kRnn:
        recurrent_step_kernel<T, kRnn><<<blocks, kCudaBlockSize, 0, stream>>>(s);
        break;
    case kLstm:
        recurrent_step_kernel<T, kLstm><<<blocks, kCudaBlockSize, 0, stream>>>(s);
        break;
    case kGruAll:
        recurrent_step_kernel<T, kGruAll><<<blocks, kCudaBlockSize, 0, stream>>>(s);
        break;
    case kGruZr:
        recurrent_step_kernel<T, kGruZr><<<blocks, kCudaBlockSize, 0, stream>>>(s);
        break;
    case kGruH:
        recurrent_step_kernel<T, kGruH><<<blocks, kCudaBlockSize, 0, stream>>>(s);
        break;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

/// The whole layer for one element type: resolve the views, carve one scratch
/// block, and walk every direction's timeline.
template <class T>
Status recurrent_impl(const cpu::RecurrentParams& p, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const int64_t B = p.batch;
    const int64_t H = p.hidden;
    const int64_t I = p.input;
    const int64_t D = p.directions;
    const int64_t Tn = p.seq;
    if (B <= 0 || H <= 0 || I <= 0 || D <= 0) {
        return Status::ok;
    }
    if (B > INT_MAX || H > INT_MAX || I > INT_MAX || Tn > INT_MAX) {
        return Status::unsupported;
    }

    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView& r = inputs[2];
    if (x.rank() < 3 || w.rank() < 3 || r.rank() < 3) {
        return Status::invalid_argument;
    }
    if (x.shape(0) != Tn || x.shape(1) != B || x.shape(2) != I) {
        return Status::invalid_argument;
    }

    const int64_t G = recurrent_gates(p.mode);
    const int64_t gate_row = G * H;
    // The GEMM front end is 32-bit in its shapes; a state or gate row that does
    // not fit is refused rather than truncated at the cast below.
    if (B * H > INT_MAX || B * gate_row > INT_MAX) {
        return Status::unsupported;
    }
    const bool lstm = p.mode == RecurrentMode::Lstm;
    const bool gru = p.mode == RecurrentMode::Gru;
    const bool lbr = gru && p.linear_before_reset;

    if (w.shape(0) != D || w.shape(1) != gate_row || w.shape(2) != I ||
        r.shape(0) != D || r.shape(1) != gate_row || r.shape(2) != H) {
        return Status::invalid_argument;
    }

    const TensorView* bias =
        (inputs.size() > 3 && !inputs[3].is_empty()) ? &inputs[3] : nullptr;
    const TensorView* seqv = (inputs.size() > 4 && !inputs[4].is_empty()) ? &inputs[4] : nullptr;
    const TensorView* init_h = (inputs.size() > 5 && !inputs[5].is_empty()) ? &inputs[5] : nullptr;
    const TensorView* init_c = (inputs.size() > 6 && !inputs[6].is_empty()) ? &inputs[6] : nullptr;

    if ((bias != nullptr && bias->data_type() != p.dtype) ||
        (seqv != nullptr && seqv->data_type() != DataType::s32) ||
        (init_h != nullptr && init_h->data_type() != p.dtype) ||
        (init_c != nullptr && init_c->data_type() != p.dtype)) {
        return Status::invalid_argument;
    }

    const T* const xp = x.ptr<T>();
    const T* const wp = w.ptr<T>();
    const T* const rp = r.ptr<T>();
    const int64_t x_ld = x.row_stride_elems();
    const int64_t w_ld = w.row_stride_elems();
    const int64_t r_ld = r.row_stride_elems();

    const T* const bias_base = bias != nullptr ? bias->ptr<T>() : nullptr;
    const int64_t bias_ld = bias != nullptr ? bias->row_stride_elems() : 0;
    const int32_t* const seq_len = seqv != nullptr ? seqv->ptr<int32_t>() : nullptr;
    const T* const init_h_ptr = init_h != nullptr ? init_h->ptr<T>() : nullptr;
    const int64_t init_h_ld = init_h != nullptr ? init_h->row_stride_elems() : 0;
    const T* const init_c_ptr = init_c != nullptr ? init_c->ptr<T>() : nullptr;
    const int64_t init_c_ld = init_c != nullptr ? init_c->row_stride_elems() : 0;

    T* const y_base = !outputs[0].is_empty() ? outputs[0].ptr<T>() : nullptr;
    const int64_t y_ld = !outputs[0].is_empty() ? outputs[0].row_stride_elems() : 0;
    T* const yh_base =
        outputs.size() > 1 && !outputs[1].is_empty() ? outputs[1].ptr<T>() : nullptr;
    const int64_t yh_ld =
        outputs.size() > 1 && !outputs[1].is_empty() ? outputs[1].row_stride_elems() : 0;
    T* const yc_base =
        outputs.size() > 2 && !outputs[2].is_empty() ? outputs[2].ptr<T>() : nullptr;
    const int64_t yc_ld =
        outputs.size() > 2 && !outputs[2].is_empty() ? outputs[2].row_stride_elems() : 0;

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const int64_t esz = static_cast<int64_t>(sizeof(T));

    const bool grn = gru && !lbr; // the two-phase GRU
    const bool concurrent =
        D == 2 && nnops::detail::cuda_rnn_serial.load(std::memory_order_relaxed) == 0;

    // The projection buffer follows the CPU's L3 budget: a chunk is however many
    // timesteps keep [chunk·B, G·H] under the cap. A bidirectional layer splits
    // the cap between its two running timelines.
    int64_t chunk = kProjectionBudget / std::max<int64_t>(B * gate_row * esz, 1);
    if (concurrent) {
        chunk /= 2;
    }
    chunk = std::max<int64_t>(1, std::min<int64_t>(chunk, std::max<int64_t>(Tn, 1)));

    /// One direction's live buffers. A direction is a self-contained timeline
    /// (its own weight rows, its own state and Y rows), so a bidirectional
    /// layer gives each a set and runs both at once.
    struct Bufs {
        T* proj = nullptr;
        T* rec = nullptr;
        T* h_a = nullptr;
        T* h_b = nullptr;
        T* c_a = nullptr; ///< LSTM only
        T* c_b = nullptr;
        T* zb = nullptr;  ///< GRU (lbr = 0) only
        T* aux = nullptr;
    };
    const int nsets = concurrent ? 2 : 1;

    // Every region starts on the same 256 B bound, so the size is the sum of the
    // aligned region sizes — exactly what the carve below consumes.
    int64_t bytes = 0;
    const auto region = [&](int64_t elems) { bytes = align_up(bytes) + elems * esz; };
    for (int set = 0; set < nsets; ++set) {
        region(chunk * B * gate_row);
        region(B * gate_row);
        region(B * H);
        region(B * H);
        if (lstm) {
            region(B * H);
            region(B * H);
        }
        if (grn) {
            region(B * H);
            region(B * H);
        }
    }

    detail::GemmWorkspace ws = detail::gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) {
        return Status::out_of_memory;
    }

    char* base = static_cast<char*>(ws.ptr);
    int64_t used = 0;
    const auto take = [&](int64_t elems) -> T* {
        used = align_up(used);
        T* q = reinterpret_cast<T*>(base + used);
        used += elems * esz;
        return q;
    };

    Bufs bufs[2];
    for (int set = 0; set < nsets; ++set) {
        Bufs& bu = bufs[set];
        bu.proj = take(chunk * B * gate_row);
        bu.rec = take(B * gate_row);
        bu.h_a = take(B * H);
        bu.h_b = take(B * H);
        bu.c_a = lstm ? take(B * H) : nullptr;
        bu.c_b = lstm ? take(B * H) : nullptr;
        bu.zb = grn ? take(B * H) : nullptr;
        bu.aux = grn ? take(B * H) : nullptr;
    }

    const int64_t state_total = B * H;
    const int state_blocks = launch_blocks(state_total);

    // One direction's whole timeline on `s`: initialise, project chunk by chunk,
    // walk the timesteps, store the final state. It touches only its own Bufs
    // and the d-th rows of every operand, so two of these can run concurrently.
    const auto run_direction = [&](int64_t d, const Bufs& bu, cudaStream_t s) -> Status {
        T* h_prev = bu.h_a;
        T* h_next = bu.h_b;
        T* c_prev = bu.c_a;
        T* c_next = bu.c_b;

        init_state_kernel<T><<<state_blocks, kCudaBlockSize, 0, s>>>(
            h_prev, init_h_ptr != nullptr ? init_h_ptr + d * B * init_h_ld : nullptr,
            init_h_ld, H, state_total);
        if (lstm) {
            init_state_kernel<T><<<state_blocks, kCudaBlockSize, 0, s>>>(
                c_prev, init_c_ptr != nullptr ? init_c_ptr + d * B * init_c_ld : nullptr,
                init_c_ld, H, state_total);
        }
        if (cudaGetLastError() != cudaSuccess) {
            return Status::internal;
        }

        const bool fwd = (D > 1) ? (d == 0) : (p.direction != RnnDirection::Reverse);
        const int64_t nchunks = (Tn + chunk - 1) / chunk;
        const T* const w_dir = wp + d * gate_row * w_ld;
        const T* const r_dir = rp + d * gate_row * r_ld;
        const T* const wb_dir = bias_base != nullptr ? bias_base + d * bias_ld : nullptr;
        const T* const rb_dir = wb_dir != nullptr ? wb_dir + gate_row : nullptr;

        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t t0 =
                fwd ? ch * chunk : std::max<int64_t>(0, Tn - (ch + 1) * chunk);
            const int64_t nrows = std::min<int64_t>(chunk, Tn - t0);
            if (nrows <= 0) {
                continue;
            }

            // The projection over the chunk: [nrows·B, I] × [G·H, I]^T.
            {
                GemmParams g;
                g.a = xp + (t0 * B) * x_ld;
                g.b = w_dir;
                g.c = bu.proj;
                g.m = static_cast<int>(nrows * B);
                g.n = static_cast<int>(gate_row);
                g.k = static_cast<int>(I);
                g.lda = x_ld;
                g.ldb = w_ld;
                g.ldc = gate_row;
                g.alpha = 1.0f;
                g.beta = 0.0f;
                // At T = 1 the projection is a short-M product too, and the same
                // block-fill rule applies (a decode step's projection is half of
                // the call).
                g.splits_hint = narrow_split_hint(nrows * B, I, gate_row, p.dtype);
                const Status st = dense_gemm_status(g, p.dtype, /*transpose_a=*/false,
                                                    /*transpose_b=*/true, s);
                if (!is_ok(st)) {
                    return st;
                }
            }

            const int64_t t_step = fwd ? 1 : -1;
            const int64_t t_first = fwd ? t0 : t0 + nrows - 1;

            for (int64_t i = 0; i < nrows; ++i) {
                const int64_t t = t_first + i * t_step;
                const T* const ip = bu.proj + ((t - t0) * B) * gate_row;
                T* const y_row =
                    y_base != nullptr ? y_base + ((t * D + d) * B) * y_ld : nullptr;

                const int64_t nphases = grn ? 2 : 1;
                for (int64_t ph = 0; ph < nphases; ++ph) {
                    StepKind kind;
                    int64_t gates;
                    int64_t col_base;
                    if (p.mode == RecurrentMode::Rnn) {
                        kind = kRnn;
                        gates = 1;
                        col_base = 0;
                    } else if (lstm) {
                        kind = kLstm;
                        gates = 4;
                        col_base = 0;
                    } else if (lbr) {
                        kind = kGruAll;
                        gates = 3;
                        col_base = 0;
                    } else {
                        kind = ph == 0 ? kGruZr : kGruH;
                        gates = ph == 0 ? 2 : 1;
                        col_base = ph == 0 ? 0 : 2 * H;
                    }

                    StepParams<T> step;
                    step.batch = B;
                    step.hidden = H;
                    step.gate_row = gate_row;
                    step.col_base = col_base;
                    step.gates = gates;
                    step.proj = ip;
                    step.rec = bu.rec;
                    step.wb = wb_dir;
                    step.rb = rb_dir;
                    step.h_prev = h_prev;
                    step.h_next = h_next;
                    step.c_prev = c_prev;
                    step.c_next = c_next;
                    step.zb = bu.zb;
                    step.aux = bu.aux;
                    step.y = y_row;
                    step.y_ld = y_ld;
                    step.seq_len = seq_len;
                    step.t = t;
                    step.clip = p.clip;
                    step.act_f = static_cast<int>(p.act_f);
                    step.act_g = static_cast<int>(p.act_g);
                    step.act_h = static_cast<int>(p.act_h);
                    step.alpha_f = p.alpha_f;
                    step.alpha_g = p.alpha_g;
                    step.alpha_h = p.alpha_h;
                    step.input_forget = p.input_forget ? 1 : 0;
                    step.lbr = lbr ? 1 : 0;

                    const T* const a = ph == 1 ? bu.aux : h_prev;
                    const Status st = run_phase<T>(p, kind, gates, gate_row, col_base, a,
                                                   r_dir, r_ld, bu.rec, B, H, step, s);
                    if (!is_ok(st)) {
                        return st;
                    }
                }

                std::swap(h_prev, h_next);
                if (lstm) {
                    std::swap(c_prev, c_next);
                }
            }
        }

        if (yh_base != nullptr) {
            store_state_kernel<T><<<state_blocks, kCudaBlockSize, 0, s>>>(
                yh_base + (d * B) * yh_ld, yh_ld, h_prev, seq_len, H, state_total);
        }
        if (lstm && yc_base != nullptr) {
            store_state_kernel<T><<<state_blocks, kCudaBlockSize, 0, s>>>(
                yc_base + (d * B) * yc_ld, yc_ld, c_prev, seq_len, H, state_total);
        }
        return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    };

    Status status = Status::ok;
    if (concurrent) {
        AuxChannel& aux = aux_channel();
        if (aux.stream != nullptr) {
            // Join the caller's stream first (the inputs may be produced by the
            // work ahead of this call), then give direction 1 its own stream and
            // make the caller's stream wait for it before returning.
            cudaEventRecord(aux.ready, stream);
            cudaStreamWaitEvent(aux.stream, aux.ready, 0);
            status = run_direction(0, bufs[0], stream);
            const Status st1 = run_direction(1, bufs[1], aux.stream);
            cudaEventRecord(aux.done, aux.stream);
            cudaStreamWaitEvent(stream, aux.done, 0);
            if (is_ok(status) && !is_ok(st1)) {
                status = st1;
            }
        } else {
            for (int64_t d = 0; d < D && is_ok(status); ++d) {
                status = run_direction(d, bufs[d], stream);
            }
        }
    } else {
        for (int64_t d = 0; d < D && is_ok(status); ++d) {
            status = run_direction(d, bufs[0], stream);
        }
    }

    detail::gemm_workspace_free(ws, stream);
    return status;
}

} // namespace

Status recurrent_cuda(const cpu::RecurrentParams& params, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    // The compute dtype and the weight dtype coincide on every route this
    // driver carries; a mismatch is the quantized form (or a bad call).
    if (params.weight_dtype != params.dtype) {
        return Status::unsupported;
    }
    if (inputs.size() < 3 || outputs.empty()) {
        return Status::invalid_argument;
    }
    if (is_packed_recurrent(inputs[1].layout())) {
        return Status::unsupported;
    }
    if (inputs[0].data_type() != params.dtype || inputs[1].data_type() != params.dtype ||
        inputs[2].data_type() != params.dtype) {
        return Status::unsupported;
    }

    switch (params.dtype) {
    case DataType::f32:
        return recurrent_impl<float>(params, outputs, inputs, ctx);
    case DataType::f16:
        return recurrent_impl<__half>(params, outputs, inputs, ctx);
    case DataType::bf16:
        return recurrent_impl<__nv_bfloat16>(params, outputs, inputs, ctx);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
