/// @file non_max_suppression.cu
/// @brief CUDA NonMaxSuppression — greedy IoU suppression, one block per
///        (batch, class).
///
/// The classes are independent, so one block owns a (batch, class) and the
/// blocks run at once; the CPU walks them in sequence only because a later
/// class appends after an earlier one. The output order is restored by a
/// device-side prefix sum over the per-class counts, so no host round trip is
/// needed after the scalars are read.
///
/// Inside a class the candidate order is (score desc, box index asc) — the same
/// deterministic tie order the CPU route uses — and it is produced by a
/// block-wide repeated maximum over a bitmask of already-taken candidates, so a
/// score list of N anchors is never materialized or sorted. The first `top_k`
/// candidates (or all, when it is 0) are enumerated, and each is suppressed
/// when its best IoU against the kept boxes exceeds the threshold; the kept
/// boxes are decoded once, into a global plane that stays in L2.
///
/// The boxes are decoded with the CPU's rules: ONNX's corner form is
/// [y1, x1, y2, x2], the center form is [xc, yc, w, h], degenerate corners are
/// normalized, and IoU is 0 when either area is 0. Output rows are
/// (batch, class, box), truncated at the caller's capacity.

#include "cuda_common.cuh"
#include "nnops/ops/non_max_suppression.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

__device__ inline bool box_iou_gt(float ax1, float ay1, float ax2, float ay2, float aarea,
                                  float bx1, float by1, float bx2, float by2, float barea,
                                  float thr)
{
    if (aarea <= 0.0f || barea <= 0.0f) {
        return false;
    }
    const float iw = fminf(ax2, bx2) - fmaxf(ax1, bx1);
    if (iw <= 0.0f) {
        return false;
    }
    const float ih = fminf(ay2, by2) - fmaxf(ay1, by1);
    if (ih <= 0.0f) {
        return false;
    }
    const float inter = iw * ih;
    const float uni = aarea + barea - inter;
    const float iou = uni > 0.0f ? inter / uni : 0.0f;
    return iou > thr;
}

/// Decode one box with the CPU's normalization.
__device__ inline void decode_box(const float* p, bool center, float& x1, float& y1, float& x2,
                                  float& y2, float& area)
{
    if (center) {
        x1 = p[0] - p[2] * 0.5f;
        x2 = p[0] + p[2] * 0.5f;
        y1 = p[1] - p[3] * 0.5f;
        y2 = p[1] + p[3] * 0.5f;
    } else {
        y1 = p[0];
        x1 = p[1];
        y2 = p[2];
        x2 = p[3];
    }
    if (x2 < x1) {
        const float t = x1;
        x1 = x2;
        x2 = t;
    }
    if (y2 < y1) {
        const float t = y1;
        y1 = y2;
        y2 = t;
    }
    area = (x2 - x1) * (y2 - y1);
}

__device__ inline float load_box_elem(const float* p) { return *p; }
__device__ inline float load_box_elem(const __half* p) { return __half2float(*p); }

template <class T>
__global__ void nms_select_kernel(const T* __restrict__ boxes, const T* __restrict__ scores,
                                  uint64_t* __restrict__ used_bits, int64_t* __restrict__ counts,
                                  int64_t* __restrict__ kept, float* __restrict__ kept_boxes,
                                  int64_t classes, int64_t N, int64_t C, int64_t bs0, int64_t bs1,
                                  int64_t ss0, int64_t ss1, int64_t words, int64_t pc,
                                  int64_t max_out, float iou_thr, float score_thr, bool center)
{
    // One warp per (batch, class): the selection is a serial chain of
    // warp-wide maxima, so a warp keeps it free of block barriers and lets a
    // block host eight independent classes.
    const int lane = int(threadIdx.x) & 31;
    const int wid = int(threadIdx.x) >> 5;
    const int warps = int(blockDim.x) >> 5;
    const int64_t total_warps = int64_t(gridDim.x) * warps;

    for (int64_t bc = int64_t(blockIdx.x) * warps + wid; bc < classes; bc += total_warps) {
        const int64_t bi = bc / C;
        const int64_t ci = bc - bi * C;
        const T* bbase = boxes + bi * bs0;
        const T* sbase = scores + bi * ss0 + ci * ss1;
        uint64_t* used = used_bits + bc * words;
        int64_t* kdst = kept + bc * pc;
        float* kboxes = kept_boxes + bc * pc * 5;

        int64_t nkept = 0;
        int64_t enumerated = 0;
        while (enumerated < pc && nkept < max_out) {
            // Warp-wide best unused candidate, tie-broken by the lower index.
            float best = -INFINITY;
            int64_t best_i = -1;
            for (int64_t i = lane; i < N; i += 32) {
                if (used[i >> 6] & (uint64_t{1} << (i & 63))) {
                    continue;
                }
                const float s = load_box_elem(sbase + i);
                if (s > best || (s == best && (best_i < 0 || i < best_i))) {
                    best = s;
                    best_i = i;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float os = __shfl_down_sync(0xffffffffu, best, off);
                const long long oi =
                    __shfl_down_sync(0xffffffffu, static_cast<long long>(best_i), off);
                if (os > best ||
                    (os == best && oi >= 0 && (best_i < 0 || oi < best_i))) {
                    best = os;
                    best_i = static_cast<int64_t>(oi);
                }
            }
            best = __shfl_sync(0xffffffffu, best, 0);
            best_i = static_cast<int64_t>(
                __shfl_sync(0xffffffffu, static_cast<long long>(best_i), 0));
            if (best_i < 0 || !(best > score_thr)) {
                break;
            }
            if (lane == 0) {
                used[best_i >> 6] |= uint64_t{1} << (best_i & 63);
            }
            __syncwarp();
            ++enumerated;

            float x1, y1, x2, y2, area;
            if constexpr (std::is_same_v<T, float>) {
                decode_box(bbase + best_i * bs1, center, x1, y1, x2, y2, area);
            } else {
                float vals[4];
                const T* raw = bbase + best_i * bs1;
                for (int k = 0; k < 4; ++k) {
                    vals[k] = elem_to_float(raw[k]);
                }
                decode_box(vals, center, x1, y1, x2, y2, area);
            }

            bool suppressed = false;
            if (nkept > 0) {
                float worst = 0.0f;
                for (int64_t k = lane; k < nkept; k += 32) {
                    const float* q = kboxes + k * 5;
                    if (box_iou_gt(x1, y1, x2, y2, area, q[0], q[1], q[2], q[3], q[4],
                                   iou_thr)) {
                        worst = 1.0f;
                    }
                }
                worst = warp_reduce_max(worst);
                worst = __shfl_sync(0xffffffffu, worst, 0);
                suppressed = worst > 0.0f;
            }

            if (!suppressed && nkept < max_out) {
                if (lane == 0) {
                    kdst[nkept] = best_i;
                    float* q = kboxes + nkept * 5;
                    q[0] = x1;
                    q[1] = y1;
                    q[2] = x2;
                    q[3] = y2;
                    q[4] = area;
                }
                ++nkept; // every lane took the same branch
            }
        }

        if (lane == 0) {
            counts[bc] = nkept;
        }
        __syncwarp();
    }
}

/// One thread turns the per-class counts into output offsets and the final
/// (capacity-clamped) row count.
__global__ void nms_prefix_kernel(const int64_t* __restrict__ counts, int64_t* __restrict__ offsets,
                                  int64_t* __restrict__ num_selected, int64_t classes,
                                  int64_t cap)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int64_t running = 0;
    for (int64_t i = 0; i < classes; ++i) {
        offsets[i] = running;
        running += counts[i];
    }
    num_selected[0] = running < cap ? running : cap;
}

__global__ void nms_scatter_kernel(const int64_t* __restrict__ counts,
                                   const int64_t* __restrict__ offsets,
                                   const int64_t* __restrict__ kept, int64_t* __restrict__ out,
                                   int64_t classes, int64_t C, int64_t pc, int64_t cap,
                                   int64_t out_ld)
{
    const int64_t bc = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bc >= classes) {
        return;
    }
    const int64_t bi = bc / C;
    const int64_t ci = bc - bi * C;
    const int64_t n = counts[bc];
    const int64_t base = offsets[bc];
    for (int64_t j = 0; j < n; ++j) {
        const int64_t pos = base + j;
        if (pos >= cap) {
            break;
        }
        out[pos * out_ld + 0] = bi;
        out[pos * out_ld + 1] = ci;
        out[pos * out_ld + 2] = kept[bc * pc + j];
    }
}

/// A stream-ordered scratch block carved for the four regions.
struct Scratch {
    void* ptr = nullptr;
    bool async = false;
    bool alloc(int64_t bytes, cudaStream_t stream)
    {
        if (bytes <= 0) {
            return true;
        }
        if (cudaMallocAsync(&ptr, size_t(bytes), stream) == cudaSuccess) {
            async = true;
            return true;
        }
        cudaGetLastError();
        if (cudaMalloc(&ptr, size_t(bytes)) == cudaSuccess) {
            return true;
        }
        cudaGetLastError();
        return false;
    }
    void release(cudaStream_t stream)
    {
        if (ptr == nullptr) {
            return;
        }
        if (async && cudaFreeAsync(ptr, stream) == cudaSuccess) {
            ptr = nullptr;
            return;
        }
        cudaGetLastError();
        cudaFree(ptr);
        ptr = nullptr;
    }
};

template <class T>
Status nms_entry(const NonMaxSuppressionAttributes& attrs, TensorView& indices_out,
                 TensorView& count_out, std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& boxes = inputs[0];
    const TensorView& scores = inputs[1];
    const int64_t B = boxes.shape(0);
    const int64_t N = boxes.shape(1);
    const int64_t C = scores.shape(1);
    const int64_t cap = indices_out.shape(0);

    // The three optionals are device scalars; the host needs them to size the
    // scratch and the per-class capacity, so they are read back once.
    int64_t max_out = 0;
    float iou_thr = 0.0f;
    float score_thr = 0.0f;
    if (inputs.size() >= 3 && !inputs[2].is_empty()) {
        const TensorView& v = inputs[2];
        if (v.data_type() == DataType::s32) {
            int32_t m = 0;
            cudaMemcpyAsync(&m, v.ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            max_out = m;
        } else {
            cudaMemcpyAsync(&max_out, v.ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost,
                            stream);
            cudaStreamSynchronize(stream);
        }
        if (max_out < 0) {
            max_out = 0;
        }
    }
    if (inputs.size() >= 4 && !inputs[3].is_empty()) {
        cudaMemcpyAsync(&iou_thr, inputs[3].ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
    }
    if (inputs.size() >= 5 && !inputs[4].is_empty()) {
        cudaMemcpyAsync(&score_thr, inputs[4].ptr<float>(), sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    if (max_out == 0 || cap == 0 || N == 0 || B == 0 || C == 0) {
        const int64_t zero = 0;
        cudaMemcpyAsync(count_out.ptr<int64_t>(), &zero, sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream);
        return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }

    const int64_t pc = std::min(max_out, attrs.top_k > 0 ? attrs.top_k : N);
    const int64_t classes = B * C;
    const int64_t words = (N + 63) / 64;

    const int64_t mask_bytes = classes * words * 8;
    const int64_t counts_bytes = classes * 8;
    const int64_t offsets_bytes = classes * 8;
    const int64_t kept_bytes = classes * pc * 8;
    const int64_t kbox_bytes = classes * pc * 5 * 4;
    const int64_t total = mask_bytes + counts_bytes + offsets_bytes + kept_bytes + kbox_bytes;

    Scratch scratch;
    if (!scratch.alloc(total, stream)) {
        return Status::out_of_memory;
    }
    char* base = static_cast<char*>(scratch.ptr);
    uint64_t* used = reinterpret_cast<uint64_t*>(base);
    int64_t* counts = reinterpret_cast<int64_t*>(base + mask_bytes);
    int64_t* offsets = reinterpret_cast<int64_t*>(base + mask_bytes + counts_bytes);
    int64_t* kept = reinterpret_cast<int64_t*>(base + mask_bytes + counts_bytes + offsets_bytes);
    float* kboxes = reinterpret_cast<float*>(base + mask_bytes + counts_bytes + offsets_bytes +
                                             kept_bytes);

    cudaMemsetAsync(used, 0, size_t(mask_bytes), stream);

    const bool center = attrs.center_point_box == 1;
    constexpr int kWarpsPerBlock = 8;
    const int64_t groups = (classes + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(groups, 65535));
    nms_select_kernel<T><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        boxes.ptr<T>(), scores.ptr<T>(), used, counts, kept, kboxes, classes, N, C,
        boxes.stride_elems(0), boxes.stride_elems(1), scores.stride_elems(0),
        scores.stride_elems(1), words, pc, std::min(max_out, pc), iou_thr, score_thr, center);

    nms_prefix_kernel<<<1, 1, 0, stream>>>(counts, offsets, count_out.ptr<int64_t>(), classes,
                                           cap);
    nms_scatter_kernel<<<static_cast<unsigned>((classes + kThreads - 1) / kThreads), kThreads, 0,
                         stream>>>(counts, offsets, kept, indices_out.ptr<int64_t>(), classes, C,
                                   pc, cap, indices_out.row_stride_elems());

    const Status st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    scratch.release(stream);
    return st;
}

} // anonymous namespace

Status nms_cuda(const NonMaxSuppressionAttributes& attrs, TensorView& indices,
                TensorView& num_selected, std::span<const TensorView> inputs,
                const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return nms_entry<float>(attrs, indices, num_selected, inputs, stream);
    case DataType::f16:
        return nms_entry<__half>(attrs, indices, num_selected, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
