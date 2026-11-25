/// @file grid_sample.cu
/// @brief CUDA GridSample — 2D bilinear / nearest and 3D trilinear / nearest on
///        NHWC / NDHWC.
///
/// The grid gives one source position per output pixel and the channels share
/// it, so a block owns one output pixel and its threads span C: the coordinate,
/// the padding wrap and the interpolation weights are computed redundantly (a
/// handful of floats) and each thread interpolates its own channels, which
/// keeps both the taps and the store coalesced. The CPU route splits the same
/// way, with its SIMD lane across C.
///
/// The rules are the CPU's: align_corners' edge vs center mapping, `fmod`-based
/// reflection, border clamping, bilinear/trilinear bounds decided per pixel, and
/// zeros padding that either zeroes (or, under add_to, leaves) the whole pixel.
///
/// f32 and f16 images; the grid is dense f32. add_to adds the sample to the
/// existing output.

#include "cuda_common.cuh"
#include "nnops/ops/grid_sample.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

__device__ inline float grid_to_pixel(float coord, int64_t size, bool align)
{
    if (align) {
        return (coord + 1.0f) * static_cast<float>(size - 1) * 0.5f;
    }
    return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
}

__device__ inline float reflect_coord(float x, int64_t size)
{
    if (size <= 1) {
        return 0.0f;
    }
    const float max_val = static_cast<float>(size - 1);
    float r = fmodf(fabsf(x), 2.0f * max_val);
    if (r > max_val) {
        r = 2.0f * max_val - r;
    }
    return r;
}

__device__ inline float apply_padding(float pixel, int64_t size, int padding)
{
    switch (padding) {
    case 1: // Border
        return fmaxf(0.0f, fminf(pixel, static_cast<float>(size - 1)));
    case 2: // Reflection
        return reflect_coord(pixel, size);
    default:
        return pixel;
    }
}

__device__ inline bool in_bounds(float pixel, int64_t size)
{
    return pixel >= 0.0f && pixel <= static_cast<float>(size - 1);
}

__device__ inline int64_t clamp_idx(int64_t idx, int64_t size)
{
    return idx < 0 ? 0 : (idx >= size ? size - 1 : idx);
}

__device__ inline float ld(const float* p) { return *p; }
__device__ inline float ld(const __half* p) { return __half2float(*p); }

template <class T> __device__ inline void st_add(T* p, float v, bool add_to)
{
    s_store(p, add_to ? elem_to_float(*p) + v : v);
}


/// The channel-vector width a type reads at: one 16 B load.
template <class T> struct ChanVec { using V = pack8<T>; static constexpr int N = 8; };
template <> struct ChanVec<float> { using V = float4; static constexpr int N = 4; };

__device__ inline float vget(float4 v, int k) noexcept
{
    return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w));
}
template <class T> __device__ inline float vget(pack8<T> v, int k) noexcept
{
    return elem_to_float(v.data[k]);
}
template <class T, int N> __device__ inline void vstore_n(T* p, const float (&o)[N]) noexcept
{
    if constexpr (N == 4) {
        v4_store(p, make_float4(o[0], o[1], o[2], o[3]));
    } else {
        v4_store(p, pack8_set<T>(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]));
    }
}


/// Threads a pixel needs: one per channel vector, rounded to a warp. A block
/// per pixel with a fixed 256 threads left eight of them active on a 64-channel
/// f16 pixel; one warp (or a few) keeps every lane busy and lets more pixels
/// be resident.
inline int channel_threads(int64_t C, int64_t vec) noexcept
{
    const int64_t groups = (C + vec - 1) / vec;
    int64_t threads = ((groups + 31) / 32) * 32;
    threads = std::max<int64_t>(32, std::min<int64_t>(threads, kThreads));
    return static_cast<int>(threads);
}

template <class T>
__global__ void grid_sample_2d_kernel(const T* __restrict__ in, const float* __restrict__ grid,
                                      T* __restrict__ out, int64_t pixels, int64_t C, int64_t IH,
                                      int64_t IW, int64_t OH, int64_t OW, int64_t in_n,
                                      int64_t in_row, int64_t out_n, int64_t out_row,
                                      int64_t grid_n, int64_t grid_row, int64_t grid_pix,
                                      bool align, int padding, bool nearest, bool add_to)
{
    for (int64_t p = blockIdx.x; p < pixels; p += gridDim.x) {
        int64_t r = p;
        const int64_t ow = r % OW;
        r /= OW;
        const int64_t oh = r % OH;
        const int64_t n = r / OH;
        const float* g = grid + n * grid_n + oh * grid_row + ow * grid_pix;
        const T* nb = in + n * in_n;
        T* dst = out + n * out_n + oh * out_row + ow * C;

        float py = grid_to_pixel(g[0], IH, align);
        float pz = grid_to_pixel(g[1], IW, align);
        const bool zeros = padding == 0;

        if (nearest) {
            if (zeros && (!in_bounds(roundf(py), IH) || !in_bounds(roundf(pz), IW))) {
                for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                    st_add(dst + c, 0.0f, add_to);
                }
                continue;
            }
            py = apply_padding(py, IH, padding);
            pz = apply_padding(pz, IW, padding);
            const T* src = nb + int64_t(roundf(py)) * in_row + int64_t(roundf(pz)) * C;
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                st_add(dst + c, ld(src + c), add_to);
            }
            continue;
        }

        if (zeros && !(in_bounds(floorf(py), IH) && in_bounds(ceilf(py), IH) &&
                       in_bounds(floorf(pz), IW) && in_bounds(ceilf(pz), IW))) {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                st_add(dst + c, 0.0f, add_to);
            }
            continue;
        }
        py = apply_padding(py, IH, padding);
        pz = apply_padding(pz, IW, padding);

        const int64_t y0 = static_cast<int64_t>(floorf(py));
        const int64_t z0 = static_cast<int64_t>(floorf(pz));
        const int64_t y1 = clamp_idx(y0 + 1, IH);
        const int64_t z1 = clamp_idx(z0 + 1, IW);
        const float wy = py - static_cast<float>(y0);
        const float wz = pz - static_cast<float>(z0);
        const float wy0 = 1.0f - wy;
        const float wz0 = 1.0f - wz;

        const T* p00 = nb + y0 * in_row + z0 * C;
        const T* p01 = nb + y0 * in_row + z1 * C;
        const T* p10 = nb + y1 * in_row + z0 * C;
        const T* p11 = nb + y1 * in_row + z1 * C;
        // A channel vector per thread when the four taps and the store all sit
        // on the 16 B bound; the scalar walk covers the rest.
        constexpr int VEC = ChanVec<T>::N;
        using V = typename ChanVec<T>::V;
        const bool vec = aligned16(p00, p01, p10) && aligned16(p11, dst) && (C % VEC) == 0;
        if (vec) {
            for (int64_t c = int64_t(threadIdx.x) * VEC; c < C;
                 c += int64_t(blockDim.x) * VEC) {
                const V a0 = v4_load(p00 + c);
                const V a1 = v4_load(p01 + c);
                const V a2 = v4_load(p10 + c);
                const V a3 = v4_load(p11 + c);
                const V old = add_to ? v4_load(dst + c) : V{};
                float o[VEC];
#pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    const float r0 = wz0 * vget(a0, k) + wz * vget(a1, k);
                    const float r1 = wz0 * vget(a2, k) + wz * vget(a3, k);
                    o[k] = wy0 * r0 + wy * r1;
                    if (add_to) {
                        o[k] += vget(old, k);
                    }
                }
                vstore_n(dst + c, o);
            }
        } else {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                const float row0 = wz0 * ld(p00 + c) + wz * ld(p01 + c);
                const float row1 = wz0 * ld(p10 + c) + wz * ld(p11 + c);
                st_add(dst + c, wy0 * row0 + wy * row1, add_to);
            }
        }
    }
}

template <class T>
__global__ void grid_sample_3d_kernel(const T* __restrict__ in, const float* __restrict__ grid,
                                      T* __restrict__ out, int64_t pixels, int64_t C, int64_t ID,
                                      int64_t IH, int64_t IW, int64_t OD, int64_t OH, int64_t OW,
                                      int64_t in_n, int64_t in_d, int64_t in_row, int64_t out_n,
                                      int64_t out_d, int64_t out_row, int64_t grid_n,
                                      int64_t grid_row, int64_t grid_pix, bool align, int padding,
                                      bool nearest, bool add_to)
{
    for (int64_t p = blockIdx.x; p < pixels; p += gridDim.x) {
        int64_t r = p;
        const int64_t ow = r % OW;
        r /= OW;
        const int64_t oh = r % OH;
        r /= OH;
        const int64_t od = r % OD;
        const int64_t n = r / OD;
        const float* g = grid + n * grid_n + (od * OH + oh) * grid_row + ow * grid_pix;
        const T* nb = in + n * in_n;
        T* dst = out + n * out_n + od * out_d + oh * out_row + ow * C;

        float px = grid_to_pixel(g[0], ID, align);
        float py = grid_to_pixel(g[1], IH, align);
        float pz = grid_to_pixel(g[2], IW, align);
        const bool zeros = padding == 0;

        if (nearest) {
            if (zeros && (!in_bounds(roundf(px), ID) || !in_bounds(roundf(py), IH) ||
                          !in_bounds(roundf(pz), IW))) {
                for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                    st_add(dst + c, 0.0f, add_to);
                }
                continue;
            }
            px = apply_padding(px, ID, padding);
            py = apply_padding(py, IH, padding);
            pz = apply_padding(pz, IW, padding);
            const T* src = nb + int64_t(roundf(px)) * in_d + int64_t(roundf(py)) * in_row +
                           int64_t(roundf(pz)) * C;
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                st_add(dst + c, ld(src + c), add_to);
            }
            continue;
        }

        if (zeros && !(in_bounds(floorf(px), ID) && in_bounds(ceilf(px), ID) &&
                       in_bounds(floorf(py), IH) && in_bounds(ceilf(py), IH) &&
                       in_bounds(floorf(pz), IW) && in_bounds(ceilf(pz), IW))) {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                st_add(dst + c, 0.0f, add_to);
            }
            continue;
        }
        px = apply_padding(px, ID, padding);
        py = apply_padding(py, IH, padding);
        pz = apply_padding(pz, IW, padding);

        const int64_t x0 = static_cast<int64_t>(floorf(px));
        const int64_t y0 = static_cast<int64_t>(floorf(py));
        const int64_t z0 = static_cast<int64_t>(floorf(pz));
        const int64_t x1 = clamp_idx(x0 + 1, ID);
        const int64_t y1 = clamp_idx(y0 + 1, IH);
        const int64_t z1 = clamp_idx(z0 + 1, IW);
        const float wx = px - static_cast<float>(x0);
        const float wy = py - static_cast<float>(y0);
        const float wz = pz - static_cast<float>(z0);
        const float wx0 = 1.0f - wx;
        const float wy0 = 1.0f - wy;
        const float wz0 = 1.0f - wz;

        const T* b00 = nb + x0 * in_d + y0 * in_row;
        const T* b01 = nb + x0 * in_d + y1 * in_row;
        const T* b10 = nb + x1 * in_d + y0 * in_row;
        const T* b11 = nb + x1 * in_d + y1 * in_row;
        constexpr int VEC = ChanVec<T>::N;
        using V = typename ChanVec<T>::V;
        const T* t00 = b00 + z0 * C;
        const T* t01 = b01 + z0 * C;
        const T* t10 = b10 + z0 * C;
        const T* t11 = b11 + z0 * C;
        const bool vec = aligned16(t00, t01, t10) && aligned16(t11, dst) &&
                         (((reinterpret_cast<uintptr_t>(t00) - reinterpret_cast<uintptr_t>(b00)) &
                           15u) == 0) &&
                         (C % VEC) == 0;
        if (vec) {
            for (int64_t c = int64_t(threadIdx.x) * VEC; c < C;
                 c += int64_t(blockDim.x) * VEC) {
                const V v00a = v4_load(t00 + c);
                const V v00b = v4_load(t00 + z1 * C + c);
                const V v01a = v4_load(t01 + c);
                const V v01b = v4_load(t01 + z1 * C + c);
                const V v10a = v4_load(t10 + c);
                const V v10b = v4_load(t10 + z1 * C + c);
                const V v11a = v4_load(t11 + c);
                const V v11b = v4_load(t11 + z1 * C + c);
                const V old = add_to ? v4_load(dst + c) : V{};
                float o[VEC];
#pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    const float r00 = wz0 * vget(v00a, k) + wz * vget(v00b, k);
                    const float r01 = wz0 * vget(v01a, k) + wz * vget(v01b, k);
                    const float r10 = wz0 * vget(v10a, k) + wz * vget(v10b, k);
                    const float r11 = wz0 * vget(v11a, k) + wz * vget(v11b, k);
                    const float s0 = wy0 * r00 + wy * r01;
                    const float s1 = wy0 * r10 + wy * r11;
                    o[k] = wx0 * s0 + wx * s1;
                    if (add_to) {
                        o[k] += vget(old, k);
                    }
                }
                vstore_n(dst + c, o);
            }
        } else {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                const float r00 = wz0 * ld(b00 + z0 * C + c) + wz * ld(b00 + z1 * C + c);
                const float r01 = wz0 * ld(b01 + z0 * C + c) + wz * ld(b01 + z1 * C + c);
                const float r10 = wz0 * ld(b10 + z0 * C + c) + wz * ld(b10 + z1 * C + c);
                const float r11 = wz0 * ld(b11 + z0 * C + c) + wz * ld(b11 + z1 * C + c);
                const float s0 = wy0 * r00 + wy * r01;
                const float s1 = wy0 * r10 + wy * r11;
                st_add(dst + c, wx0 * s0 + wx * s1, add_to);
            }
        }
    }
}

inline int padding_code(GridSamplePaddingMode m) noexcept
{
    switch (m) {
    case GridSamplePaddingMode::Border:
        return 1;
    case GridSamplePaddingMode::Reflection:
        return 2;
    default:
        return 0;
    }
}

template <class T>
Status grid_sample_entry(const GridSampleAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& input = inputs[0];
    const TensorView& grid = inputs[1];
    const int64_t irank = input.rank();
    if (irank != 4 && irank != 5) {
        return Status::invalid_argument;
    }
    const int64_t N = input.shape(0);
    const int64_t C = input.shape(irank - 1);
    const int64_t srank = irank - 2;
    const int64_t ID = srank == 3 ? input.shape(1) : 1;
    const int64_t IH = input.shape(srank - 1);
    const int64_t IW = input.shape(srank);
    const int64_t OD = srank == 3 ? output.shape(1) : 1;
    const int64_t OH = output.shape(srank - 1);
    const int64_t OW = output.shape(srank);
    if (N == 0 || C == 0 || OD == 0 || OH == 0 || OW == 0) {
        return Status::ok;
    }

    // A channel-last row is one W line, so the H step is the row pitch and the
    // W step is one channel vector (stride 1, indexed by C). The D step is the
    // (H · W · C) stride, which the view already reports.
    const int64_t in_n = input.stride_elems(0);
    const int64_t in_d = srank == 3 ? input.stride_elems(1) : 0;
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_d = srank == 3 ? output.stride_elems(1) : 0;
    const int64_t out_row = output.row_stride_elems();
    const int64_t grid_n = grid.stride_elems(0);
    const int64_t grid_row = grid.stride_elems(grid.rank() - 3);
    const int64_t grid_pix = grid.stride_elems(grid.rank() - 2);

    const int64_t pixels = srank == 3 ? N * OD * OH * OW : N * OH * OW;
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(pixels, int64_t{1} << 20));
    const int pad = padding_code(attrs.padding_mode);
    const bool nearest = attrs.mode == GridSampleMode::Nearest;

    const int64_t vec = sizeof(T) == 2 ? 8 : 4;
    const int threads = channel_threads(C, vec);
    if (srank == 3) {
        grid_sample_3d_kernel<T><<<blocks, threads, 0, stream>>>(
            input.ptr<T>(), grid.ptr<float>(), output.ptr<T>(), pixels, C, ID, IH, IW, OD, OH,
            OW, in_n, in_d, in_row, out_n, out_d, out_row, grid_n, grid_row, grid_pix,
            attrs.align_corners, pad, nearest, attrs.add_to);
    } else {
        grid_sample_2d_kernel<T><<<blocks, threads, 0, stream>>>(
            input.ptr<T>(), grid.ptr<float>(), output.ptr<T>(), pixels, C, IH, IW, OH, OW, in_n,
            in_row, out_n, out_row, grid_n, grid_row, grid_pix, attrs.align_corners, pad,
            nearest, attrs.add_to);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status grid_sample_cuda(const GridSampleAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return grid_sample_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return grid_sample_entry<__half>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
