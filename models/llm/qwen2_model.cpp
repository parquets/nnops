/// @file qwen2_model.cpp
/// @brief Fake-weight Qwen2 forward — see qwen2_model.hpp for the graph.
///
/// The weight builder emits llama.cpp Q4_1 blocks straight from a per-block
/// random draw, so no f32 copy of a weight matrix ever exists — the embedding
/// table included, whose lookup decodes a row at a time. Only the norms and the
/// q/k/v biases are dense, in the activation dtype. A tied lm_head is the
/// embedding table itself, which is exactly what the checkpoint tie means.
///
/// The KV cache is dense and per layer: [H_kv, capacity, D] s8 beside
/// [H_kv, capacity] f32 scales. A step writes its new rows and then hands the
/// whole planes to GroupQueryAttention with kv_seqlens = length, so the
/// operator's causal row range is the only thing that sees the valid prefix.

#include "models/llm/qwen2_model.hpp"

#include "nnops/detail/half.hpp"
#include "nnops/detail/q4.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/rope.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>

#ifdef NNOPS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace nnops::llm {

using backend::cpu::float_to_half;
using backend::cpu::half;
using backend::cpu::half_to_float;

namespace {

// ---- device memory --------------------------------------------------------

bool dev_alloc(void*& p, size_t bytes)
{
#ifdef NNOPS_HAS_CUDA
    return cudaMalloc(&p, bytes > 0 ? bytes : 1) == cudaSuccess;
#else
    (void)p;
    (void)bytes;
    return false;
#endif
}

void dev_free(void* p)
{
#ifdef NNOPS_HAS_CUDA
    if (p != nullptr) {
        cudaFree(p);
    }
#else
    (void)p;
#endif
}

bool dev_copy(void* dst, const void* src, size_t bytes, bool to_device)
{
#ifdef NNOPS_HAS_CUDA
    if (bytes == 0) {
        return true;
    }
    return (to_device ? cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) :
                        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)) == cudaSuccess;
#else
    (void)dst;
    (void)src;
    (void)bytes;
    (void)to_device;
    return false;
#endif
}

/// Stream the running forward is using, so a phase timer can wait for its
/// launches before reading the clock. Null on the CPU.
void* g_profile_stream = nullptr;

// ---- deterministic random draws -------------------------------------------

struct Rng {
    uint64_t state;

    explicit Rng(uint64_t seed) : state(seed != 0 ? seed : 0x9E3779B97F4A7C15ull) {}

    uint64_t next() noexcept
    {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    float uniform(float lo, float hi) noexcept
    {
        const uint64_t u = next() >> 40; // 24 random bits
        return lo + (hi - lo) * (static_cast<float>(u) * (1.0f / 16777216.0f));
    }
};

// ---- tensor helpers -------------------------------------------------------

TensorView make_view(std::initializer_list<int64_t> dims, DataType dt, void* data)
{
    return TensorView(std::span<const int64_t>(dims.begin(), dims.size()), dt, data);
}

float elem_f32(DataType dt, const void* p, int64_t i) noexcept
{
    if (dt == DataType::f32) {
        return static_cast<const float*>(p)[i];
    }
    return half_to_float(static_cast<const half*>(p)[i]);
}

void store_f32(DataType dt, void* p, int64_t i, float v) noexcept
{
    if (dt == DataType::f32) {
        static_cast<float*>(p)[i] = v;
    } else {
        static_cast<half*>(p)[i] = float_to_half(v);
    }
}

// ---- fake tensors ---------------------------------------------------------

void fill_uniform(Buffer& b, float lo, float hi, Rng& rng)
{
    const int64_t n = b.numel();
    for (int64_t i = 0; i < n; ++i) {
        store_f32(b.dtype, b.bytes.data(), i, rng.uniform(lo, hi));
    }
}

Q4Matrix make_q4(int64_t rows, int64_t cols, float lo, float hi, Rng& rng)
{
    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.blocks = cols / q4_1_block_elems;
    m.bytes.assign(static_cast<size_t>(rows * m.blocks * q4_1_block_bytes), 0);
    float values[q4_1_block_elems];
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < m.blocks; ++b) {
            for (float& v : values) {
                v = rng.uniform(lo, hi);
            }
            uint8_t* block =
                m.bytes.data() + static_cast<size_t>((r * m.blocks + b) * q4_1_block_bytes);
            quantize_q4(values, q4_1_block_elems, block, DataType::q4_1);
        }
    }
    return m;
}

bool config_ok(const Qwen2Config& c)
{
    const int64_t block = q4_1_block_elems;
    return c.vocab_size > 0 && c.hidden_size > 0 && c.head_dim > 0 && c.head_dim % 2 == 0 &&
           c.num_attention_heads > 0 && c.num_key_value_heads > 0 &&
           c.num_attention_heads % c.num_key_value_heads == 0 &&
           c.num_hidden_layers > 0 && c.intermediate_size > 0 &&
           c.hidden_size % block == 0 && c.q_dim() % block == 0 &&
           c.kv_dim() % block == 0 && c.intermediate_size % block == 0;
}

/// Gather q4_1 embedding rows and decode them into the activation dtype.
/// Embed has no q4_1 form (its weight is f32/f16/s8/u8), so the lookup decodes
/// one 32-value block at a time — the same bytes the lm_head GEMM reads when
/// the table is tied.
void embed_rows(const Q4Matrix& table, std::span<const int64_t> tokens, void* dst, DataType dt)
{
    float values[q4_1_block_elems];
    const int64_t blocks = table.blocks;
    const int64_t cols = table.cols;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const uint8_t* row = table.bytes.data() +
                             static_cast<size_t>(tokens[i]) * blocks * q4_1_block_bytes;
        for (int64_t b = 0; b < blocks; ++b) {
            q4_unpack_block(row + static_cast<size_t>(b) * q4_1_block_bytes, values,
                            DataType::q4_1);
            for (int64_t j = 0; j < q4_1_block_elems; ++j) {
                store_f32(dt, dst, static_cast<int64_t>(i) * cols + b * q4_1_block_elems + j,
                          values[j]);
            }
        }
    }
}

// ---- phase timing ---------------------------------------------------------

/// Wall time around one operator call. Constructed on every call; the clock
/// read is skipped when profiling is off.
struct PhaseTimer {
    double& slot;
    bool on;
    std::chrono::steady_clock::time_point t0;

    PhaseTimer(double& s, bool o) : slot(s), on(o)
    {
        if (on) {
            t0 = std::chrono::steady_clock::now();
        }
    }

    ~PhaseTimer()
    {
        if (!on) {
            return;
        }
#ifdef NNOPS_HAS_CUDA
        // The launches are asynchronous; without this the clock would measure
        // the enqueue, not the kernel.
        if (g_profile_stream != nullptr) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(g_profile_stream));
        }
#endif
        slot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
    }
};

// ---- q8 cache rows --------------------------------------------------------

inline float to_float(float v) { return v; }
inline float to_float(half v) { return half_to_float(v); }

template <typename T>
void quantize_row(const T* src, int64_t n, int8_t* dst, float& scale)
{
    float amax = 0.0f;
    for (int64_t d = 0; d < n; ++d) {
        amax = std::max(amax, std::fabs(to_float(src[d])));
    }
    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    scale = s;
    for (int64_t d = 0; d < n; ++d) {
        const float q = std::round(to_float(src[d]) / s);
        dst[d] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
}

struct CacheLayer {
    std::vector<int8_t> k; ///< [H_kv, capacity, D] host planes (the staging)
    std::vector<int8_t> v;
    std::vector<float> k_scale; ///< [H_kv, capacity]
    std::vector<float> v_scale;
    DeviceBlock k_dev, v_dev;    ///< device planes the operator reads
    DeviceBlock ks_dev, vs_dev;
};

template <typename T>
void append_cache(const Qwen2Config& cfg, CacheLayer& c, const TensorView& k,
                  const TensorView& v, int64_t capacity, int64_t start, int64_t S)
{
    const int64_t D = cfg.head_dim;
    const int64_t kvd = cfg.kv_dim();
    const T* kp = k.ptr<T>(); // [1, H_kv, S, D]
    const T* vp = v.ptr<T>(); // [S, H_kv*D]
    for (int64_t h = 0; h < cfg.num_key_value_heads; ++h) {
        for (int64_t t = 0; t < S; ++t) {
            const int64_t row = h * capacity + start + t;
            quantize_row(kp + (h * S + t) * D, D, c.k.data() + row * D, c.k_scale[row]);
            quantize_row(vp + t * kvd + h * D, D, c.v.data() + row * D, c.v_scale[row]);
        }
    }
}

// ---- scratch --------------------------------------------------------------

struct Scratch {
    Buffer xn, q_lin, k_lin, v_lin;
    Buffer q_bhsd, k_bhsd, q_rope, k_rope;
    Buffer attn_bhsd, attn_merged;
    Buffer o, h;
    Buffer gate, up, act, ffn;
    Buffer x_a, x_b;
    Buffer last_norm, logits;

    void alloc(const Qwen2Config& cfg, DataType dt, int64_t S, bool device)
    {
        const int64_t H = cfg.num_attention_heads;
        const int64_t Hkv = cfg.num_key_value_heads;
        const int64_t D = cfg.head_dim;
        xn.alloc({S, cfg.hidden_size}, dt, device);
        q_lin.alloc({S, cfg.q_dim()}, dt, device);
        k_lin.alloc({S, cfg.kv_dim()}, dt, device);
        v_lin.alloc({S, cfg.kv_dim()}, dt, device);
        q_bhsd.alloc({1, H, S, D}, dt, device);
        k_bhsd.alloc({1, Hkv, S, D}, dt, device);
        q_rope.alloc({1, H, S, D}, dt, device);
        k_rope.alloc({1, Hkv, S, D}, dt, device);
        attn_bhsd.alloc({1, H, S, D}, dt, device);
        attn_merged.alloc({S, cfg.q_dim()}, dt, device);
        o.alloc({S, cfg.hidden_size}, dt, device);
        h.alloc({S, cfg.hidden_size}, dt, device);
        gate.alloc({S, cfg.intermediate_size}, dt, device);
        up.alloc({S, cfg.intermediate_size}, dt, device);
        act.alloc({S, cfg.intermediate_size}, dt, device);
        ffn.alloc({S, cfg.hidden_size}, dt, device);
        x_a.alloc({S, cfg.hidden_size}, dt, device);
        x_b.alloc({S, cfg.hidden_size}, dt, device);
        last_norm.alloc({1, cfg.hidden_size}, dt, device);
        logits.alloc({1, cfg.vocab_size}, dt, device);
    }
};

} // anonymous namespace

// ---- device blocks --------------------------------------------------------

DeviceBlock::~DeviceBlock()
{
    dev_free(ptr);
}

DeviceBlock::DeviceBlock(DeviceBlock&& other) noexcept : ptr(other.ptr), bytes(other.bytes)
{
    other.ptr = nullptr;
    other.bytes = 0;
}

DeviceBlock& DeviceBlock::operator=(DeviceBlock&& other) noexcept
{
    if (this != &other) {
        reset();
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}

void DeviceBlock::reset() noexcept
{
    dev_free(ptr);
    ptr = nullptr;
    bytes = 0;
}

// ---- Buffer ---------------------------------------------------------------

void Buffer::alloc(std::vector<int64_t> shape_, DataType dtype_, bool on_device)
{
    shape = std::move(shape_);
    dtype = dtype_;
    const size_t nbytes = static_cast<size_t>(numel()) * data_type_size(dtype);
    dev.reset();
    if (on_device) {
        bytes.clear();
        if (dev_alloc(dev.ptr, nbytes)) {
            dev.bytes = nbytes;
        }
        return;
    }
    bytes.assign(nbytes, 0);
}

size_t Buffer::nbytes() const noexcept
{
    return dev.ok() ? dev.bytes : bytes.size();
}

TensorView Buffer::view() noexcept
{
    return TensorView(shape, dtype, data());
}

TensorView Buffer::view() const noexcept
{
    return TensorView(shape, dtype, const_cast<void*>(data()));
}

bool Buffer::upload(const void* src)
{
    return dev.ok() && dev_copy(dev.ptr, src, dev.bytes, true);
}

bool Buffer::download(void* dst) const
{
    return dev.ok() && dev_copy(dst, dev.ptr, dev.bytes, false);
}

bool Q4Matrix::to_device()
{
    if (dev.ok()) {
        return true;
    }
    if (bytes.empty() || !dev_alloc(dev.ptr, bytes.size())) {
        return false;
    }
    dev.bytes = bytes.size();
    return dev_copy(dev.ptr, bytes.data(), bytes.size(), true);
}

bool Qwen2Weights::to_device()
{
    auto upload = [](Buffer& b) {
        if (b.bytes.empty() || b.dev.ok()) {
            return true;
        }
        if (!dev_alloc(b.dev.ptr, b.bytes.size())) {
            return false;
        }
        b.dev.bytes = b.bytes.size();
        return dev_copy(b.dev.ptr, b.bytes.data(), b.dev.bytes, true);
    };

    bool ok = embed.to_device() && upload(final_norm);
    if (!cfg.tie_word_embeddings) {
        ok = ok && lm_head.to_device();
    }
    for (auto& l : layers) {
        ok = ok && l.q_proj.to_device() && l.k_proj.to_device() && l.v_proj.to_device() &&
             l.o_proj.to_device() && l.gate_proj.to_device() && l.up_proj.to_device() &&
             l.down_proj.to_device();
        ok = ok && upload(l.q_bias) && upload(l.k_bias) && upload(l.v_bias) &&
             upload(l.attn_norm) && upload(l.ffn_norm);
    }
    return ok;
}

int64_t Buffer::numel() const noexcept
{
    if (shape.empty()) {
        return 0;
    }
    int64_t n = 1;
    for (const int64_t d : shape) {
        n *= d;
    }
    return n;
}

// ---- fake checkpoint ------------------------------------------------------

size_t Qwen2Weights::q4_nbytes() const noexcept
{
    size_t n = embed.nbytes() + (cfg.tie_word_embeddings ? 0 : lm_head.nbytes());
    for (const auto& l : layers) {
        n += l.q_proj.nbytes() + l.k_proj.nbytes() + l.v_proj.nbytes() + l.o_proj.nbytes() +
             l.gate_proj.nbytes() + l.up_proj.nbytes() + l.down_proj.nbytes();
    }
    return n;
}

size_t Qwen2Weights::dense_nbytes() const noexcept
{
    size_t n = final_norm.nbytes();
    for (const auto& l : layers) {
        n += l.q_bias.nbytes() + l.k_bias.nbytes() + l.v_bias.nbytes() + l.attn_norm.nbytes() +
             l.ffn_norm.nbytes();
    }
    return n;
}

Qwen2Weights make_fake_weights(const Qwen2Config& cfg, DataType act_dtype, uint64_t seed)
{
    Qwen2Weights w;
    w.cfg = cfg;
    w.act_dtype = act_dtype;
    if (act_dtype != DataType::f32 && act_dtype != DataType::f16) {
        return w;
    }

    Rng rng(seed);
    const float kWeight = 0.02f;  // Qwen's initializer_range
    w.embed = make_q4(cfg.vocab_size, cfg.hidden_size, -kWeight, kWeight, rng);
    w.final_norm.alloc({cfg.hidden_size}, act_dtype);
    fill_uniform(w.final_norm, 0.99f, 1.01f, rng);

    w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
    for (LayerWeights& l : w.layers) {
        l.q_proj = make_q4(cfg.q_dim(), cfg.hidden_size, -kWeight, kWeight, rng);
        l.k_proj = make_q4(cfg.kv_dim(), cfg.hidden_size, -kWeight, kWeight, rng);
        l.v_proj = make_q4(cfg.kv_dim(), cfg.hidden_size, -kWeight, kWeight, rng);
        l.o_proj = make_q4(cfg.hidden_size, cfg.q_dim(), -kWeight, kWeight, rng);
        l.gate_proj = make_q4(cfg.intermediate_size, cfg.hidden_size, -kWeight, kWeight, rng);
        l.up_proj = make_q4(cfg.intermediate_size, cfg.hidden_size, -kWeight, kWeight, rng);
        l.down_proj = make_q4(cfg.hidden_size, cfg.intermediate_size, -kWeight, kWeight, rng);

        if (cfg.qkv_bias()) {
            l.q_bias.alloc({cfg.q_dim()}, act_dtype);
            l.k_bias.alloc({cfg.kv_dim()}, act_dtype);
            l.v_bias.alloc({cfg.kv_dim()}, act_dtype);
            fill_uniform(l.q_bias, -kWeight, kWeight, rng);
            fill_uniform(l.k_bias, -kWeight, kWeight, rng);
            fill_uniform(l.v_bias, -kWeight, kWeight, rng);
        }

        l.attn_norm.alloc({cfg.hidden_size}, act_dtype);
        l.ffn_norm.alloc({cfg.hidden_size}, act_dtype);
        fill_uniform(l.attn_norm, 0.99f, 1.01f, rng);
        fill_uniform(l.ffn_norm, 0.99f, 1.01f, rng);
    }

    if (!cfg.tie_word_embeddings) {
        w.lm_head = make_q4(cfg.vocab_size, cfg.hidden_size, -kWeight, kWeight, rng);
    }
    return w;
}

// ---- state ----------------------------------------------------------------

struct Qwen2State::Impl {
    Qwen2Config cfg;
    DataType dtype = DataType::f32;
    Backend backend = Backend::CPU;
    void* stream = nullptr; ///< cudaStream_t for a device run
    bool cuda = false;
    int64_t capacity = 0;
    int64_t length = 0;
    int64_t chunk = -1; ///< S the scratch was allocated for
    std::vector<CacheLayer> cache;
    Scratch scratch;

    // Host staging at the two boundaries of a device run: the embedding lookup
    // and the q8 cache quantisation are host code, so their rows cross the bus.
    std::vector<uint8_t> embed_stage;
    std::vector<uint8_t> kv_stage;
    std::vector<uint8_t> logits_stage;
    DeviceBlock pos_dev; ///< int64 [1]
    DeviceBlock len_dev; ///< int32 [1]

    bool prof_on = false;
    Qwen2Profile prof;

    std::unique_ptr<Norm> norm;
    std::unique_ptr<Linear> linear;
    std::unique_ptr<Linear> linear_silu;
    std::unique_ptr<RoPE> rope;
    std::unique_ptr<Permute> permute;
    std::unique_ptr<Eltwise> add;
    std::unique_ptr<Eltwise> mul;
    std::unique_ptr<GroupQueryAttention> attn;

    void ensure_chunk(int64_t S)
    {
        if (chunk == S) {
            return;
        }
        scratch.alloc(cfg, dtype, S, cuda);
        chunk = S;
    }
};

Qwen2State::Qwen2State(const Qwen2Config& cfg, DataType act_dtype, int64_t capacity,
                       Backend backend, void* stream)
    : impl_(std::make_unique<Impl>())
{
    Impl& m = *impl_;
    m.cfg = cfg;
    m.dtype = act_dtype;
    m.backend = backend;
    m.stream = stream;
    m.cuda = backend == Backend::CUDA;
    m.capacity = std::max<int64_t>(capacity, 1);

    const int64_t plane = cfg.num_key_value_heads * m.capacity * cfg.head_dim;
    const int64_t scale_plane = cfg.num_key_value_heads * m.capacity;
    m.cache.resize(static_cast<size_t>(cfg.num_hidden_layers));
    for (CacheLayer& c : m.cache) {
        c.k.assign(static_cast<size_t>(plane), 0);
        c.v.assign(static_cast<size_t>(plane), 0);
        c.k_scale.assign(static_cast<size_t>(scale_plane), 1.0f);
        c.v_scale.assign(static_cast<size_t>(scale_plane), 1.0f);
        if (m.cuda) {
            const size_t kbytes = static_cast<size_t>(plane);
            const size_t sbytes = static_cast<size_t>(scale_plane) * sizeof(float);
            const bool ok = dev_alloc(c.k_dev.ptr, kbytes) && dev_alloc(c.v_dev.ptr, kbytes) &&
                            dev_alloc(c.ks_dev.ptr, sbytes) && dev_alloc(c.vs_dev.ptr, sbytes);
            c.k_dev.bytes = c.v_dev.bytes = kbytes;
            c.ks_dev.bytes = c.vs_dev.bytes = sbytes;
            if (!ok) {
                return;
            }
            dev_copy(c.k_dev.ptr, c.k.data(), kbytes, true);
            dev_copy(c.v_dev.ptr, c.v.data(), kbytes, true);
            dev_copy(c.ks_dev.ptr, c.k_scale.data(), sbytes, true);
            dev_copy(c.vs_dev.ptr, c.v_scale.data(), sbytes, true);
        }
    }

    NormAttributes na;
    na.type = NormType::RMSNorm;
    na.axis = -1;
    na.epsilon = cfg.rms_norm_eps;
    m.norm = Norm::create(na, backend);

    LinearAttributes la;
    m.linear = Linear::create(la, backend);
    LinearAttributes ga;
    ga.epilogue.type = EpilogueActivateType::Silu;
    m.linear_silu = Linear::create(ga, backend);

    RoPEAttributes ra;
    ra.base = cfg.rope_theta;
    ra.interleaved = false; // Qwen2 rotates the split halves (HF rotate_half)
    ra.rotary_dim = 0;      // the whole head
    m.rope = RoPE::create(ra, backend);

    PermuteAttributes pa;
    pa.perm = {0, 2, 1, 3};
    m.permute = Permute::create(pa, backend);

    EltwiseAttributes add_attrs;
    add_attrs.type = EltwiseType::Add;
    m.add = Eltwise::create(add_attrs, backend);
    EltwiseAttributes mul_attrs;
    mul_attrs.type = EltwiseType::Mul;
    m.mul = Eltwise::create(mul_attrs, backend);

    GroupQueryAttentionAttributes attn_attrs;
    attn_attrs.num_heads = cfg.num_attention_heads;
    attn_attrs.kv_num_heads = cfg.num_key_value_heads;
    attn_attrs.causal = true; // the KV-cache form
    m.attn = GroupQueryAttention::create(attn_attrs, backend);
}

Qwen2State::~Qwen2State() = default;

Backend Qwen2State::backend() const noexcept
{
    return impl_->backend;
}

void Qwen2State::reset() noexcept
{
    impl_->length = 0;
}

int64_t Qwen2State::length() const noexcept
{
    return impl_->length;
}

int64_t Qwen2State::capacity() const noexcept
{
    return impl_->capacity;
}

void Qwen2State::set_profile(bool on) noexcept
{
    impl_->prof_on = on;
}

void Qwen2State::reset_profile() noexcept
{
    impl_->prof = Qwen2Profile{};
}

const Qwen2Profile& Qwen2State::profile() const noexcept
{
    return impl_->prof;
}

size_t Qwen2State::cache_nbytes() const noexcept
{
    return impl_->cache.size() *
           (2 * static_cast<size_t>(impl_->cfg.num_key_value_heads) *
                static_cast<size_t>(impl_->capacity) *
                (static_cast<size_t>(impl_->cfg.head_dim) + sizeof(float)));
}

namespace {

Status run_linear(Linear& op, TensorView out, const TensorView& x, const Q4Matrix& w,
                  const Buffer* bias, const ComputeContext& ctx)
{
    TensorView ins[3] = {x, w.view(), TensorView{}};
    const size_t n = bias != nullptr && !bias->empty() ? 3 : 2;
    if (n == 3) {
        ins[2] = bias->view();
    }
    TensorView outs[1] = {out};
    return op.compute(outs, std::span<const TensorView>(ins, n), ctx);
}

} // anonymous namespace

Status Qwen2State::run(const Qwen2Weights& w, std::span<const int64_t> tokens,
                       const ComputeContext& ctx, std::vector<float>& last_logits)
{
    Impl& m = *impl_;
    const Qwen2Config& cfg = m.cfg;
    const int64_t S = static_cast<int64_t>(tokens.size());
    if (!config_ok(cfg) || w.act_dtype != m.dtype || w.cfg.hidden_size != cfg.hidden_size ||
        w.cfg.num_hidden_layers != cfg.num_hidden_layers || w.layers.size() != m.cache.size()) {
        return Status::invalid_argument;
    }
    if (S <= 0) {
        return Status::invalid_argument;
    }
    if (m.length + S > m.capacity) {
        return Status::invalid_argument;
    }
    if (m.cuda && (ctx.cuda_stream == nullptr || m.dtype != DataType::f16)) {
        return Status::unsupported;
    }
    struct StreamGuard {
        ~StreamGuard() { g_profile_stream = nullptr; }
    } stream_guard;
    g_profile_stream = m.cuda && m.prof_on ? ctx.cuda_stream : nullptr;

    const int64_t start = m.length;
    const int64_t H = cfg.num_attention_heads;
    const int64_t Hkv = cfg.num_key_value_heads;
    const int64_t D = cfg.head_dim;
    const int64_t hidden = cfg.hidden_size;
    const DataType dt = m.dtype;

    m.ensure_chunk(S);
    Scratch& s = m.scratch;

    int64_t pos0 = start;
    int32_t seq_len = static_cast<int32_t>(start + S);
    TensorView pos_view;
    TensorView len_view;
    if (m.cuda) {
        if (!m.pos_dev.ok() && !dev_alloc(m.pos_dev.ptr, sizeof(int64_t))) {
            return Status::out_of_memory;
        }
        if (!m.len_dev.ok() && !dev_alloc(m.len_dev.ptr, sizeof(int32_t))) {
            return Status::out_of_memory;
        }
        m.pos_dev.bytes = sizeof(int64_t);
        m.len_dev.bytes = sizeof(int32_t);
        dev_copy(m.pos_dev.ptr, &pos0, sizeof(pos0), true);
        dev_copy(m.len_dev.ptr, &seq_len, sizeof(seq_len), true);
        pos_view = make_view({1}, DataType::s64, m.pos_dev.ptr);
        len_view = make_view({1}, DataType::s32, m.len_dev.ptr);
    } else {
        pos_view = make_view({1}, DataType::s64, &pos0);
        len_view = make_view({1}, DataType::s32, &seq_len);
    }

    // ---- embedding --------------------------------------------------------
    Buffer* x_in = &s.x_a;
    Buffer* x_out = &s.x_b;
    {
        PhaseTimer t(m.prof.embed_ms, m.prof_on);
        if (m.cuda) {
            m.embed_stage.resize(static_cast<size_t>(S) * static_cast<size_t>(hidden) *
                                 data_type_size(dt));
            embed_rows(w.embed, tokens, m.embed_stage.data(), dt);
            if (!x_in->upload(m.embed_stage.data())) {
                return Status::internal;
            }
        } else {
            embed_rows(w.embed, tokens, x_in->data(), dt);
        }
    }

    const int64_t qdim = cfg.q_dim();
    const int64_t kvdim = cfg.kv_dim();

    for (size_t li = 0; li < m.cache.size(); ++li) {
        const LayerWeights& lw = w.layers[li];
        CacheLayer& cl = m.cache[li];

        // ---- attention pre-norm ------------------------------------------
        TensorView xn = s.xn.view();
        {
            PhaseTimer t(m.prof.norm_ms, m.prof_on);
            TensorView ins[2] = {x_in->view(), lw.attn_norm.view()};
            const Status st = m.norm->compute({&xn, 1}, ins, ctx);
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- q/k/v projections -------------------------------------------
        {
            PhaseTimer t(m.prof.qkv_ms, m.prof_on);
            TensorView q = s.q_lin.view();
            Status st = run_linear(*m.linear, q, xn, lw.q_proj,
                                   cfg.qkv_bias() ? &lw.q_bias : nullptr, ctx);
            if (st != Status::ok) {
                return st;
            }
            TensorView k = s.k_lin.view();
            st = run_linear(*m.linear, k, xn, lw.k_proj,
                            cfg.qkv_bias() ? &lw.k_bias : nullptr, ctx);
            if (st != Status::ok) {
                return st;
            }
            TensorView v = s.v_lin.view();
            st = run_linear(*m.linear, v, xn, lw.v_proj,
                            cfg.qkv_bias() ? &lw.v_bias : nullptr, ctx);
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- [S, H, D] -> [1, H, S, D], then RoPE -----------------------
        {
            TensorView q_row = make_view({1, S, H, D}, dt, s.q_lin.data());
            TensorView q_bhsd = s.q_bhsd.view();
            TensorView q_rope = s.q_rope.view();
            Status st;
            {
                PhaseTimer t(m.prof.permute_ms, m.prof_on);
                st = m.permute->compute({&q_bhsd, 1}, {&q_row, 1}, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView rin[6] = {q_bhsd, TensorView{}, TensorView{}, TensorView{},
                                 TensorView{}, pos_view};
            {
                PhaseTimer t(m.prof.rope_ms, m.prof_on);
                st = m.rope->compute({&q_rope, 1}, rin, ctx);
            }
            if (st != Status::ok) {
                return st;
            }

            TensorView k_row = make_view({1, S, Hkv, D}, dt, s.k_lin.data());
            TensorView k_bhsd = s.k_bhsd.view();
            TensorView k_rope = s.k_rope.view();
            {
                PhaseTimer t(m.prof.permute_ms, m.prof_on);
                st = m.permute->compute({&k_bhsd, 1}, {&k_row, 1}, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView kin[6] = {k_bhsd, TensorView{}, TensorView{}, TensorView{},
                                 TensorView{}, pos_view};
            {
                PhaseTimer t(m.prof.rope_ms, m.prof_on);
                st = m.rope->compute({&k_rope, 1}, kin, ctx);
            }
            if (st != Status::ok) {
                return st;
            }

            // ---- append the step's K/V as q8 ----------------------------
            // A device run quantizes from a downloaded copy, then uploads the
            // rows the step wrote into the device planes.
            PhaseTimer t(m.prof.cache_ms, m.prof_on);
            TensorView k_src = s.k_rope.view();
            TensorView v_src = s.v_lin.view();
            if (m.cuda) {
                const size_t kv_bytes = static_cast<size_t>(S) * static_cast<size_t>(kvdim) *
                                        data_type_size(dt);
                m.kv_stage.resize(2 * kv_bytes);
                s.k_rope.download(m.kv_stage.data());
                s.v_lin.download(m.kv_stage.data() + kv_bytes);
                k_src = make_view({1, Hkv, S, D}, dt, m.kv_stage.data());
                v_src = make_view({S, kvdim}, dt, m.kv_stage.data() + kv_bytes);
            }
            if (dt == DataType::f32) {
                append_cache<float>(cfg, cl, k_src, v_src, m.capacity, start, S);
            } else {
                append_cache<half>(cfg, cl, k_src, v_src, m.capacity, start, S);
            }
            if (m.cuda) {
                for (int64_t h = 0; h < Hkv; ++h) {
                    const int64_t row = h * m.capacity + start;
                    const size_t rows = static_cast<size_t>(S * D);
                    dev_copy(static_cast<int8_t*>(cl.k_dev.ptr) + row * D, cl.k.data() + row * D,
                             rows, true);
                    dev_copy(static_cast<int8_t*>(cl.v_dev.ptr) + row * D, cl.v.data() + row * D,
                             rows, true);
                    dev_copy(static_cast<float*>(cl.ks_dev.ptr) + row, cl.k_scale.data() + row,
                             static_cast<size_t>(S) * sizeof(float), true);
                    dev_copy(static_cast<float*>(cl.vs_dev.ptr) + row, cl.v_scale.data() + row,
                             static_cast<size_t>(S) * sizeof(float), true);
                }
            }
        }

        // ---- cached causal attention ------------------------------------
        {
            TensorView q = s.q_rope.view();
            void* kp = m.cuda ? cl.k_dev.ptr : static_cast<void*>(cl.k.data());
            void* vp = m.cuda ? cl.v_dev.ptr : static_cast<void*>(cl.v.data());
            void* ksp = m.cuda ? cl.ks_dev.ptr : static_cast<void*>(cl.k_scale.data());
            void* vsp = m.cuda ? cl.vs_dev.ptr : static_cast<void*>(cl.v_scale.data());
            TensorView k_cache = make_view({1, Hkv, m.capacity, D}, DataType::s8, kp);
            TensorView v_cache = make_view({1, Hkv, m.capacity, D}, DataType::s8, vp);
            TensorView k_scale = make_view({1, Hkv, m.capacity}, DataType::f32, ksp);
            TensorView v_scale = make_view({1, Hkv, m.capacity}, DataType::f32, vsp);
            TensorView ins[9] = {q, k_cache, v_cache, len_view, pos_view, k_scale, v_scale,
                                 TensorView{}, TensorView{}};
            TensorView out = s.attn_bhsd.view();
            PhaseTimer t(m.prof.attention_ms, m.prof_on);
            const Status st = m.attn->compute({&out, 1}, ins, ctx);
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- output projection and residual ------------------------------
        {
            TensorView attn_row = make_view({1, S, H, D}, dt, s.attn_merged.data());
            TensorView attn_bhsd = s.attn_bhsd.view();
            Status st;
            {
                PhaseTimer t(m.prof.permute_ms, m.prof_on);
                st = m.permute->compute({&attn_row, 1}, {&attn_bhsd, 1}, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView merged = make_view({S, qdim}, dt, s.attn_merged.data());
            TensorView o = s.o.view();
            {
                PhaseTimer t(m.prof.o_proj_ms, m.prof_on);
                st = run_linear(*m.linear, o, merged, lw.o_proj, nullptr, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView h = s.h.view();
            TensorView ins[2] = {x_in->view(), o};
            {
                PhaseTimer t(m.prof.residual_ms, m.prof_on);
                st = m.add->compute({&h, 1}, ins, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- FFN pre-norm -----------------------------------------------
        {
            PhaseTimer t(m.prof.norm_ms, m.prof_on);
            TensorView xn2 = s.xn.view();
            TensorView ins[2] = {s.h.view(), lw.ffn_norm.view()};
            const Status st = m.norm->compute({&xn2, 1}, ins, ctx);
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- SwiGLU FFN --------------------------------------------------
        {
            TensorView xn2 = s.xn.view();
            TensorView gate = s.gate.view();
            TensorView gin[2] = {xn2, lw.gate_proj.view()};
            Status st;
            {
                PhaseTimer t(m.prof.ffn_gate_ms, m.prof_on);
                st = m.linear_silu->compute({&gate, 1}, gin, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView up = s.up.view();
            TensorView uin[2] = {xn2, lw.up_proj.view()};
            {
                PhaseTimer t(m.prof.ffn_up_ms, m.prof_on);
                st = m.linear->compute({&up, 1}, uin, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView act = s.act.view();
            TensorView min[2] = {gate, up};
            {
                PhaseTimer t(m.prof.ffn_mul_ms, m.prof_on);
                st = m.mul->compute({&act, 1}, min, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
            TensorView ffn = s.ffn.view();
            {
                PhaseTimer t(m.prof.ffn_down_ms, m.prof_on);
                st = run_linear(*m.linear, ffn, act, lw.down_proj, nullptr, ctx);
            }
            if (st != Status::ok) {
                return st;
            }
        }

        // ---- FFN residual, then ping-pong the hidden state ---------------
        {
            PhaseTimer t(m.prof.residual_ms, m.prof_on);
            TensorView out = x_out->view();
            TensorView ins[2] = {s.h.view(), s.ffn.view()};
            const Status st = m.add->compute({&out, 1}, ins, ctx);
            if (st != Status::ok) {
                return st;
            }
        }
        std::swap(x_in, x_out);
    }

    // ---- final norm on the last row, then the LM head ---------------------
    {
        const size_t esize = data_type_size(dt);
        TensorView last =
            make_view({1, hidden}, dt,
                      static_cast<uint8_t*>(x_in->data()) +
                          static_cast<size_t>(S - 1) * static_cast<size_t>(hidden) * esize);
        TensorView ln = s.last_norm.view();
        TensorView ins[2] = {last, w.final_norm.view()};
        Status st;
        {
            PhaseTimer t(m.prof.norm_ms, m.prof_on);
            st = m.norm->compute({&ln, 1}, ins, ctx);
        }
        if (st != Status::ok) {
            return st;
        }
        TensorView logits = s.logits.view();
        const Q4Matrix& head = w.cfg.tie_word_embeddings ? w.embed : w.lm_head;
        {
            PhaseTimer t(m.prof.lm_head_ms, m.prof_on);
            st = run_linear(*m.linear, logits, ln, head, nullptr, ctx);
            if (st != Status::ok) {
                return st;
            }
            last_logits.resize(static_cast<size_t>(cfg.vocab_size));
            if (m.cuda) {
                m.logits_stage.resize(static_cast<size_t>(cfg.vocab_size) * data_type_size(dt));
                if (!s.logits.download(m.logits_stage.data())) {
                    return Status::internal;
                }
            }
            const void* logit_src = m.cuda ? m.logits_stage.data() : s.logits.data();
            for (int64_t i = 0; i < cfg.vocab_size; ++i) {
                last_logits[static_cast<size_t>(i)] = elem_f32(dt, logit_src, i);
            }
        }
    }

    if (m.prof_on) {
        Qwen2Profile& p = m.prof;
        p.total_ms = p.embed_ms + p.norm_ms + p.qkv_ms + p.permute_ms + p.rope_ms +
                     p.cache_ms + p.attention_ms + p.o_proj_ms + p.residual_ms +
                     p.ffn_gate_ms + p.ffn_up_ms + p.ffn_mul_ms + p.ffn_down_ms +
                     p.lm_head_ms;
    }

    m.length += S;
    return Status::ok;
}

} // namespace nnops::llm
