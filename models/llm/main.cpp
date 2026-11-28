/// @file main.cpp
/// @brief Qwen2/Qwen2.5 fake-weight inference demo.
///
/// Builds the fake checkpoint, runs a prefill block, then greedily decodes a
/// few tokens, printing per-phase timings and the last row's top logits. With
/// --selftest it replays the prompt one token at a time and compares the
/// stepwise logits with the prefill's — the check that the q8 cache, RoPE
/// positions and the causal range agree between the two attention routes.
///
/// There is no tokenizer: the prompt is random token ids drawn from the seeded
/// generator, which is enough for a shape- and route-level demonstration.

#include "models/llm/qwen2_model.hpp"

#include "nnops/core/openmp_backend.hpp"

#ifdef NNOPS_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace nnops;
using namespace nnops::llm;

namespace {

struct Args {
    std::string preset = "qwen2.5-0.5b";
    std::string dtype = "f32";
    int64_t prompt = 16;
    int64_t steps = 8;
    int64_t max_seq = 0;
    int threads = 0;
    uint64_t seed = 1234;
    bool selftest = false;
    bool bench = false;
    bool profile = false;
    bool cuda = false;
};

void usage()
{
    std::printf(
        "usage: nnops_qwen2_llm [options]\n"
        "  --preset tiny|qwen2.5-0.5b|qwen2.5-4b   model shape (default qwen2.5-0.5b)\n"
        "  --dtype f32|f16                        activation dtype (default f32)\n"
        "  --prompt N                             prompt tokens (default 16)\n"
        "  --steps N                              decoded tokens (default 8)\n"
        "  --max-seq N                            KV-cache capacity (default prompt+steps+8)\n"
        "  --threads N                            CPU workers (default OpenMP max)\n"
        "  --backend cpu|cuda                     where the operators run (default cpu)\n"
        "  --seed N                               fake-weight / prompt seed (default 1234)\n"
        "  --selftest                             stepwise vs prefill logits check\n"
        "  --bench                                prefill 64/128/256/384/512 + 1-token decode\n"
        "  --profile                              with --bench: per-phase time breakdown\n"
        "  --help\n");
}

bool parse_args(int argc, char** argv, Args& a, bool& help)
{
    help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::printf("%s needs a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            help = true;
        } else if (arg == "--selftest") {
            a.selftest = true;
        } else if (arg == "--bench") {
            a.bench = true;
        } else if (arg == "--profile") {
            a.profile = true;
        } else if (arg == "--preset") {
            const char* v = value("--preset");
            if (!v) { return false; }
            a.preset = v;
            if (a.preset != "tiny" && a.preset != "qwen2.5-0.5b" && a.preset != "qwen2.5-4b") {
                std::printf("unknown preset: %s\n", v);
                return false;
            }
        } else if (arg == "--dtype") {
            const char* v = value("--dtype");
            if (!v) { return false; }
            a.dtype = v;
            if (a.dtype != "f32" && a.dtype != "f16") {
                std::printf("unknown dtype: %s\n", v);
                return false;
            }
        } else if (arg == "--prompt") {
            const char* v = value("--prompt");
            if (!v) { return false; }
            a.prompt = std::atoll(v);
        } else if (arg == "--steps") {
            const char* v = value("--steps");
            if (!v) { return false; }
            a.steps = std::atoll(v);
        } else if (arg == "--max-seq") {
            const char* v = value("--max-seq");
            if (!v) { return false; }
            a.max_seq = std::atoll(v);
        } else if (arg == "--backend") {
            const char* v = value("--backend");
            if (!v) { return false; }
            if (std::strcmp(v, "cuda") == 0) {
                a.cuda = true;
            } else if (std::strcmp(v, "cpu") != 0) {
                std::printf("unknown backend: %s\n", v);
                return false;
            }
        } else if (arg == "--threads") {
            const char* v = value("--threads");
            if (!v) { return false; }
            a.threads = static_cast<int>(std::atoll(v));
        } else if (arg == "--seed") {
            const char* v = value("--seed");
            if (!v) { return false; }
            a.seed = static_cast<uint64_t>(std::strtoull(v, nullptr, 10));
        } else {
            std::printf("unknown option: %s\n", argv[i]);
            return false;
        }
    }
    if (a.profile) {
        a.bench = true;
    }
    return a.prompt > 0 && a.steps >= 0;
}

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Top1 {
    int64_t id = 0;
    float value = 0.0f;
};

Top1 argmax(const std::vector<float>& v)
{
    Top1 best;
    best.value = v.empty() ? 0.0f : v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > best.value) {
            best.value = v[i];
            best.id = static_cast<int64_t>(i);
        }
    }
    return best;
}

void print_topk(const std::vector<float>& v, int k)
{
    std::vector<int64_t> ids(v.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int64_t>(i);
    }
    const int take = std::min<int>(k, static_cast<int>(ids.size()));
    std::partial_sort(ids.begin(), ids.begin() + take, ids.end(),
                      [&](int64_t a, int64_t b) {
                          return v[static_cast<size_t>(a)] > v[static_cast<size_t>(b)];
                      });
    for (int i = 0; i < take; ++i) {
        const int64_t id = ids[static_cast<size_t>(i)];
        std::printf("    id %7lld   % .6f\n", static_cast<long long>(id),
                    static_cast<double>(v[static_cast<size_t>(id)]));
    }
}

double mb(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

/// Best of the repeated runs: the host schedules a 4-thread OpenMP region
/// across a 6P+8E core mix, so single runs drift by 2-3x and the minimum is the
/// stable estimate of the kernel's own cost.
double best(std::vector<double> v)
{
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

/// Prefill latency at the requested prompt lengths, then the latency of one
/// decoded token with that context already in the q8 cache. Each number is the
/// median of repeated runs after one warm-up, so scratch allocation and cold
/// pages are not in the timed path.
bool run_bench(const Qwen2Weights& weights, const ComputeContext& ctx, uint64_t seed,
               bool profile)
{
    const Qwen2Config& cfg = weights.cfg;
    const int64_t lengths[] = {64, 128, 256, 384, 512};
    const int64_t max_len = 512;
    constexpr int kPrefillReps = 5;
    constexpr int kDecodeReps = 9;

    Qwen2State state(cfg, weights.act_dtype, max_len + 32, ctx.expected_backend,
                     ctx.cuda_stream);
    std::printf("  kv cache    %.1f MB\n", mb(state.cache_nbytes()));

    std::mt19937_64 rng(seed ^ 0xC0FFEE0DDBA11ull);
    std::uniform_int_distribution<int64_t> dist(0, cfg.vocab_size - 1);
    std::vector<int64_t> tokens(static_cast<size_t>(max_len));
    for (int64_t& t : tokens) {
        t = dist(rng);
    }
    std::vector<float> logits;

    std::printf("\nbenchmark (best of %d prefill runs, %d decode steps)\n", kPrefillReps,
                kDecodeReps);
    std::printf("  seq   prefill ms   prefill tok/s   decode ms/token\n");
    for (const int64_t L : lengths) {
        const std::span<const int64_t> block(tokens.data(), static_cast<size_t>(L));
        state.reset();
        if (state.run(weights, block, ctx, logits) != Status::ok) {
            std::printf("prefill warm-up failed\n");
            return false;
        }

        std::vector<double> prefill_ms;
        for (int r = 0; r < kPrefillReps; ++r) {
            state.reset();
            const auto t0 = std::chrono::steady_clock::now();
            const Status st = state.run(weights, block, ctx, logits);
            prefill_ms.push_back(ms_since(t0));
            if (st != Status::ok) {
                std::printf("prefill failed: %s\n", to_string(st));
                return false;
            }
        }

        int64_t token = argmax(logits).id;
        if (state.run(weights, std::span<const int64_t>(&token, 1), ctx, logits) !=
            Status::ok) {
            std::printf("decode warm-up failed\n");
            return false;
        }
        std::vector<double> decode_ms;
        for (int r = 0; r < kDecodeReps; ++r) {
            token = argmax(logits).id;
            const auto t0 = std::chrono::steady_clock::now();
            const Status st = state.run(weights, std::span<const int64_t>(&token, 1), ctx,
                                        logits);
            decode_ms.push_back(ms_since(t0));
            if (st != Status::ok) {
                std::printf("decode failed: %s\n", to_string(st));
                return false;
            }
        }

        const double prefill = best(prefill_ms);
        const double decode = best(decode_ms);
        std::printf("  %4lld   %10.2f   %13.1f   %11.2f\n", static_cast<long long>(L),
                    prefill, static_cast<double>(L) * 1000.0 / prefill, decode);
        state.reset();
    }

    if (!profile) {
        return true;
    }

    // ---- phase breakdown at the longest context --------------------------
    constexpr int64_t kProfLen = 512;
    constexpr int kProfPrefill = 3;
    constexpr int kProfDecode = 9;
    const std::span<const int64_t> block(tokens.data(), static_cast<size_t>(kProfLen));

    state.set_profile(false);
    state.reset();
    if (state.run(weights, block, ctx, logits) != Status::ok) {
        std::printf("profile warm-up failed\n");
        return false;
    }

    state.set_profile(true);
    state.reset_profile();
    for (int r = 0; r < kProfPrefill; ++r) {
        state.reset();
        if (state.run(weights, block, ctx, logits) != Status::ok) {
            std::printf("profile prefill failed\n");
            return false;
        }
    }
    const Qwen2Profile pre = state.profile();

    state.set_profile(false);
    int64_t token = argmax(logits).id;
    if (state.run(weights, std::span<const int64_t>(&token, 1), ctx, logits) != Status::ok) {
        std::printf("profile decode warm-up failed\n");
        return false;
    }
    state.set_profile(true);
    state.reset_profile();
    for (int r = 0; r < kProfDecode; ++r) {
        token = argmax(logits).id;
        if (state.run(weights, std::span<const int64_t>(&token, 1), ctx, logits) != Status::ok) {
            std::printf("profile decode failed\n");
            return false;
        }
    }
    const Qwen2Profile dec = state.profile();
    state.set_profile(false);
    state.reset();

    struct Row {
        const char* name;
        double Qwen2Profile::*field;
    };
    static constexpr Row rows[] = {
        {"embed", &Qwen2Profile::embed_ms},
        {"norm", &Qwen2Profile::norm_ms},
        {"qkv", &Qwen2Profile::qkv_ms},
        {"permute", &Qwen2Profile::permute_ms},
        {"rope", &Qwen2Profile::rope_ms},
        {"cache q8", &Qwen2Profile::cache_ms},
        {"attention", &Qwen2Profile::attention_ms},
        {"o_proj", &Qwen2Profile::o_proj_ms},
        {"residual", &Qwen2Profile::residual_ms},
        {"ffn gate", &Qwen2Profile::ffn_gate_ms},
        {"ffn up", &Qwen2Profile::ffn_up_ms},
        {"ffn mul", &Qwen2Profile::ffn_mul_ms},
        {"ffn down", &Qwen2Profile::ffn_down_ms},
        {"lm_head", &Qwen2Profile::lm_head_ms},
    };
    const double pre_sum = pre.total_ms / kProfPrefill;
    const double dec_sum = dec.total_ms / kProfDecode;

    std::printf("\nphase breakdown (per run: prefill %lld, decode 1; %d / %d runs)\n",
                static_cast<long long>(kProfLen), kProfPrefill, kProfDecode);
    std::printf("  %-12s %11s %7s %11s %7s\n", "phase", "prefill ms", "share",
                "decode ms", "share");
    for (const Row& row : rows) {
        const double pm = (pre.*row.field) / kProfPrefill;
        const double dm = (dec.*row.field) / kProfDecode;
        std::printf("  %-12s %11.3f %6.1f%% %11.3f %6.1f%%\n", row.name, pm,
                    pre_sum > 0.0 ? pm * 100.0 / pre_sum : 0.0, dm,
                    dec_sum > 0.0 ? dm * 100.0 / dec_sum : 0.0);
    }
    std::printf("  %-12s %11.3f %6.1f%% %11.3f %6.1f%%\n", "total", pre_sum, 100.0, dec_sum,
                100.0);
    return true;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    Args args;
    bool help = false;
    if (!parse_args(argc, argv, args, help)) {
        usage();
        return 1;
    }
    if (help) {
        usage();
        return 0;
    }

    const Qwen2Config cfg = qwen2_config_preset(args.preset);
    const DataType act = args.dtype == "f16" ? DataType::f16 : DataType::f32;
    const int64_t capacity =
        args.max_seq > 0 ? args.max_seq : args.prompt + args.steps + 8;

    std::printf("Qwen2 fake-weight inference\n");
    std::printf("  preset      %.*s: %lld layers, hidden %lld, %lld heads / %lld kv, "
                "head_dim %lld, ffn %lld, vocab %lld\n",
                static_cast<int>(cfg.name.size()), cfg.name.data(),
                static_cast<long long>(cfg.num_hidden_layers),
                static_cast<long long>(cfg.hidden_size),
                static_cast<long long>(cfg.num_attention_heads),
                static_cast<long long>(cfg.num_key_value_heads),
                static_cast<long long>(cfg.head_dim),
                static_cast<long long>(cfg.intermediate_size),
                static_cast<long long>(cfg.vocab_size));
    std::printf("  activations %s | weights q4_1 | kv cache q8 (continuous, capacity %lld)\n",
                args.dtype.c_str(), static_cast<long long>(capacity));

    ComputeContext ctx;
    ctx.cpu = openmp_backend(args.threads);
    void* stream = nullptr;
    if (args.cuda) {
#ifdef NNOPS_HAS_CUDA
        if (act != DataType::f16) {
            std::printf("  cuda        needs --dtype f16 (the q4_1 device route is w4a16)\n");
            return 1;
        }
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            std::printf("  cuda        stream creation failed\n");
            return 1;
        }
        stream = s;
        ctx.cuda_stream = stream;
        ctx.expected_backend = Backend::CUDA;
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::printf("  cuda        %s, %d SMs, stream %p\n", prop.name, prop.multiProcessorCount,
                    stream);
#else
        std::printf("  cuda        this build has no CUDA backend\n");
        return 1;
#endif
    }
    std::printf("  cpu         %d workers%s\n", ctx.cpu.worker_count(),
                openmp_available() ? "" : " (built without OpenMP)");

    auto t0 = std::chrono::steady_clock::now();
    Qwen2Weights weights = make_fake_weights(cfg, act, args.seed);
    const double weight_ms = ms_since(t0);
    std::printf("  weights     %.1f MB q4_1 + %.1f MB dense in %.1f ms\n",
                mb(weights.q4_nbytes()), mb(weights.dense_nbytes()), weight_ms);
    if (args.cuda) {
        const auto tu = std::chrono::steady_clock::now();
        if (!weights.to_device()) {
            std::printf("  cuda        weight upload failed\n");
            return 1;
        }
        std::printf("  upload      %.1f MB in %.1f ms\n",
                    mb(weights.q4_nbytes() + weights.dense_nbytes()), ms_since(tu));
    }

    if (args.bench) {
        return run_bench(weights, ctx, args.seed, args.profile) ? 0 : 1;
    }

    Qwen2State state(cfg, act, capacity, ctx.expected_backend, ctx.cuda_stream);
    std::printf("  kv cache    %.1f MB\n", mb(state.cache_nbytes()));

    std::mt19937_64 rng(args.seed ^ 0x51ED270B7F1A3C5Dull);
    std::uniform_int_distribution<int64_t> token_dist(0, cfg.vocab_size - 1);
    std::vector<int64_t> prompt(static_cast<size_t>(args.prompt));
    for (int64_t& id : prompt) {
        id = token_dist(rng);
    }
    std::printf("  prompt      ");
    for (const int64_t id : prompt) {
        std::printf("%lld ", static_cast<long long>(id));
    }
    std::printf("\n\n");

    std::vector<float> logits;

    // ---- prefill ---------------------------------------------------------
    t0 = std::chrono::steady_clock::now();
    Status st = state.run(weights, prompt, ctx, logits);
    const double prefill_ms = ms_since(t0);
    if (st != Status::ok) {
        std::printf("prefill failed: %s\n", to_string(st));
        return 1;
    }
    std::printf("prefill  %lld tokens in %.1f ms (%.1f tok/s)\n",
                static_cast<long long>(args.prompt), prefill_ms,
                static_cast<double>(args.prompt) * 1000.0 / prefill_ms);
    std::printf("  last-row logits, top 5:\n");
    print_topk(logits, 5);
    int64_t next = argmax(logits).id;
    std::printf("  greedy next id %lld\n\n", static_cast<long long>(next));

    // ---- selftest: the same prompt one token at a time --------------------
    if (args.selftest) {
        const std::vector<float> prefill_logits = logits;
        state.reset();
        std::vector<float> step_logits;
        auto ts = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt.size(); ++i) {
            st = state.run(weights, std::span<const int64_t>(&prompt[i], 1), ctx, step_logits);
            if (st != Status::ok) {
                std::printf("selftest step %zu failed: %s\n", i, to_string(st));
                return 1;
            }
        }
        const double step_ms = ms_since(ts);
        float max_diff = 0.0f;
        float max_abs = 0.0f;
        for (size_t i = 0; i < prefill_logits.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(prefill_logits[i] - step_logits[i]));
            max_abs = std::max(max_abs, std::fabs(prefill_logits[i]));
        }
        const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
        const float tol = act == DataType::f16 ? 5e-2f : 2e-3f;
        const bool pass = rel <= tol && argmax(prefill_logits).id == argmax(step_logits).id;
        std::printf("selftest  stepwise %.1f ms; max |diff| %.3e (%.3e of max %.3e)\n",
                    step_ms, static_cast<double>(max_diff), static_cast<double>(rel),
                    static_cast<double>(max_abs));
        std::printf("          greedy %lld vs %lld -> %s\n",
                    static_cast<long long>(argmax(prefill_logits).id),
                    static_cast<long long>(argmax(step_logits).id), pass ? "PASS" : "FAIL");
        if (!pass) {
            return 1;
        }
    }

    // ---- decode ----------------------------------------------------------
    std::vector<int64_t> generated;
    for (int64_t i = 0; i < args.steps; ++i) {
        const int64_t token = next;
        t0 = std::chrono::steady_clock::now();
        st = state.run(weights, std::span<const int64_t>(&token, 1), ctx, logits);
        const double step_ms = ms_since(t0);
        if (st != Status::ok) {
            std::printf("decode step %lld failed: %s\n", static_cast<long long>(i), to_string(st));
            return 1;
        }
        next = argmax(logits).id;
        generated.push_back(token);
        std::printf("decode %2lld  token %7lld -> next %7lld   %.2f ms\n",
                    static_cast<long long>(i + 1), static_cast<long long>(token),
                    static_cast<long long>(next), step_ms);
    }

    std::printf("\ngenerated ids:");
    for (const int64_t id : generated) {
        std::printf(" %lld", static_cast<long long>(id));
    }
    std::printf("\nkvcache length %lld / %lld\n", static_cast<long long>(state.length()),
                static_cast<long long>(state.capacity()));
    return 0;
}
