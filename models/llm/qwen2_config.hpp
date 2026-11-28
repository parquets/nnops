#pragma once
/// @file qwen2_config.hpp
/// @brief The Qwen2/Qwen2.5 decoder shape, and the presets this demo builds.
///
/// Qwen2 is the plain decoder stack: RMSNorm before each sub-layer, a GQA
/// attention with an optional bias on the three projections, NeoX-style RoPE
/// over the whole head, a SwiGLU FFN, and a tied (0.5B/1.5B) or untied output
/// projection. There is no sliding window in the 2.5 checkpoints
/// (`use_sliding_window=false`), no QK norm, and no output gate — those are
/// Qwen3/Qwen3.5.
///
/// The demo builds fake weights, so every field here is a shape decision, not
/// a pretrained value. `qwen2.5-0.5b` matches the checkpoint in
/// D:\llm_models\models\qwen\Qwen2___5-0___5B-Instruct field for field;
/// `tiny` keeps the same architecture at a size a unit test can run; and
/// `qwen2.5-4b` is the 4B profile llama.cpp's qwen2 loader classifies
/// (40 layers, 20 query heads) filled out with a Qwen3-4B-width FFN, since no
/// official Qwen2.5-4B config exists. Head dims, vocab and FFN widths are all
/// multiples of the q4_1 block (32), which is what lets one weight row be
/// quantized block for block without a tail.

#include <cstdint>
#include <string_view>

namespace nnops::llm {

struct Qwen2Config {
    std::string_view name = "tiny";

    int64_t vocab_size = 2048;
    int64_t hidden_size = 256;
    int64_t intermediate_size = 1024;
    int64_t num_hidden_layers = 4;

    int64_t num_attention_heads = 4;
    int64_t num_key_value_heads = 2;
    /// Qwen2.5 uses hidden_size / num_attention_heads; kept explicit because
    /// Qwen3 widened it to 128 independently.
    int64_t head_dim = 64;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 1e6f;

    bool tie_word_embeddings = true;
    bool attention_bias = true;

    int64_t q_dim() const noexcept { return num_attention_heads * head_dim; }
    int64_t kv_dim() const noexcept { return num_key_value_heads * head_dim; }

    /// q/k/v bias tensors are part of the checkpoint when attention_bias is on;
    /// o_proj never carries one.
    bool qkv_bias() const noexcept { return attention_bias; }
};

inline Qwen2Config qwen2_config_preset(std::string_view name)
{
    if (name == "tiny") {
        return Qwen2Config{
            .name = "tiny",
            .vocab_size = 2048,
            .hidden_size = 256,
            .intermediate_size = 1024,
            .num_hidden_layers = 4,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .head_dim = 64,
        };
    }
    if (name == "qwen2.5-0.5b") {
        return Qwen2Config{
            .name = "qwen2.5-0.5b",
            .vocab_size = 151936,
            .hidden_size = 896,
            .intermediate_size = 4864,
            .num_hidden_layers = 24,
            .num_attention_heads = 14,
            .num_key_value_heads = 2,
            .head_dim = 64,
        };
    }
    if (name == "qwen2.5-4b") {
        // No official Qwen2.5-4B checkpoint; llama.cpp classifies 40 layers /
        // 20 heads as its 4B qwen2 type, and 9728 is the Qwen3-4B FFN width
        // that makes the parameter count land near 4B.
        return Qwen2Config{
            .name = "qwen2.5-4b",
            .vocab_size = 151936,
            .hidden_size = 2560,
            .intermediate_size = 9728,
            .num_hidden_layers = 40,
            .num_attention_heads = 20,
            .num_key_value_heads = 4,
            .head_dim = 128,
            .tie_word_embeddings = false,
        };
    }
    return Qwen2Config{};
}

} // namespace nnops::llm
