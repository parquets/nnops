# Qwen2.5 fake-weight inference

A runnable Qwen2/Qwen2.5 decoder built from the library's operators, with
**random q4_1 weights**, a **continuous (dense) q8 KV cache**, and **f32 or f16
activations**. It exists to exercise the operator set on a real transformer
graph shape; there is no tokenizer and no pretrained checkpoint.

```sh
cmake --build build --config Release --target nnops_qwen2_llm
./build/llm/Release/nnops_qwen2_llm --preset qwen2.5-0.5b --dtype f16 \
    --prompt 16 --steps 8 --threads 20
```

```
Qwen2 fake-weight inference
  preset      qwen2.5-0.5b: 24 layers, hidden 896, 14 heads / 2 kv, head_dim 64, ffn 4864, vocab 151936
  activations f16 | weights q4_1 | kv cache q8 (continuous, capacity 32)
  cpu         20 workers
  weights     294.4 MB q4_1 + 0.1 MB dense in 1747.4 ms
  kv cache    0.2 MB
  ...
prefill  16 tokens in 57.8 ms (276.7 tok/s)
decode  1  token   41937 -> next   41937   12.14 ms
```

## Model

The graph is HuggingFace's `Qwen2ForCausalLM` forward (and llama.cpp's
`models/qwen2.cpp`), with Qwen2.5's values: RMSNorm epsilon 1e-6, RoPE theta
1e6, NeoX (split-half) rotation over the whole head, biases on q/k/v, SwiGLU,
and a tied output projection on the 0.5B.

| step | operator |
|---|---|
| token embedding | q4_1 row decode (`Embed` has no q4_1 form) |
| attention pre-norm | `Norm` RMSNorm |
| q/k/v projections | `Linear`, q4_1 weight + f32/f16 bias |
| [S,H,D] → [H,S,D] | `Permute` (0,2,1,3) |
| RoPE | `RoPE`, `interleaved = false`, rotary_dim = head_dim, position_ids input |
| cache append | caller-owned q8 quantization (below) |
| attention | `GroupQueryAttention`, `causal = true`, s8 cache + f32 scales |
| output projection | `Permute` back, `Linear` q4_1 |
| residual | `Eltwise` Add |
| FFN | `Norm`, `Linear` gate (Silu epilogue), `Linear` up, `Eltwise` Mul, `Linear` down |
| residual | `Eltwise` Add |
| final norm + LM head | `Norm` on the last row, `Linear` q4_1 (tied table quantized) |

`Linear`'s fused gated form only accepts matching fp operands, so the SwiGLU
combine is an explicit Mul over two q4_1 projections. Everything else is one
operator call per step.

## Weights

`make_fake_weights` draws uniform values in ±0.02 (Qwen's `initializer_range`)
and quantizes each 32-value block with `quantize_q4` — the same encoder
`test_q4_1.cpp` checks bit for bit against llama.cpp. No dense copy of a weight
matrix is kept: the q4_1 blocks are the storage, the embedding table included
(the lookup decodes one row at a time). Only the norm weights and the q/k/v
biases are dense, in the activation dtype. A tied `lm_head` is the embedding
table itself.

Presets (`llm/qwen2_config.hpp`):

| preset | layers | hidden | heads / kv | head_dim | FFN | notes |
|---|---|---|---|---|---|---|
| `tiny` | 4 | 256 | 4 / 2 | 64 | 1024 | fast self-test size |
| `qwen2.5-0.5b` | 24 | 896 | 14 / 2 | 64 | 4864 | matches the local checkpoint config |
| `qwen2.5-4b` | 40 | 2560 | 20 / 4 | 128 | 9728 | approximate: no official Qwen2.5-4B checkpoint; llama.cpp classifies 40 layers / 20 heads as 4B |

## KV cache

The cache is continuous, not paged: per layer one `[H_kv, capacity, D]` s8
plane and one `[H_kv, capacity]` f32 scale plane, allocated up front. A step
quantizes its new K/V rows symmetrically — `scale = amax / 127` per
(head, token) — writes them at `length`, and passes the whole planes to
`GroupQueryAttention` with `kv_seqlens = length` and `position_ids = length`.
The operator derives the causal row range from those two values; rows past the
valid prefix are never read. Prefill and decode are the same `run` call with a
different token count.

## Benchmark

`--bench` runs the requested prefill lengths and then one decoded token with
that context already in the q8 cache. Each row is the best of 5 prefill runs
and 9 decode steps after a warm-up (scratch allocation and cold pages are
outside the timed path). The host schedules a 4-thread OpenMP region across a
6P+8E core mix, so single runs drift by 2-3x; the minimum is the stable
estimate of the kernel's own cost.

```sh
./build/llm/Release/nnops_qwen2_llm --preset qwen2.5-0.5b --dtype f32 \
    --bench --threads 4
```

qwen2.5-0.5b on an i5-13600KF (6P+8E), 2026-11-28. Four OpenMP workers:

| seq | f32 prefill ms | f32 tok/s | f32 decode ms/token | f16 prefill ms | f16 tok/s | f16 decode ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 152.1 | 420.9 | 12.62 | 243.8 | 262.6 | 11.00 |
| 128 | 247.9 | 516.3 | 10.91 | 409.6 | 312.5 | 11.13 |
| 256 | 457.1 | 560.0 | 10.61 | 781.2 | 327.7 | 13.70 |
| 384 | 657.5 | 584.0 | 10.69 | 1148.5 | 334.4 | 13.41 |
| 512 | 881.5 | 580.8 | 12.31 | 1534.6 | 333.6 | 11.37 |

Prefill throughput rises with the block because the q4_1 GEMM stays the same
per token while the fixed per-call cost amortizes; decode is ~8-13 ms a token
and nearly flat in context length here because 512 rows of a 2-head, 64-wide
q8 cache are still small beside the 0.5B weights.

Six workers — the P-core count — is the best configuration on this host:

| seq | f32 prefill ms | f32 tok/s | f32 decode ms/token | f16 prefill ms | f16 tok/s | f16 decode ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 102.9 | 621.9 | 7.76 | 177.9 | 359.7 | 8.27 |
| 128 | 177.8 | 719.9 | 7.73 | 292.8 | 437.2 | 8.21 |
| 256 | 308.3 | 830.5 | 7.79 | 557.6 | 459.2 | 8.29 |
| 384 | 447.4 | 858.3 | 7.94 | 828.4 | 463.6 | 8.41 |
| 512 | 604.2 | 847.5 | 8.14 | 1110.3 | 461.2 | 8.46 |

4T -> 6T is 1.47x overall, but 1.59x on the FFN gate and only 1.32-1.40x on
qkv/down: the narrow-N projections are the ones that stop scaling. Twenty
workers is *slower* than six (682.6 ms at 512, decode 10.1 ms) — the static
block split hands whole n-blocks to E-core workers, and a block is the
scheduling quantum.

### Headroom

Per-phase throughput at 6 threads (2 FLOP per MAC):

| phase | GFLOP | ms | GFLOPS |
| --- | ---: | ---: | ---: |
| ffn gate | 107.1 | 144.4 | 742 |
| ffn up | 107.1 | 142.1 | 754 |
| ffn down | 107.1 | 185.0 | 579 |
| qkv | 25.4 | 45.1 | 563 |
| o_proj | 19.7 | 32.9 | 600 |
| whole prefill | 366 | 602.8 | 607 |

Measured against the rest of the operator set, in order of what it is worth:

1. **The f16 activation route runs 1.84x f32** (1110 vs 603 ms at 512), and
   every phase pays it: gate 295.9 vs 144.4, qkv 76.1 vs 45.1, attention 49.8
   vs 26.8. This is not bandwidth. `f16_a_bcast`
   (`src/backend/cpu/x86_64/mma_f16.hpp`) converts one half per (row, k) with
   `vcvtph2ps`, and a per-element convert costs ~2.3 FMA slots on this host —
   the A broadcast doubles the inner loop. Converting a k-run of eight halves
   once and broadcasting f32 lanes should land f16 at f32's rate; the same
   shape appears in attention's f16 query over an s8 cache.
2. **The narrow-N projections leave 20-30% on the table.** down (N=896), qkv
   (N=1152) and o_proj (N=896) reach 563-600 GFLOPS where the 4864-wide gate
   reaches 742-754. A 512-row down_proj is ten n-blocks over six workers — a
   two-wave split — and each kernel call streams six B panels per A load
   instead of nine. Split-K (the reduction is a 2 MB f32 pass) or an nc that
   yields an even two blocks per worker would recover part of it: ~5-8% of a
   prefill.
3. **Decode is DRAM-bound at 36 GB/s** — 309 MB of q4_1 weights a token in
   8.6 ms, against DDR5-4800's 76.8 GB/s — and flat: 4T -> 6T is 1.44x, 20T is
   worse. The 20-byte block straddles cache lines, but the prefetchers already
   cover it; the remaining lever is weight layout, not compute.
4. **FFN fusion does not pay.** Sending gate+up as one [2I, H] q4_1 weight
   through `Linear`'s gated form (after letting its validation accept a
   block-packed weight) measured 614.8 ms vs 599.8 at 512, 6T — 2.5% slower in
   three interleaved A/B pairs, decode unchanged. Nothing redundant is
   removed: the same MACs and the same weight stream, plus an [S, 2I] fp
   temporary and a combine pass. Folding silu*up into down's A pack would save
   one [S, I] write and read, ~2% of a prefill, for a new operator.

### Where the time goes

`--profile` times every operator phase (3 prefill / 9 decode runs at context
512). f32, 4 threads, qwen2.5-0.5b:

| phase | prefill 512 ms | share | decode ms | share |
| --- | ---: | ---: | ---: | ---: |
| embed | 0.33 | 0.0% | 0.00 | 0.0% |
| norm | 2.21 | 0.2% | 0.04 | 0.4% |
| qkv | 61.04 | 6.9% | 0.74 | 6.3% |
| permute | 2.82 | 0.3% | 0.08 | 0.6% |
| rope | 3.14 | 0.4% | 0.13 | 1.1% |
| cache q8 | 7.47 | 0.8% | 0.02 | 0.1% |
| attention | 34.18 | 3.8% | 0.55 | 4.7% |
| o_proj | 46.50 | 5.2% | 0.47 | 4.0% |
| residual | 2.72 | 0.3% | 0.04 | 0.4% |
| ffn gate | 230.40 | 25.9% | 2.23 | 18.9% |
| ffn up | 223.39 | 25.1% | 2.29 | 19.4% |
| ffn mul | 10.70 | 1.2% | 0.05 | 0.4% |
| ffn down | 260.63 | 29.3% | 2.42 | 20.6% |
| lm_head | 3.07 | 0.3% | 2.72 | 23.1% |
| **total** | **888.60** | 100% | **11.77** | 100% |

- **The FFN q4_1 GEMMs are the model.** gate + up + down are 80% of prefill
  and 59% of decode; doubling the FFN width or improving the q4_1 GEMM's M=512
  throughput is the only lever that moves the total materially.
- **Decode is weight-stream bound, not compute bound.** One token reads every
  q4_1 weight once: 224 MB of layer matrices (196 MB of that is FFN) plus the
  85 MB tied lm_head, ~309 MB in 11.8 ms ≈ 26 GB/s at four threads (8.6 ms ≈
  36 GB/s at six). That is why extra workers stop helping past the P cores and
  why f16 decode is within noise of f32.
- **The lm_head is 23% of a decode step** at the 151936 vocabulary — a single
  GEMV over the whole embedding table. Prefill amortizes it across the block
  (0.3%), and only the last row is computed in either case.
- **Attention is 4-5% at 512** (2 KV heads × 64 lanes is thin); it grows
  quadratically and would overtake the projections somewhere past 4k context.
- **f16 prefill is 1.7x f32 with the same shares.** The q4_1 kernel decodes
  weights into f32 registers either way, so f16 only adds F16C converts and
  narrowing on the activation path.

## Self-test

`--selftest` replays the prompt one token at a time after the prefill and
compares the last-row logits. It checks that the prefill and decode attention
routes, the RoPE positions and the q8 cache agree:

```
selftest  stepwise 423.6 ms; max |diff| 6.693e-04 (4.179e-04 of max 1.601e+00)
          greedy 41937 vs 41937 -> PASS
```

Measured on the qwen2.5-0.5b preset (16-token prompt, i5-13600KF):
f32 4.2e-4 relative, f16 5.2e-3 (q8 cache + f16 rounding).

## Limits

- CPU backend only; the operators are created with `Backend::CPU`.
- Greedy decoding; no sampler, no tokenizer, no chat template.
- Weights are random. Logits and generated ids are deterministic for a seed
  but meaningless.
- `qwen2.5-4b` is a shape preset, not a checkpoint config.
