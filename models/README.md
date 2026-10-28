# End-to-end model tests

These are whole networks, not operator tests: `resnet18.cpp` and
`mobilenetv2.cpp` rebuild the torchvision graph out of nnops operators and
compare **every layer** against the activations PyTorch recorded for the same
input. A mismatch names the layer that drifted, so a precision or dispatch bug
is localized instead of showing up as "the logits are off".

The C++ side needs no Python at test time. Everything — weights, input, and
reference activations — comes from an artifact directory exported once by
`export/export_model.py`:

```
models/data/<model>/manifest.txt    name dtype offset nbytes dims...
models/data/<model>/data.bin        the tensors' raw f32 bytes, concatenated
```

`models/data/` is generated and git-ignored; regenerate it (or delete it and
watch the tests skip).

## Exporting

Needs a Python environment with torch and torchvision — this repo was developed
against the conda env `pat_pytorch` (torch 2.1 / torchvision 0.16):

```bash
conda run -n pat_pytorch python models/export/export_model.py --model resnet18
conda run -n pat_pytorch python models/export/export_model.py --model mobilenetv2
```

Weights come from torchvision's download cache; `--random-weights` skips the
download and uses a seeded initialization instead (the comparison stays
meaningful, the logits just are not ImageNet). `--input-seed` picks another
input; `--out` writes somewhere other than `models/data/<model>`.

The input is seeded N(0,1) noise rather than a real image: the test measures
numerical agreement with PyTorch, not accuracy, and noise exercises every
channel without dragging an image decoder into the build.

## Running

```bash
cmake --build build --config Release --target model_resnet18 model_mobilenetv2
./build/models/Release/model_resnet18.exe
./build/models/Release/model_mobilenetv2.exe
```

or through CTest (`ctest -R model_`). Both are registered as tests and return
**77** when the artifact is missing, which CTest reports as skipped, so a
checkout without the exported data still has a green test run.

Arguments: `model_resnet18 [artifact-dir] [--atol X] [--rtol Y]`.

After the checkpoint table each runner prints a profile of the timed pass
(`common/model_prof.hpp`): wall clock **per layer** (the checkpoint names, so a
slow one is named the way a failure would be) and **per operator**, with the
convolution rows split by route (`conv2d/wino` vs `conv2d/im2col`) so the two
can be priced against each other. Both tables are off during the untimed
warm-up, and the layer table's `(other)` row is what the layers do not
explain — the per-checkpoint comparison and the frees.

## What is compared

| model | checkpoints |
| --- | --- |
| ResNet-18 | `conv1`, `bn1`, `relu`, `maxpool`, `layer{1..4}.{0,1}`, `avgpool`, `logits` |
| MobileNetV2 | `features.0`, `features.1`..`features.17` (one per inverted-residual block), `features.18`, `logits` |

The tolerance is the usual scaled one, element by element:

```
|actual - expected| <= atol + rtol * |expected|
```

Each row prints `max|d|`, the same error relative to that tensor's peak
magnitude, and how many elements fell outside. Defaults are `atol = rtol =
1e-3`.

Measured on this machine (MSVC Release, `/arch:AVX2`), the worst checkpoint is
`1.4e-5` absolute for ResNet-18 (`layer4.1`) and `4.8e-5` for MobileNetV2
(`features.18`), with relative errors below `1e-5` — pure f32 accumulation-order
difference against PyTorch. A real bug lands orders of magnitude above the
default (a misplaced expansion ratio in an early block was `1e+2`), so the
tolerance is loose enough to survive a different SIMD path and still catch
anything structural.

## Layout notes

The runners stay in NCHW, which is what PyTorch records, so nothing is
converted on the way in:

- **Global average pooling** is the pooling operator's `global` mode
  (`attrs.global`, the `AdaptiveAvgPool2d(1)` / `GlobalAveragePool` form): the
  window is the whole plane and every spatial axis of the output is 1, so the
  kernel scans each channel plane instead of walking a window.

Pooling and depthwise convolution take NCHW directly, so `max_pool()` and
`depthwise_conv()` in `common/net_ops.hpp` hand them the tensor (and, for the
depthwise stage, the exported `[C, 1, 3, 3]` weight) with no repacking.

Convolutions take the Winograd route where `winograd_applicable` says it pays —
3×3, stride 1, at least 32 input channels and a 14×14 input — with the weight
transformed (`common/net_ops.hpp`'s `winograd_weight`) and cached on first use;
everything else stays on im2col. Both runners do one untimed pass first, so the
weight transform and the first-touch allocation costs are paid before the clock
starts and the reported `forward` is the steady state.

BatchNorm statistics are folded from the exported `running_mean` /
`running_var`, so the networks run in eval mode exactly as exported.

## Adding a model

1. Add a builder + checkpoint list to `MODELS` in
   `export/export_model.py` (the checkpoint list names the modules whose
   outputs the C++ runner will also produce).
2. Add the runner to `MODEL_SOURCES` in `CMakeLists.txt`, rewriting the graph
   with the helpers in `common/net_ops.hpp` and calling
   `check.compare("ref-name", ...)` at each checkpoint.
3. Export, build, run. A missing weight name aborts with the tensor name —
   that is the signal that the Python and C++ descriptions drifted apart.
