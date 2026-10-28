"""Export a torchvision classifier to an nnops model artifact.

The artifact is what models/resnet18.cpp and models/mobilenetv2.cpp consume:

    <out>/manifest.txt   one line per tensor:  name dtype offset nbytes d0 d1 ...
    <out>/data.bin       raw little-endian tensor data, concatenated

Everything - the weights, the input, and the reference activations that the C++
runner compares against - lives in those two files, so the C++ side needs no
Python, no numpy, and no network at test time.

The input is a fixed seed's N(0,1) tensor rather than a real image: the test
measures numerical agreement with PyTorch, not accuracy, and a generated input
keeps the artifact free of any JPEG/decoder dependency.

Usage (from the repo root, with the conda env that has torch):

    conda run -n pat_pytorch python models/export/export_model.py --model resnet18
    conda run -n pat_pytorch python models/export/export_model.py --model mobilenetv2

Weights come from torchvision (its download cache is used when present). Pass
--random-weights to skip the download and use a seeded random initialization
instead - the comparison is still meaningful, the logits just are not ImageNet.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torchvision import models


# Per model: the builder, its pretrained weights enum, and the modules whose
# outputs the C++ runner also produces (as "ref.<module name>").
def _resnet18_checkpoints(net):
    names = ["conv1", "bn1", "relu", "maxpool"]
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        names += [f"{layer}.{i}" for i in range(len(getattr(net, layer)))]
    names.append("avgpool")
    return names


def _mobilenetv2_checkpoints(net):
    # features = [stem, InvertedResidual x17, final 1x1 conv]; the head after
    # that is the classifier, whose output is ref.logits already.
    names = ["features.0", "features.1"]
    names += [n for n, m in net.named_modules()
              if isinstance(m, models.mobilenetv2.InvertedResidual)]
    names += ["features.18"]
    return names


MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT,
                 _resnet18_checkpoints),
    "mobilenetv2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT,
                    _mobilenetv2_checkpoints),
}


def build(name, random_weights):
    builder, weights_enum, checkpoints = MODELS[name]
    if random_weights:
        torch.manual_seed(0)
        net = builder(weights=None)
    else:
        net = builder(weights=weights_enum)
    net.eval()
    return net, checkpoints(net)


def capture(net, input_tensor, module_names):
    """Run the model once, recording the output of each named module."""
    captured = {}
    handles = []

    def hook(module_name):
        def fn(_module, _inputs, output):
            # clone(): torchvision's ReLU/ReLU6 are in-place, so the next
            # module overwrites this buffer; without the copy a checkpoint
            # records the state at the end of the forward pass, not here.
            captured[module_name] = output.detach().cpu().clone().numpy()
        return fn

    for module_name in module_names:
        module = net.get_submodule(module_name)
        handles.append(module.register_forward_hook(hook(module_name)))
    with torch.no_grad():
        logits = net(input_tensor).cpu().numpy()
    for handle in handles:
        handle.remove()
    return logits, captured


def write_artifact(out_dir, tensors):
    """Write manifest.txt + data.bin for an ordered dict name -> np.ndarray."""
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# nnops model artifact v1",
             "# name dtype offset nbytes dims..."]
    offset = 0
    with open(os.path.join(out_dir, "data.bin"), "wb") as data:
        for name, array in tensors.items():
            array = np.ascontiguousarray(array, dtype=np.float32)
            dims = " ".join(str(d) for d in array.shape)
            line = f"{name} f32 {offset} {array.nbytes}"
            if dims:
                line += f" {dims}"
            lines.append(line)
            data.write(array.tobytes())
            offset += array.nbytes
    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="ascii",
              newline="\n") as manifest:
        manifest.write("\n".join(lines) + "\n")
    return offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--out", default=None,
                        help="output directory (default models/data/<model>)")
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--random-weights", action="store_true",
                        help="no download; seeded random initialization")
    args = parser.parse_args()

    net, checkpoints = build(args.model, args.random_weights)

    rng = np.random.default_rng(args.input_seed)
    x = torch.from_numpy(rng.standard_normal((1, 3, 224, 224), dtype=np.float32))

    logits, captured = capture(net, x, checkpoints)

    tensors = {"input": x.numpy()}
    for name, value in net.state_dict().items():
        # num_batches_tracked is a scalar counter, not something nnops reads.
        if value.is_floating_point():
            tensors[name] = value.detach().cpu().numpy()
    for name, value in captured.items():
        tensors["ref." + name] = value
    tensors["ref.logits"] = logits

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    out_dir = args.out or os.path.join(repo_root, "models", "data", args.model)
    nbytes = write_artifact(out_dir, tensors)

    top5 = np.argsort(logits[0])[::-1][:5]
    print(f"{args.model}: {len(tensors)} tensors, {nbytes / 1e6:.1f} MB -> {out_dir}")
    print("  input", tuple(x.shape), " logits", tuple(logits.shape))
    print("  top-5 classes", top5.tolist(),
          " probs", np.round(np.exp(logits[0][top5]) / np.exp(logits[0]).sum(), 3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
