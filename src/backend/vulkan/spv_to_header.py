"""Convert SPIR-V binary to a C header with uint32_t array.

Usage:
    python spv_to_header.py --name g_eltwise_f32 --spv eltwise_f32.spv --out eltwise_f32_spv.h
"""
import argparse, os, sys

def spv_to_c(name, spv_path, h_path):
    with open(spv_path, 'rb') as f:
        data = f.read()
    words = []
    for i in range(0, len(data), 4):
        w = int.from_bytes(data[i:i+4], 'little')
        words.append(str(w))

    with open(h_path, 'w') as f:
        f.write(f'// Auto-generated from {os.path.basename(spv_path)} — do not edit.\n')
        f.write('#pragma once\n')
        f.write('#include <cstdint>\n')
        f.write('namespace nnops::backend::vulkan {\n')
        f.write(f'constexpr uint32_t {name}_spv[] = {{\n')
        for i in range(0, len(words), 8):
            f.write('    ' + ', '.join(words[i:i+8]) + ',\n')
        f.write('};\n')
        f.write(f'constexpr size_t {name}_spv_len = sizeof({name}_spv);\n')
        f.write('}  // namespace nnops::backend::vulkan\n')
    print(f'Generated {h_path}: {len(words)} words ({len(data)} bytes)')

def main():
    parser = argparse.ArgumentParser(description='Convert SPIR-V binary to C header')
    parser.add_argument('--name', required=True, help='C array name prefix')
    parser.add_argument('--spv', required=True, help='Input SPIR-V binary file')
    parser.add_argument('--out', required=True, help='Output C header file')
    args = parser.parse_args()
    spv_to_c(args.name, args.spv, args.out)

if __name__ == '__main__':
    main()
