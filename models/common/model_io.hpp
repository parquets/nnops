#pragma once
/// @file model_io.hpp
/// @brief Read a model artifact exported by models/export/export_model.py, and
///        the activation-buffer plumbing the model runners share.
///
/// An artifact is a directory holding two files:
///   manifest.txt   "# ..." comments, then one line per tensor:
///                    <name> <dtype> <offset> <nbytes> <dim0> <dim1> ...
///   data.bin       the tensors' raw little-endian bytes, concatenated
///
/// All tensors are f32 today; the dtype column exists so the format can grow
/// without a version bump.

#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace model {

namespace detail {

/// The activation free list: one plain `vector<float>` per distinct element
/// count. A runner allocates the same handful of shapes on every forward pass,
/// so from the second pass on every activation handed out is the block the
/// previous pass returned — no `malloc`, no first-touch page faults, and no
/// zero-fill (`vector` zeroes on construction, not on reuse).
///
/// Keyed on the exact word count, not a size class: `Buf::data.size()` is what
/// the BN fold divides by the output-channel count, and a padded block would
/// silently change that. The pool holds ~6 blocks for these networks.
///
/// Never destroyed: the weight-fold and Winograd-U caches are function-local
/// statics in net_ops.hpp that destroy their `Buf`s at exit, and a `Buf` gives
/// its block back here. This static is still alive then; the alternative is a
/// destruction-order dependency between two translation-unit-local caches.
/// Deliberately single-threaded — a runner drives its network from one thread.
inline std::vector<std::vector<float>>& block_pool()
{
    static auto* pool = new std::vector<std::vector<float>>();
    return *pool;
}

inline std::vector<float> take_block(size_t words)
{
    auto& pool = block_pool();
    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->size() == words) {
            std::vector<float> block = std::move(*it);
            pool.erase(it);
            return block;
        }
    }
    return std::vector<float>(words);
}

inline void give_block(std::vector<float> block)
{
    if (block.empty()) { return; }
    block_pool().push_back(std::move(block));
}

}  // namespace detail

/// A forward-pass activation: the storage and, over it, the view an operator
/// reads and writes.
///
/// Non-copyable on purpose — the view points into `data`, so a copy would leave
/// the view dangling on the original buffer. Moving is safe: a vector move
/// keeps its heap block, so the pointer stays valid.
///
/// A dying `Buf` hands its block back to the free list, and a move-assignment
/// releases the target's block the same way; that is what makes the steady
/// state allocation-free. The contents are then left over from the previous
/// pass, which is harmless because every operator in these networks writes its
/// whole output — and it makes a kernel that does not stand out as a mismatch
/// instead of hiding behind a zero.
struct Buf {
    std::vector<float> data;
    nnops::TensorView view;

    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& other) noexcept : data(std::move(other.data)), view(other.view)
    {
        other.view = nnops::TensorView{};
    }
    Buf& operator=(Buf&& other) noexcept
    {
        if (this != &other) {
            detail::give_block(std::move(data));
            data = std::move(other.data);
            view = other.view;
            other.view = nnops::TensorView{};
        }
        return *this;
    }
    ~Buf() { detail::give_block(std::move(data)); }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

/// Allocate a buffer described by @p desc and a view over it. A recycled block
/// when one of this exact size is on the free list; a zeroed one otherwise.
///
/// A packed layout (NCHWC8) pads every row to the descriptor's pitch, so the
/// view has to be built with that pitch and the storage with that many words —
/// the dense constructor would describe a tighter tensor than the kernel writes.
inline Buf alloc(const nnops::TensorDesc& desc)
{
    Buf buf;
    std::vector<int64_t> shape(desc.dims.begin(),
                               desc.dims.begin() + desc.rank);
    const bool packed = nnops::layout_channel_pack(desc.layout) > 1;
    const size_t words = packed
        ? (desc.storage_bytes() + sizeof(float) - 1) / sizeof(float)
        : static_cast<size_t>(desc.numel());
    buf.data = detail::take_block(words);
    if (packed) {
        buf.view = nnops::TensorView(shape, desc.dtype, buf.data.data(),
                                     desc.row_pitch(), desc.layout);
    } else {
        buf.view = nnops::TensorView(shape, desc.dtype, buf.data.data(),
                                     desc.layout);
    }
    return buf;
}

/// Abort with the failing status — a model runner treats a rejected op as a
/// hard error and prints where it happened.
inline void require_ok(nnops::Status st, const char* what)
{
    if (nnops::is_ok(st)) { return; }
    std::fprintf(stderr, "model: %s failed: %s\n", what,
                 nnops::to_string(st));
    std::abort();
}

class ModelArtifact {
public:
    /// Load @p dir into memory. Returns false (with a message in @p error) when
    /// the artifact is absent or malformed — the caller turns that into a skip.
    bool load(const std::string& dir, std::string& error)
    {
        const std::string manifest_path = dir + "/manifest.txt";
        const std::string data_path = dir + "/data.bin";

        std::ifstream manifest(manifest_path);
        if (!manifest) {
            error = "no manifest at " + manifest_path +
                    " (export it: see models/README.md)";
            return false;
        }

        std::vector<std::string> names;
        std::vector<int64_t> offsets, sizes;
        std::vector<std::vector<int64_t>> shapes;

        std::string line;
        while (std::getline(manifest, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream in(line);
            std::string name, dtype;
            int64_t offset = 0, nbytes = 0;
            in >> name >> dtype >> offset >> nbytes;
            if (!in || dtype != "f32") {
                error = "malformed manifest line: " + line;
                return false;
            }
            std::vector<int64_t> dims;
            int64_t d = 0;
            while (in >> d) { dims.push_back(d); }
            names.push_back(name);
            offsets.push_back(offset);
            sizes.push_back(nbytes);
            shapes.push_back(std::move(dims));
        }

        std::ifstream data(data_path, std::ios::binary | std::ios::ate);
        if (!data) {
            error = "no data at " + data_path;
            return false;
        }
        const auto total = static_cast<size_t>(data.tellg());
        data.seekg(0);
        // Stored as float words so the views are aligned for f32 access.
        data_.resize((total + sizeof(float) - 1) / sizeof(float));
        data.read(reinterpret_cast<char*>(data_.data()),
                  static_cast<std::streamsize>(total));

        for (size_t i = 0; i < names.size(); ++i) {
            if (offsets[i] % sizeof(float) != 0 ||
                offsets[i] + sizes[i] > static_cast<int64_t>(total)) {
                error = "tensor out of bounds: " + names[i];
                return false;
            }
            const size_t word = static_cast<size_t>(offsets[i] / sizeof(float));
            tensors_.emplace(names[i],
                             nnops::TensorView(shapes[i], nnops::DataType::f32,
                                               data_.data() + word));
        }
        return true;
    }

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }

    /// The tensor named @p name. Aborts when absent: a missing tensor means the
    /// Python and C++ descriptions of the network drifted apart.
    nnops::TensorView at(const std::string& name) const
    {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            std::fprintf(stderr, "model: no tensor '%s' in the artifact\n",
                         name.c_str());
            std::abort();
        }
        return it->second;
    }

    size_t count() const { return tensors_.size(); }

private:
    std::vector<float> data_;
    std::map<std::string, nnops::TensorView> tensors_;
};

}  // namespace model
