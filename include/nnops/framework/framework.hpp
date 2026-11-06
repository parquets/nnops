#pragma once
/// @file framework.hpp
/// @brief Umbrella include for the ONNX → execution framework.
///
///   Model model;
///   if (is_error(model.load_from_file("net.onnx"))) { ... }
///
///   Session session(model);                        // lower once
///   std::vector<Tensor> outputs;
///   session.run({&input, 1}, outputs);             // plan and run
///
/// Model owns the parsed graph, Session owns the buffers. The model must
/// outlive every session built from it.

#include "nnops/framework/graph.hpp"
#include "nnops/framework/model.hpp"
#include "nnops/framework/session.hpp"
#include "nnops/framework/tensor.hpp"
