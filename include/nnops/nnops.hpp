#pragma once
/// @file nnops.hpp
/// @brief Top-level convenience include for the nnops library.

// Core types
#include "nnops/core/data_type.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/op_type.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/op_base.hpp"

// Operators
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/activation.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/attention.hpp"
