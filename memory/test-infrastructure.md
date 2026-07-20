---
name: test-infrastructure
description: "Test framework design, random data generation, and benchmark approach for nnops"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T15:07:23.561Z
---

# Test Infrastructure

Self-contained testing with zero external dependencies.

**Why:** The project must have no third-party deps, so we cannot use Google Test, Catch2, or similar. All test utilities are hand-rolled.

**How to apply:** Every new operator needs tests covering: (1) hand-verified small shapes, (2) random data sanity checks, (3) class API vs functional API parity.

## Components

- `tests/common/test_harness.hpp` — `NNOPS_TEST(name)` macro + `NNOPS_EXPECT_EQ` / `NNOPS_EXPECT_NEAR` / `NNOPS_EXPECT_TRUE`
- `tests/common/random_tensor.hpp` — `XorShift128` PRNG + `make_random_tensor(shape)` → `(vector<float>, TensorView)`
- `tests/common/compare.hpp` — `allclose(a, b, rtol, atol)` for tolerance-based tensor comparison
- `tests/common/bench_harness.hpp` — `benchmark()` and `report_bench()` for std::chrono-based timing

## Test Organization

Each operator has a corresponding `tests/test_<op>.cpp` with tests in three categories:
1. **Hand-verified**: Small fixed inputs with analytically computed expected outputs
2. **Random sanity**: Larger random tensors — verify no NaN/Inf, outputs are in valid range
3. **API parity**: Run same operation via both class API and functional API, compare results with `allclose()`

## Running Tests

```bash
cd build && cmake --build . --config Release --target nnops_test && ctest -C Release
```

Tests are registered via CTest. Currently 44 tests across 7 test files.

## Benchmark Pattern

When adding benchmarks, use:
```cpp
#include "common/bench_harness.hpp"
auto result = test::benchmark("op_name", 100, [&]() { op->compute(...); });
test::report_bench(result);
```
