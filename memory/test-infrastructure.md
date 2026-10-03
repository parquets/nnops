---
name: test-infrastructure
description: "Test framework design, random data generation, and benchmark approach for nnops"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-22T16:20:59.559Z
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

Each `tests/test_*.cpp` builds into its own standalone demo executable (one per file, each linked
against `common/test_harness.cpp`, which supplies `main()`). Run every demo with:

```bash
scripts/run_tests.sh Release      # prints [PASS]/[FAIL] per demo + final tally; exit 0 iff all pass
```

Demos live at `build/tests/<Config>/test_*.exe` (Visual Studio generator) and are also registered
with CTest, so `ctest --test-dir build -C Release --output-on-failure` works too. Benchmarks follow
the same one-executable-per-file layout — each `tests/bench_*.cpp` builds into its own
`bench_<family>` demo (linked against `common/bench_harness.cpp`, which supplies `main()`), gated by
`NNOPS_BUILD_BENCHMARKS`. They are not registered with CTest, since benchmarks are not pass/fail.

## Benchmark Pattern

When adding benchmarks, use:
```cpp
#include "common/bench_harness.hpp"
auto result = test::benchmark("op_name", 100, [&]() { op->compute(...); });
test::report_bench(result);
```
