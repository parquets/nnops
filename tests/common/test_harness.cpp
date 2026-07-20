/// @file test_harness.cpp
/// @brief Test runner — main() entry point.

#include "test_harness.hpp"

int main() {
    std::cout << "=== nnops tests ===" << std::endl;
    return nnops::test::TestRegistry::instance().run_all();
}
