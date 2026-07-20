#pragma once
/// @file test_harness.hpp
/// @brief Minimal self-contained test framework for nnops.
///
/// No third-party dependencies. Provides:
///   - NNOPS_TEST(name)          — register a test
///   - NNOPS_EXPECT_EQ(a, b)      — equality assertion
///   - NNOPS_EXPECT_NEAR(a, b, tol) — approximate float equality
///   - NNOPS_EXPECT_TRUE(cond)     — boolean assertion

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnops::test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void add(const std::string& name, std::function<void()> fn) {
        tests_.push_back({name, std::move(fn)});
    }

    int run_all() {
        int passed = 0, failed = 0;
        for (auto& t : tests_) {
            std::cout << "  RUN    " << t.name << std::endl;
            try {
                t.fn();
                std::cout << "  PASSED " << t.name << std::endl;
                ++passed;
            } catch (const std::exception& e) {
                std::cout << "  FAILED " << t.name << ": " << e.what() << std::endl;
                ++failed;
            } catch (...) {
                std::cout << "  FAILED " << t.name << ": unknown exception" << std::endl;
                ++failed;
            }
        }
        std::cout << "\n" << passed << " passed, " << failed << " failed" << std::endl;
        return failed;
    }

private:
    std::vector<TestCase> tests_;
};

}  // namespace nnops::test

// ---- Test registration macro ----
#define NNOPS_TEST(name) \
    static void _nnops_test_##name(); \
    static struct _nnops_reg_##name { \
        _nnops_reg_##name() { \
            nnops::test::TestRegistry::instance().add(#name, _nnops_test_##name); \
        } \
    } _nnops_reg_inst_##name; \
    static void _nnops_test_##name()

// ---- Assertion macros ----
#define NNOPS_EXPECT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) throw std::runtime_error( \
        std::string(__FILE__ ":") + std::to_string(__LINE__) + \
        ": EXPECT_EQ failed"); \
} while(0)

#define NNOPS_EXPECT_NEAR(a, b, tol) do { \
    auto _a = (a); auto _b = (b); \
    if (std::abs(_a - _b) > (tol)) throw std::runtime_error( \
        std::string(__FILE__ ":") + std::to_string(__LINE__) + \
        ": EXPECT_NEAR failed: |" + std::to_string(_a) + " - " + \
        std::to_string(_b) + "| > " + std::to_string(tol)); \
} while(0)

#define NNOPS_EXPECT_TRUE(cond) do { \
    if (!(cond)) throw std::runtime_error( \
        std::string(__FILE__ ":") + std::to_string(__LINE__) + \
        ": EXPECT_TRUE failed: " #cond); \
} while(0)

#define NNOPS_EXPECT_FALSE(cond) do { \
    if (cond) throw std::runtime_error( \
        std::string(__FILE__ ":") + std::to_string(__LINE__) + \
        ": EXPECT_FALSE failed: " #cond); \
} while(0)
