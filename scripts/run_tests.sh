#!/usr/bin/env bash
# run_tests.sh — run every standalone test demo and aggregate pass/fail.
#
# Usage:
#   scripts/run_tests.sh [config]     # config defaults to Release
#
# Each tests/test_*.cpp builds into its own executable under
#   <build>/tests/<config>/test_*.exe   (Windows)  or  <build>/tests/test_*  (POSIX).
# A demo's main() returns its failed-test count, so a non-zero exit means failure.
#
# The build directory is auto-detected as <repo-root>/build; override with
# NNOPS_BUILD_DIR if you configured CMake elsewhere.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-Release}"
BUILD_DIR="${NNOPS_BUILD_DIR:-$ROOT/build}"
TEST_DIR="$BUILD_DIR/tests/$CONFIG"

shopt -s nullglob

case "$OSTYPE" in
    msys*|cygwin*|win32*)
        exes=("$TEST_DIR"/test_*.exe)
        ;;
    *)
        exes=("$TEST_DIR"/test_*)
        ;;
esac

if ((${#exes[@]} == 0)); then
    echo "No test demos found in: $TEST_DIR" >&2
    echo "Build them first, e.g.:  cmake --build \"$BUILD_DIR\" --config $CONFIG" >&2
    exit 2
fi

pass=0
fail=0
failed_names=()

for exe in "${exes[@]}"; do
    name="$(basename "$exe" .exe)"
    # Capture output so it can be replayed only on failure.
    if out="$("$exe" 2>&1)"; then
        echo "[PASS] $name"
        ((pass++))
    else
        echo "[FAIL] $name"
        ((fail++))
        failed_names+=("$name")
        printf '%s\n' "$out" | sed 's/^/        /'
    fi
done

echo
echo "=================================================="
echo "$pass passed, $fail failed"
if ((fail > 0)); then
    printf 'Failed: %s\n' "${failed_names[*]}"
    exit 1
fi
exit 0
