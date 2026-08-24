#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out="${1:-$repo_dir/tools/cb_core_bench}"

cc="${CC:-cc}"
"$cc" -O3 -std=c99 -DCB_CORE_BENCH \
    "$repo_dir/src/c/cb_core.c" \
    -o "$out" -lm

printf '%s\n' "$out"
