#!/usr/bin/env bash
set -euo pipefail

# Build the sample Lean module to ensure mathlib and the toolchain are working.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${HERE%/scripts}"

cd "$ROOT"

echo "[lean] Building Autoformalizer.Basic via lake build"
lake build Autoformalizer.Basic

echo "[lean] Building default Lake targets"
lake build
