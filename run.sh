#!/usr/bin/env bash
# Run from anywhere: bash /path/to/this/repo/run.sh python|r|all
set -euo pipefail
cd -- "$(dirname -- "$0")"
language="${1:-all}"
case "$language" in
  python|r) languages=("$language") ;;
  all) languages=(python r) ;;
  *) printf 'Usage: bash run.sh [python|r|all]\n' >&2; exit 2 ;;
esac
for input in train.csv test.csv; do
  if [[ ! -f "src/data/$input" ]]; then
    printf 'Missing src/data/%s. Follow the Kaggle download steps in README.md.\n' "$input" >&2
    exit 1
  fi
done
if ! docker info >/dev/null 2>&1; then
  printf 'Docker is not running. Start Docker Desktop or your Docker engine, then retry.\n' >&2
  exit 1
fi
for language in "${languages[@]}"; do
  output="$(pwd)/outputs/$language"
  mkdir -p "$output"
  docker build -f "src/$language/Dockerfile" -t "titanic-hw3-$language" .
  docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
    --mount "type=bind,source=$output,target=/app/output" \
    "titanic-hw3-$language" 2>&1 | tee "$output/run.log"
done
