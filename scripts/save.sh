#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository."
  exit 1
fi

if [ -z "${1-}" ]; then
  msg="Auto save $(date '+%Y-%m-%d %H:%M:%S')"
else
  msg="$1"
fi

if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to save."
  exit 0
fi

git add -A

git commit -m "$msg"

git push

