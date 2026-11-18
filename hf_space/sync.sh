#!/usr/bin/env bash
# Copy the static explorer into a cloned Hugging Face Space repository.
set -euo pipefail
SPACE="${1:?usage: bash hf_space/sync.sh <path-to-space-clone>}"
cp docs/index.html "$SPACE/index.html"
mkdir -p "$SPACE/data"
cp docs/data/* "$SPACE/data/"
cp hf_space/README.md "$SPACE/README.md"
echo "Synced. Now: cd $SPACE && git add -A && git commit -m 'Update catalog' && git push"
