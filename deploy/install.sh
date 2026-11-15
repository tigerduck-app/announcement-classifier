#!/usr/bin/env bash
# Usage: deploy/install.sh <tag>   (for example model-2026-11-15)
# Downloads the release assets into ./model, checks SHA256SUMS, builds ./.venv and runs the service test.
set -euo pipefail

TAG="${1:?usage: install.sh <tag>}"
REPO="tigerduck-app/announcement-classifier"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ROOT}/model"

mkdir -p "${MODEL}"
gh release download "${TAG}" --repo "${REPO}" --dir "${MODEL}" --clobber
(cd "${MODEL}" && shasum -a 256 -c SHA256SUMS)

uv venv "${ROOT}/.venv"
uv pip install --python "${ROOT}/.venv/bin/python" -r "${ROOT}/deploy/requirements.txt"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/test_serve.py" "${MODEL}"
