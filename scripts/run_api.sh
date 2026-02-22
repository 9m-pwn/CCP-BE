#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-poc-env}"
echo "Starting API with conda env '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python run_api.py
