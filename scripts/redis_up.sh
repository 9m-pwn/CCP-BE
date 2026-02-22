#!/usr/bin/env bash
set -euo pipefail

docker compose up -d redis
docker compose ps redis
