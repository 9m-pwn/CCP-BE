#!/usr/bin/env bash
set -euo pipefail

docker compose stop redis
docker compose ps redis
