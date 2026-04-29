#!/usr/bin/env bash
# CAL — Paragem do ambiente de desenvolvimento
# Os dados ficam preservados em pgdata/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "▶ A parar containers CAL ..."
docker compose down

echo "✓ Containers parados. Dados preservados em pgdata/"
echo "  Para voltar: ./start.sh"
