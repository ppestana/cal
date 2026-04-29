#!/usr/bin/env bash
# CAL — Arranque do ambiente de desenvolvimento
# Usar em Linux ou macOS.
#
# Primeira execução:  ./start.sh --init
# Execuções normais:  ./start.sh

set -euo pipefail

# ── Cores ───────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYN}▶${NC} $*"; }
ok()    { echo -e "${GRN}✓${NC} $*"; }
warn()  { echo -e "${YEL}!${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "\n${BOLD}CAL — Criticar a Arbitragem Legalmente${NC}"
echo -e "────────────────────────────────────────\n"

# ── Verificações ────────────────────────────────────────────
command -v docker &>/dev/null       || error "Docker não encontrado. Instalar em https://docs.docker.com/get-docker/"
docker info &>/dev/null 2>&1        || error "Docker daemon não está a correr. Iniciar o Docker Desktop (ou o serviço)."
command -v docker compose &>/dev/null \
  || docker-compose version &>/dev/null \
  || error "docker compose não encontrado."

# ── .env ────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        warn ".env não encontrado. A criar a partir de .env.example ..."
        cp .env.example .env
        warn "EDITAR .env e definir DB_PASSWORD antes de continuar."
        warn "  nano .env"
        exit 1
    else
        error ".env e .env.example em falta. Estrutura de ficheiros incompleta."
    fi
fi

# ── pgdata ──────────────────────────────────────────────────
if [ ! -d "pgdata" ]; then
    info "Primeira execução detectada — a criar directório pgdata/ ..."
    mkdir -p pgdata
    ok "pgdata/ criado. O schema será inicializado automaticamente."
fi

# ── Notebooks ───────────────────────────────────────────────
mkdir -p notebooks

# ── Build da imagem (se não existir ou --build passado) ─────
BUILD_FLAG=""
if [[ "${1:-}" == "--build" ]] || ! docker image inspect cal_dev:latest &>/dev/null 2>&1; then
    info "A construir imagem cal_dev (pode demorar alguns minutos na primeira vez) ..."
    BUILD_FLAG="--build"
fi

# ── Docker Compose up ───────────────────────────────────────
info "A iniciar containers ..."
docker compose up -d $BUILD_FLAG

# ── Aguardar healthcheck da DB ───────────────────────────────
info "A aguardar PostgreSQL ficar disponível ..."
MAX=30; COUNT=0
while ! docker compose exec -T db pg_isready -U cal_user -d cal &>/dev/null 2>&1; do
    COUNT=$((COUNT+1))
    [ $COUNT -ge $MAX ] && error "PostgreSQL não respondeu após ${MAX} tentativas."
    printf "."
    sleep 2
done
echo ""
ok "PostgreSQL pronto."

# ── Resumo ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Ambiente activo${NC}"
echo "────────────────────────────────────────"
echo -e "  ${GRN}PostgreSQL${NC}  → localhost:$(grep DB_PORT .env | cut -d= -f2 || echo 5432)"
echo -e "  ${GRN}FastAPI${NC}     → http://localhost:8000    (Fase 5)"
echo -e "  ${GRN}Streamlit${NC}   → http://localhost:8501    (Fase 5)"
echo -e "  ${GRN}Jupyter${NC}     → http://localhost:8888    (exploração)"
echo ""
echo -e "  ${CYN}Shell de desenvolvimento:${NC}"
echo -e "  docker compose exec dev bash"
echo ""
echo -e "  ${CYN}Correr ingestão:${NC}"
echo -e "  docker compose exec dev python run_ingest.py \"2024/25\""
echo ""
echo -e "  ${CYN}Parar o ambiente:${NC}"
echo -e "  ./stop.sh"
echo "────────────────────────────────────────"
