#!/usr/bin/env python3
"""
run_ingest.py — Entry point para a pipeline de ingestão do CAL

Uso:
    # Todas as temporadas disponíveis
    python run_ingest.py

    # Temporadas específicas
    python run_ingest.py "2023/24" "2022/23"

    # Apenas a mais recente
    python run_ingest.py "2024/25"
"""
import sys
import logging
import dotenv
from rich.logging import RichHandler

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

from cal.ingest.footballdata import run

if __name__ == "__main__":
    seasons = sys.argv[1:] or None
    run(seasons)
