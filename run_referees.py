#!/usr/bin/env python
"""
run_referees.py — Enriquecer árbitros via Sofascore API.

Uso:
    python run_referees.py              # todas as temporadas
    python run_referees.py "2023/24"    # temporada específica
    python run_referees.py "2022/23" "2023/24" "2024/25"
"""
import sys
import logging
import dotenv
from rich.logging import RichHandler

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

from cal.ingest.sofascore import run

seasons = sys.argv[1:] or None
run(seasons)
