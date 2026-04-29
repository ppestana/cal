#!/usr/bin/env python
"""
run_cards.py — Análise de cartões por árbitro × equipa.

Uso:
    python run_cards.py
"""
import logging
import dotenv
from rich.logging import RichHandler

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

from cal.analysis.cards_by_team import run
run()
