#!/usr/bin/env python
"""
run_bias.py — Calcular bias scores (Fase 4).

Uso:
    python run_bias.py
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

from cal.bias_engine import run
run()
