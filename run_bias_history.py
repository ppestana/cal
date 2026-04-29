#!/usr/bin/env python
"""run_bias_history.py — Calcular histórico de bias scores por jornada (Fase 6)."""
import sys, logging, dotenv
from rich.logging import RichHandler
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(show_path=False)])
from cal.analysis.bias_history import run
run(sys.argv[1:] or None)
