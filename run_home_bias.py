#!/usr/bin/env python
"""
run_home_bias.py — Calcular home bias por contexto de marcador (Fase 7)

Uso:
    python run_home_bias.py              # todas as épocas completas
    python run_home_bias.py "2023/24"    # época específica
"""
import sys, logging, dotenv
from rich.logging import RichHandler
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(show_path=False)])
from cal.analysis.home_bias import run
run(sys.argv[1:] or None)
