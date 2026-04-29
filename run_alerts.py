#!/usr/bin/env python
"""
run_alerts.py — Gerar alertas automáticos pós-jornada (Fase 6).

Uso:
    python run_alerts.py                    # época actual (2024/25)
    python run_alerts.py "2023/24"          # época específica
    python run_alerts.py "2024/25" --save   # guardar alertas na DB
"""
import sys, logging, dotenv
from rich.logging import RichHandler
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(show_path=False)])
from cal.analysis.alerts import run
season = next((a for a in sys.argv[1:] if "/" in a), None)
save   = "--save" in sys.argv
report = run(season=season, save=save)
print(report)
