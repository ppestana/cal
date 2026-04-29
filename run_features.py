#!/usr/bin/env python
"""
run_features.py — Construir feature matrix (Fase 2).

Uso:
    python run_features.py              # todas as temporadas
    python run_features.py "2023/24"    # temporada específica
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

from cal.features.engineering import build, save_to_db

seasons = sys.argv[1:] or None
features = build(seasons=seasons)
save_to_db(features)

print(f"\nFeature matrix: {len(features)} linhas × {len(features.columns)} colunas")
print(f"Colunas: {list(features.columns)}")
