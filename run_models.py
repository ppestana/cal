#!/usr/bin/env python
"""
run_models.py — Treinar modelos estatísticos (Fase 3).

Uso:
    python run_models.py          # treinar + guardar + popular expected_probabilities
    python run_models.py --eval   # só mostrar métricas LOSO sem escrever na DB
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

from cal.models.train import load_features, loso_evaluate, train_final, save_predictions
from rich.table import Table
from rich.console import Console

df = load_features()

logging.getLogger().info("A correr LOSO cross-validation...")
metrics = loso_evaluate(df)

console = Console()
seasons = sorted(metrics["season"].unique())

t1 = Table(title="LOSO — xYC e xRC  (Brier score — quanto mais baixo melhor)")
t1.add_column("Época",        style="cyan")
t1.add_column("xYC Brier",   justify="right")
t1.add_column("xYC pred",    justify="right")
t1.add_column("xYC actual",  justify="right")
t1.add_column("xRC Brier",   justify="right")
t1.add_column("xRC pred",    justify="right")
t1.add_column("xRC actual",  justify="right")

for s in seasons:
    yc = metrics[(metrics["season"] == s) & (metrics["model"] == "xYC")]
    rc = metrics[(metrics["season"] == s) & (metrics["model"] == "xRC")]
    if not yc.empty and not rc.empty:
        t1.add_row(
            s,
            str(yc.iloc[0]["brier"]),
            str(yc.iloc[0]["mean_pred"]),
            str(yc.iloc[0]["mean_actual"]),
            str(rc.iloc[0]["brier"]),
            str(rc.iloc[0]["mean_pred"]),
            str(rc.iloc[0]["mean_actual"]),
        )
console.print(t1)

t2 = Table(title="LOSO — xF Poisson  (MAE — quanto mais baixo melhor)")
t2.add_column("Época",        style="cyan")
t2.add_column("MAE faltas",   justify="right")
t2.add_column("Pred médio",   justify="right")
t2.add_column("Actual médio", justify="right")

for s in seasons:
    xf = metrics[(metrics["season"] == s) & (metrics["model"] == "xF")]
    if not xf.empty:
        t2.add_row(
            s,
            str(xf.iloc[0]["mae"]),
            str(xf.iloc[0]["mean_pred"]),
            str(xf.iloc[0]["mean_actual"]),
        )
console.print(t2)

if "--eval" not in sys.argv:
    train_final(df)
    save_predictions(df)
    logging.getLogger().info("Fase 3 concluída — modelos guardados e predições na DB.")
else:
    logging.getLogger().info("Modo --eval: sem escrita na DB.")
