"""
cal/models/train.py
Fase 3 — Modelos Estatísticos

Treina três modelos a partir da feature matrix (match_features):
  - xYC : regressão logística — P(cartão amarelo | contexto)
  - xRC : regressão logística — P(cartão vermelho | contexto)
  - xF  : regressão Poisson   — E(faltas | contexto)

Estratégia de validação: leave-one-season-out (LOSO)
  Para cada temporada T:
    - Treinar em todas as temporadas excepto T
    - Prever em T
    - Guardar probabilidades esperadas em expected_probabilities

Os modelos finais (treinados em TODOS os dados) são guardados em
  cal/models/saved/models.joblib

Uso:
    python -m cal.models.train          # treinar + guardar + popular expected_probabilities
    python -m cal.models.train --eval   # só mostrar métricas LOSO sem guardar na DB
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.api as sm

from cal.db import get_cursor

log = logging.getLogger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)

# ── Features usadas em cada modelo ───────────────────────────────────────────
FEATURES_YC = [
    "is_home",
    "matchday",
    "team_points_r10",
    "opp_points_r10",
    "team_goals_scored_r10",
    "team_goals_conceded_r10",
    "ref_yellows_pg_r20",
    "ref_fouls_pg_r20",
    "ref_games_total",
]

FEATURES_RC = [
    "is_home",
    "matchday",
    "team_points_r10",
    "opp_points_r10",
    "ref_reds_pg_r20",
    "ref_yellows_pg_r20",
    "ref_games_total",
]

FEATURES_F = [
    "is_home",
    "matchday",
    "team_points_r10",
    "opp_points_r10",
    "team_goals_scored_r10",
    "team_goals_conceded_r10",
    "ref_fouls_pg_r20",
    "ref_games_total",
]

SEASONS_ORDER = [
    "2017/18", "2018/19", "2019/20", "2020/21",
    "2021/22", "2022/23", "2023/24", "2024/25",
]


# ── Carregar feature matrix ───────────────────────────────────────────────────

def load_features(seasons: Optional[list[str]] = None) -> pd.DataFrame:
    """Carrega match_features da DB. Exclui jogos sem árbitro ou sem stats."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("SELECT * FROM match_features ORDER BY match_date, match_id")
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    log.info("match_features carregadas: %d linhas", len(df))

    # Excluir jogos sem árbitro válido
    df = df[df["referee_id"].notna() & (df["referee_id"] > 1)]

    # Excluir linhas sem variáveis alvo
    df = df[df["yellow_cards"].notna() & df["fouls"].notna()]

    # Apenas temporadas históricas completas para treino/validação
    df = df[df["season"].isin(SEASONS_ORDER)]

    if seasons:
        df = df[df["season"].isin(seasons)]

    # Converter colunas numéricas
    num_cols = FEATURES_YC + FEATURES_RC + FEATURES_F + ["yellow_cards", "red_cards", "fouls"]
    for col in set(num_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    log.info("Após filtros: %d linhas", len(df))
    return df.reset_index(drop=True)


# ── Modelos ───────────────────────────────────────────────────────────────────

def _fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple:
    """Treina regressão logística com standardização. Devolve (model, scaler)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    model.fit(Xs, y)
    return model, scaler


def _predict_logistic(model, scaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    return model.predict_proba(Xs)[:, 1]


def _fit_poisson(X: np.ndarray, y: np.ndarray):
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    return Poisson(y.astype(float), Xc).fit(disp=False, maxiter=200)


def _predict_poisson(result, X: np.ndarray) -> np.ndarray:
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    return result.predict(Xc)


# ── Métricas ──────────────────────────────────────────────────────────────────

def _metrics_logistic(y_true, y_pred_prob, name: str) -> dict:
    y_bin = (y_true > 0).astype(int)
    return {
        "model":       name,
        "brier":       round(brier_score_loss(y_bin, y_pred_prob), 4),
        "log_loss":    round(log_loss(y_bin, y_pred_prob), 4),
        "mean_pred":   round(float(y_pred_prob.mean()), 4),
        "mean_actual": round(float(y_bin.mean()), 4),
    }


def _metrics_poisson(y_true, y_pred, name: str) -> dict:
    return {
        "model":       name,
        "mae":         round(mean_absolute_error(y_true, y_pred), 4),
        "mean_pred":   round(float(y_pred.mean()), 4),
        "mean_actual": round(float(y_true.mean()), 4),
    }


# ── LOSO cross-validation ─────────────────────────────────────────────────────

def loso_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-One-Season-Out cross-validation. Devolve métricas por temporada."""
    results = []
    for val_season in sorted(df["season"].unique()):
        train = df[df["season"] != val_season]
        val   = df[df["season"] == val_season]
        if len(train) < 100 or len(val) < 10:
            continue

        log.info("  LOSO val=%s (treino=%d val=%d)", val_season, len(train), len(val))

        # xYC
        m, s = _fit_logistic(train[FEATURES_YC].values, (train["yellow_cards"] > 0).astype(int).values)
        p = _predict_logistic(m, s, val[FEATURES_YC].values)
        r = _metrics_logistic(val["yellow_cards"].values, p, "xYC")
        r["season"] = val_season
        results.append(r)

        # xRC
        m, s = _fit_logistic(train[FEATURES_RC].values, (train["red_cards"] > 0).astype(int).values)
        p = _predict_logistic(m, s, val[FEATURES_RC].values)
        r = _metrics_logistic(val["red_cards"].values, p, "xRC")
        r["season"] = val_season
        results.append(r)

        # xF
        rf = _fit_poisson(train[FEATURES_F].values, train["fouls"].values)
        p  = _predict_poisson(rf, val[FEATURES_F].values)
        r  = _metrics_poisson(val["fouls"].values, p, "xF")
        r["season"] = val_season
        results.append(r)

    return pd.DataFrame(results)


# ── Modelos finais ────────────────────────────────────────────────────────────

def train_final(df: pd.DataFrame) -> dict:
    """Treina modelos finais em todos os dados e guarda em disco."""
    log.info("A treinar modelos finais em %d linhas...", len(df))
    models = {
        "xYC": _fit_logistic(df[FEATURES_YC].values, (df["yellow_cards"] > 0).astype(int).values),
        "xRC": _fit_logistic(df[FEATURES_RC].values, (df["red_cards"] > 0).astype(int).values),
        "xF":  _fit_poisson(df[FEATURES_F].values, df["fouls"].values),
    }
    path = MODELS_DIR / "models.joblib"
    joblib.dump(models, path)
    log.info("Modelos guardados em %s", path)
    return models


def load_models() -> dict:
    path = MODELS_DIR / "models.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelos não encontrados em {path}.")
    return joblib.load(path)


# ── Popular expected_probabilities ────────────────────────────────────────────

def save_predictions(df: pd.DataFrame) -> None:
    """Calcula e guarda expected_probabilities via LOSO."""
    CREATE = """
    CREATE TABLE IF NOT EXISTS expected_probabilities (
        match_id            INTEGER NOT NULL,
        team_id             INTEGER NOT NULL,
        expected_yellows    NUMERIC(8,4),
        expected_reds       NUMERIC(8,4),
        expected_fouls      NUMERIC(8,4),
        model_version       VARCHAR(20) DEFAULT 'v1.0',
        computed_at         TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (match_id, team_id)
    );"""

    UPSERT = """
    INSERT INTO expected_probabilities
        (match_id, team_id, expected_yellows, expected_reds, expected_fouls, model_version)
    VALUES (%s, %s, %s, %s, %s, 'v1.0')
    ON CONFLICT (match_id, team_id) DO UPDATE SET
        expected_yellows = EXCLUDED.expected_yellows,
        expected_reds    = EXCLUDED.expected_reds,
        expected_fouls   = EXCLUDED.expected_fouls,
        computed_at      = NOW()
    """

    all_preds = []
    for val_season in sorted(df["season"].unique()):
        train = df[df["season"] != val_season]
        val   = df[df["season"] == val_season]
        if len(train) < 100:
            continue

        m_yc, s_yc = _fit_logistic(train[FEATURES_YC].values, (train["yellow_cards"] > 0).astype(int).values)
        m_rc, s_rc = _fit_logistic(train[FEATURES_RC].values, (train["red_cards"] > 0).astype(int).values)
        r_f        = _fit_poisson(train[FEATURES_F].values, train["fouls"].values)

        preds = val[["match_id", "team_id"]].copy()
        preds["expected_yellows"] = _predict_logistic(m_yc, s_yc, val[FEATURES_YC].values)
        preds["expected_reds"]    = _predict_logistic(m_rc, s_rc, val[FEATURES_RC].values)
        preds["expected_fouls"]   = _predict_poisson(r_f, val[FEATURES_F].values)
        all_preds.append(preds)
        log.info("  %s — %d predições", val_season, len(preds))

    if not all_preds:
        log.warning("Sem predições para guardar.")
        return

    predictions = pd.concat(all_preds, ignore_index=True)

    with get_cursor() as (conn, cur):
        cur.execute(CREATE)
        conn.commit()
        records = [
            (int(r["match_id"]), int(r["team_id"]),
             round(float(r["expected_yellows"]), 4),
             round(float(r["expected_reds"]), 4),
             round(float(r["expected_fouls"]), 4))
            for _, r in predictions.iterrows()
        ]
        cur.executemany(UPSERT, records)
        conn.commit()

    log.info("Guardadas %d predições em expected_probabilities", len(predictions))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import dotenv
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.console import Console

    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])

    eval_only = "--eval" in sys.argv
    df = load_features()

    log.info("A correr LOSO cross-validation...")
    metrics = loso_evaluate(df)

    console = Console()
    seasons = sorted(metrics["season"].unique())

    t1 = Table(title="LOSO — xYC e xRC (Brier score ↓ melhor)")
    for col in ["Época", "xYC Brier", "xYC mean_pred", "xYC mean_actual", "xRC Brier"]:
        t1.add_column(col)
    for s in seasons:
        yc = metrics[(metrics["season"] == s) & (metrics["model"] == "xYC")]
        rc = metrics[(metrics["season"] == s) & (metrics["model"] == "xRC")]
        if not yc.empty and not rc.empty:
            t1.add_row(s, str(yc.iloc[0]["brier"]), str(yc.iloc[0]["mean_pred"]),
                       str(yc.iloc[0]["mean_actual"]), str(rc.iloc[0]["brier"]))
    console.print(t1)

    t2 = Table(title="LOSO — xF Poisson (MAE ↓ melhor)")
    for col in ["Época", "MAE faltas", "Mean pred", "Mean actual"]:
        t2.add_column(col)
    for s in seasons:
        xf = metrics[(metrics["season"] == s) & (metrics["model"] == "xF")]
        if not xf.empty:
            t2.add_row(s, str(xf.iloc[0]["mae"]), str(xf.iloc[0]["mean_pred"]),
                       str(xf.iloc[0]["mean_actual"]))
    console.print(t2)

    if not eval_only:
        train_final(df)
        save_predictions(df)
        log.info("Fase 3 concluída.")
