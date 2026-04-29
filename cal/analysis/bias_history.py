"""
cal/analysis/bias_history.py
Fase 6 — Histórico de Bias Scores por Jornada

Calcula os Z-scores de viés de forma incremental — após cada jornada —
permitindo ver como o padrão de um árbitro evolui ao longo da época.

Para cada árbitro, por cada jornada (acumulando jogos anteriores):
  - yellow_diff_bias_z  : Z-score acumulado até esta jornada
  - red_diff_bias_z
  - fouls_diff_bias_z
  - suspicion_score
  - n_games             : jogos acumulados

Guardado em:
  - referee_bias_history (referee_id, season, matchday, métricas)

Uso:
    python run_bias_history.py              # todas as épocas
    python run_bias_history.py "2024/25"    # época específica
"""

import logging
import numpy as np
import pandas as pd
from cal.db import get_cursor

log = logging.getLogger(__name__)

SEASONS = [
    "2017/18","2018/19","2019/20","2020/21",
    "2021/22","2022/23","2023/24","2024/25",
]

QUERY = """
SELECT
    mf.match_id,
    mf.match_date,
    mf.season,
    mf.referee_id,
    r.name                          AS referee,
    mf.matchday,
    -- Agregar as duas equipas por jogo (somar cartões e faltas)
    SUM(mf.yellow_cards)            AS obs_yellows,
    SUM(mf.red_cards)               AS obs_reds,
    SUM(mf.fouls)                   AS obs_fouls,
    SUM(ep.expected_yellows)        AS exp_yellows,
    SUM(ep.expected_reds)           AS exp_reds,
    SUM(ep.expected_fouls)          AS exp_fouls
FROM match_features mf
JOIN expected_probabilities ep USING (match_id, team_id)
JOIN referees r ON r.referee_id = mf.referee_id
WHERE mf.referee_id IS NOT NULL
  AND mf.referee_id > 1
  AND mf.yellow_cards IS NOT NULL
  AND mf.season = ANY(%(seasons)s)
GROUP BY mf.match_id, mf.match_date, mf.season, mf.referee_id, r.name, mf.matchday
ORDER BY mf.match_date, mf.match_id
"""


def _z_binomial(obs: float, exp: float, n: int) -> float:
    if n < 5 or exp <= 0: return 0.0
    var = exp * (1 - exp / n)
    return (obs - exp) / np.sqrt(var) if var > 0 else 0.0


def _z_ttest(obs_mean: float, exp_mean: float, std: float, n: int) -> float:
    if n < 5 or std <= 0: return 0.0
    return (obs_mean - exp_mean) / (std / np.sqrt(n))


def compute_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada árbitro e época, calcula Z-scores acumulados por jornada.
    Devolve DataFrame com uma linha por (referee_id, season, matchday).
    """
    rows = []
    for (ref_id, season), grp in df.groupby(["referee_id", "season"]):
        grp = grp.sort_values(["matchday", "match_date"]).reset_index(drop=True)
        ref_name = grp["referee"].iloc[0]

        for jornada in sorted(grp["matchday"].unique()):
            # Dados acumulados até esta jornada
            acc = grp[grp["matchday"] <= jornada]
            n = len(acc)
            if n < 3:
                continue

            obs_y = int((acc["obs_yellows"] > 0).sum())
            exp_y = float(acc["exp_yellows"].sum())
            obs_r = int((acc["obs_reds"] > 0).sum())
            exp_r = float(acc["exp_reds"].sum())

            obs_f_mean = float(acc["obs_fouls"].mean())
            exp_f_mean = float(acc["exp_fouls"].mean())
            std_f      = float(acc["obs_fouls"].std()) if n > 1 else 1.0

            z_y = _z_binomial(obs_y, exp_y, n)
            z_r = _z_binomial(obs_r, exp_r, n)
            z_f = _z_ttest(obs_f_mean, exp_f_mean, std_f, n)
            suspicion = abs(z_y) + abs(z_r) + abs(z_f)

            rows.append({
                "referee_id":         ref_id,
                "referee":            ref_name,
                "season":             season,
                "matchday":           int(jornada),
                "n_games":            n,
                "yellow_diff_bias_z": round(z_y, 4),
                "red_diff_bias_z":    round(z_r, 4),
                "fouls_diff_bias_z":  round(z_f, 4),
                "suspicion_score":    round(suspicion, 4),
            })

    return pd.DataFrame(rows)


def save_history(df: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_bias_history (
                referee_id          INTEGER NOT NULL,
                season              VARCHAR(10) NOT NULL,
                matchday            SMALLINT NOT NULL,
                n_games             INTEGER,
                yellow_diff_bias_z  NUMERIC(8,4),
                red_diff_bias_z     NUMERIC(8,4),
                fouls_diff_bias_z   NUMERIC(8,4),
                suspicion_score     NUMERIC(8,4),
                PRIMARY KEY (referee_id, season, matchday)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rbh_season
                ON referee_bias_history (season, matchday)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rbh_suspicion
                ON referee_bias_history (suspicion_score DESC)
        """)
        conn.commit()

        records = df.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany("""
            INSERT INTO referee_bias_history
                (referee_id, season, matchday, n_games,
                 yellow_diff_bias_z, red_diff_bias_z,
                 fouls_diff_bias_z, suspicion_score)
            VALUES
                (%(referee_id)s, %(season)s, %(matchday)s, %(n_games)s,
                 %(yellow_diff_bias_z)s, %(red_diff_bias_z)s,
                 %(fouls_diff_bias_z)s, %(suspicion_score)s)
            ON CONFLICT (referee_id, season, matchday) DO UPDATE SET
                n_games             = EXCLUDED.n_games,
                yellow_diff_bias_z  = EXCLUDED.yellow_diff_bias_z,
                red_diff_bias_z     = EXCLUDED.red_diff_bias_z,
                fouls_diff_bias_z   = EXCLUDED.fouls_diff_bias_z,
                suspicion_score     = EXCLUDED.suspicion_score
        """, records)
        conn.commit()
    log.info("Guardados %d registos em referee_bias_history", len(df))


def run(seasons=None):
    targets = seasons if seasons else SEASONS
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY, {"seasons": targets})
        rows = cur.fetchall()

    if not rows:
        log.warning("Query retornou 0 linhas para %s — a tentar query alternativa", targets)
        # Fallback: usar match_stats + expected_probabilities sem match_features
        with get_cursor(dict_cursor=True) as (conn, cur):
            cur.execute("""
                SELECT
                    m.match_id,
                    m.match_date,
                    s.label                         AS season,
                    m.referee_id,
                    r.name                          AS referee,
                    m.matchday,
                    SUM(ms.yellow_cards)            AS obs_yellows,
                    SUM(ms.red_cards)               AS obs_reds,
                    SUM(ms.fouls)                   AS obs_fouls,
                    SUM(ep.expected_yellows)        AS exp_yellows,
                    SUM(ep.expected_reds)           AS exp_reds,
                    SUM(ep.expected_fouls)          AS exp_fouls
                FROM matches m
                JOIN seasons s USING (season_id)
                JOIN referees r ON r.referee_id = m.referee_id
                JOIN match_stats ms USING (match_id)
                JOIN expected_probabilities ep ON ep.match_id = m.match_id
                    AND ep.team_id = ms.team_id
                WHERE m.referee_id > 1
                  AND s.label = ANY(%(seasons)s)
                GROUP BY m.match_id, m.match_date, s.label, m.referee_id, r.name, m.matchday
                ORDER BY m.match_date, m.match_id
            """, {"seasons": targets})
            rows = cur.fetchall()
        log.info("Query alternativa: %d linhas", len(rows))

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("Sem dados para calcular histórico — épocas: %s", targets)
        return

    log.info("Dados carregados: %d linhas para %s", len(df), targets)

    for col in ["obs_yellows","obs_reds","obs_fouls",
                "exp_yellows","exp_reds","exp_fouls","matchday"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    history = compute_history(df)
    log.info("Histórico calculado: %d entradas", len(history))
    save_history(history)


if __name__ == "__main__":
    import sys, dotenv, logging
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])
    run(sys.argv[1:] or None)
