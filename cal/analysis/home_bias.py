"""
cal/analysis/home_bias.py
Fase 7 — Análise de Home Bias por Contexto de Marcador

Para cada árbitro, compara as decisões (faltas, amarelos) dadas à equipa
da casa vs. visitante, segmentadas pelo contexto do marcador:

  - HOME_WINNING  : equipa da casa a ganhar ao intervalo (ht_score_diff > 0)
  - DRAW          : empate ao intervalo (ht_score_diff = 0)
  - HOME_LOSING   : equipa da casa a perder ao intervalo (ht_score_diff < 0)

O viés manifesta-se quando o árbitro trata as duas equipas de forma
desigual no mesmo contexto — especialmente quando a equipa da casa
está a perder (maior pressão dos adeptos).

Métricas calculadas:
  - media_faltas_casa / media_faltas_visitante  por contexto
  - media_amarelos_casa / media_amarelos_visitante  por contexto
  - home_bias_fouls_z    : Z-score do diferencial casa-visitante em faltas
  - home_bias_yellow_z   : Z-score do diferencial casa-visitante em amarelos
  - pressure_bias_index  : diferença entre viés quando casa perde vs. casa ganha

Guardado em:
  - referee_home_bias (referee_id, season, score_context, métricas)

Uso:
    python run_home_bias.py              # todas as épocas
    python run_home_bias.py "2023/24"    # época específica
"""

import logging
import numpy as np
import pandas as pd
from cal.db import get_cursor

log = logging.getLogger(__name__)

SEASONS_COMPLETE = [
    "2017/18","2018/19","2019/20","2020/21",
    "2021/22","2022/23","2023/24",
]

QUERY = """
SELECT
    m.match_id,
    s.label                             AS season,
    m.referee_id,
    r.name                              AS referee,
    -- Equipa da casa
    ms_h.yellow_cards                   AS yc_home,
    ms_h.red_cards                      AS rc_home,
    ms_h.fouls                          AS fouls_home,
    -- Equipa visitante
    ms_a.yellow_cards                   AS yc_away,
    ms_a.red_cards                      AS rc_away,
    ms_a.fouls                          AS fouls_away,
    -- Contexto do marcador ao intervalo
    CASE
        WHEN m.ht_home_goals > m.ht_away_goals  THEN 'HOME_WINNING'
        WHEN m.ht_home_goals = m.ht_away_goals  THEN 'DRAW'
        ELSE                                         'HOME_LOSING'
    END                                 AS score_context,
    -- Diferença no marcador ao intervalo (perspectiva da casa)
    (m.ht_home_goals - m.ht_away_goals) AS ht_score_diff
FROM matches m
JOIN seasons s USING (season_id)
JOIN referees r ON r.referee_id = m.referee_id
-- match_stats para equipa da casa
JOIN match_stats ms_h ON ms_h.match_id = m.match_id
    AND ms_h.team_id = m.home_team_id
-- match_stats para equipa visitante
JOIN match_stats ms_a ON ms_a.match_id = m.match_id
    AND ms_a.team_id = m.away_team_id
WHERE m.referee_id > 1
  AND ms_h.fouls IS NOT NULL
  AND ms_a.fouls IS NOT NULL
  AND s.label = ANY(%(seasons)s)
ORDER BY m.referee_id, s.label, m.match_id
"""


# ── Cálculo ───────────────────────────────────────────────────────────────────

def compute_home_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada árbitro+época+contexto calcula métricas de home bias.
    """
    rows = []

    for (ref_id, season), grp in df.groupby(["referee_id", "season"]):
        ref_name = grp["referee"].iloc[0]
        n_total  = len(grp)

        if n_total < 5:
            continue

        # Médias globais do árbitro nesta época (referência)
        global_fouls_diff  = float((grp["fouls_home"]  - grp["fouls_away"]).mean())
        global_yellow_diff = float((grp["yc_home"]     - grp["yc_away"]).mean())
        global_fouls_std   = float((grp["fouls_home"]  - grp["fouls_away"]).std())  if n_total > 1 else 1.0
        global_yellow_std  = float((grp["yc_home"]     - grp["yc_away"]).std())     if n_total > 1 else 1.0

        for context in ["HOME_WINNING", "DRAW", "HOME_LOSING"]:
            ctx = grp[grp["score_context"] == context]
            n = len(ctx)
            if n < 3:
                continue

            fouls_diff_vals  = (ctx["fouls_home"] - ctx["fouls_away"]).astype(float)
            yellow_diff_vals = (ctx["yc_home"]    - ctx["yc_away"]).astype(float)

            mean_fouls_diff  = float(fouls_diff_vals.mean())
            mean_yellow_diff = float(yellow_diff_vals.mean())

            # Z-score: desvio do contexto face à média global do árbitro
            z_fouls  = (mean_fouls_diff  - global_fouls_diff)  / (global_fouls_std  / np.sqrt(n)) \
                       if global_fouls_std > 0 else 0.0
            z_yellow = (mean_yellow_diff - global_yellow_diff) / (global_yellow_std / np.sqrt(n)) \
                       if global_yellow_std > 0 else 0.0

            rows.append({
                "referee_id":           ref_id,
                "referee":              ref_name,
                "season":               season,
                "score_context":        context,
                "n_games":              n,
                "media_fouls_home":     round(float(ctx["fouls_home"].mean()), 3),
                "media_fouls_away":     round(float(ctx["fouls_away"].mean()), 3),
                "media_fouls_diff":     round(mean_fouls_diff, 3),
                "media_yc_home":        round(float(ctx["yc_home"].mean()), 3),
                "media_yc_away":        round(float(ctx["yc_away"].mean()), 3),
                "media_yc_diff":        round(mean_yellow_diff, 3),
                "home_bias_fouls_z":    round(z_fouls, 4),
                "home_bias_yellow_z":   round(z_yellow, 4),
            })

    df_out = pd.DataFrame(rows)

    # Calcular pressure_bias_index por árbitro+época:
    # diferença entre o viés quando casa PERDE vs. quando GANHA
    # (positivo = árbitro protege mais a casa quando perde = home bias clássico)
    if not df_out.empty:
        pivot = df_out.pivot_table(
            index=["referee_id","referee","season"],
            columns="score_context",
            values="media_fouls_diff",
            aggfunc="mean"
        ).reset_index()

        if "HOME_LOSING" in pivot.columns and "HOME_WINNING" in pivot.columns:
            pivot["pressure_bias_index"] = (
                pivot["HOME_LOSING"].fillna(0) - pivot["HOME_WINNING"].fillna(0)
            ).round(3)
            df_out = df_out.merge(
                pivot[["referee_id","season","pressure_bias_index"]],
                on=["referee_id","season"], how="left"
            )
        else:
            df_out["pressure_bias_index"] = 0.0

    return df_out


def save_home_bias(df: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_home_bias (
                referee_id              INTEGER NOT NULL,
                season                  VARCHAR(10) NOT NULL,
                score_context           VARCHAR(20) NOT NULL,
                n_games                 INTEGER,
                media_fouls_home        NUMERIC(6,3),
                media_fouls_away        NUMERIC(6,3),
                media_fouls_diff        NUMERIC(6,3),
                media_yc_home           NUMERIC(6,3),
                media_yc_away           NUMERIC(6,3),
                media_yc_diff           NUMERIC(6,3),
                home_bias_fouls_z       NUMERIC(8,4),
                home_bias_yellow_z      NUMERIC(8,4),
                pressure_bias_index     NUMERIC(6,3),
                PRIMARY KEY (referee_id, season, score_context)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rhb_referee_season
                ON referee_home_bias (referee_id, season)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rhb_pressure
                ON referee_home_bias (pressure_bias_index DESC)
        """)
        conn.commit()

        records = df.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany("""
            INSERT INTO referee_home_bias (
                referee_id, season, score_context, n_games,
                media_fouls_home, media_fouls_away, media_fouls_diff,
                media_yc_home, media_yc_away, media_yc_diff,
                home_bias_fouls_z, home_bias_yellow_z, pressure_bias_index
            ) VALUES (
                %(referee_id)s, %(season)s, %(score_context)s, %(n_games)s,
                %(media_fouls_home)s, %(media_fouls_away)s, %(media_fouls_diff)s,
                %(media_yc_home)s, %(media_yc_away)s, %(media_yc_diff)s,
                %(home_bias_fouls_z)s, %(home_bias_yellow_z)s, %(pressure_bias_index)s
            )
            ON CONFLICT (referee_id, season, score_context) DO UPDATE SET
                n_games             = EXCLUDED.n_games,
                media_fouls_home    = EXCLUDED.media_fouls_home,
                media_fouls_away    = EXCLUDED.media_fouls_away,
                media_fouls_diff    = EXCLUDED.media_fouls_diff,
                media_yc_home       = EXCLUDED.media_yc_home,
                media_yc_away       = EXCLUDED.media_yc_away,
                media_yc_diff       = EXCLUDED.media_yc_diff,
                home_bias_fouls_z   = EXCLUDED.home_bias_fouls_z,
                home_bias_yellow_z  = EXCLUDED.home_bias_yellow_z,
                pressure_bias_index = EXCLUDED.pressure_bias_index
        """, records)
        conn.commit()
    log.info("Guardados %d registos em referee_home_bias", len(df))


def print_preview(df: pd.DataFrame) -> None:
    """Mostra árbitros com maior pressure_bias_index no terminal."""
    if df.empty:
        print("Sem dados.")
        return

    print("\n" + "═"*80)
    print("HOME BIAS — ÁRBITROS COM MAIOR PRESSURE BIAS INDEX")
    print("(positivo = mais favorável à casa quando perde do que quando ganha)")
    print("═"*80)

    summary = (
        df.groupby(["referee_id","referee","season"])["pressure_bias_index"]
        .first().reset_index()
        .sort_values("pressure_bias_index", ascending=False)
        .head(15)
    )
    print(f"\n  {'Árbitro':<32} {'Época':<10} {'PressureBiasIndex':>18}")
    print(f"  {'─'*32} {'─'*10} {'─'*18}")
    for _, row in summary.iterrows():
        flag = " ← ATENÇÃO" if abs(float(row["pressure_bias_index"])) > 2 else ""
        print(f"  {row['referee']:<32} {row['season']:<10} "
              f"{float(row['pressure_bias_index']):>+18.3f}{flag}")

    print("\n" + "═"*80)
    print("CONTEXTO: casa a perder — faltas a favor da casa vs. visitante")
    print("═"*80)
    losing = df[df["score_context"] == "HOME_LOSING"].sort_values(
        "home_bias_fouls_z", ascending=False
    ).head(10)
    print(f"\n  {'Árbitro':<32} {'Época':<10} {'Z Faltas':>9} {'Z Amarelos':>11} "
          f"{'F Casa':>8} {'F Visit.':>9} {'Jogos':>6}")
    print(f"  {'─'*32} {'─'*10} {'─'*9} {'─'*11} {'─'*8} {'─'*9} {'─'*6}")
    for _, row in losing.iterrows():
        print(f"  {row['referee']:<32} {row['season']:<10} "
              f"{float(row['home_bias_fouls_z']):>+9.3f} "
              f"{float(row['home_bias_yellow_z']):>+11.3f} "
              f"{float(row['media_fouls_home']):>8.1f} "
              f"{float(row['media_fouls_away']):>9.1f} "
              f"{int(row['n_games']):>6}")


def run(seasons=None) -> None:
    targets = seasons if seasons else SEASONS_COMPLETE

    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY, {"seasons": targets})
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    log.info("Dados carregados: %d jogos para %s", len(df), targets)

    if df.empty:
        log.warning("Sem dados — verificar se match_stats tem ht_home_goals/ht_away_goals")
        return

    for col in ["yc_home","rc_home","fouls_home","yc_away","rc_away","fouls_away","ht_score_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    result = compute_home_bias(df)
    log.info("Home bias calculado: %d entradas", len(result))

    if not result.empty:
        save_home_bias(result)
        print_preview(result)
    else:
        log.warning("Resultado vazio — amostras insuficientes por contexto")


if __name__ == "__main__":
    import sys, dotenv, logging
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])
    run(sys.argv[1:] or None)
