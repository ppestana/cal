"""
cal/analysis/cards_by_team.py
Análise de cartões por árbitro × equipa

Para cada árbitro, mostra quais as equipas a que deu mais e menos
cartões amarelos, vermelhos e faltas marcadas por jogo, por época e nas 7 épocas completas.

Resultados guardados em:
  - referee_team_cards  (estatísticas por árbitro+equipa+época)
  - referee_team_cards_total (acumulado 8 épocas)

Uso:
    python -m cal.analysis.cards_by_team
    python run_cards.py
"""

import logging
import pandas as pd
import numpy as np
from cal.db import get_cursor

log = logging.getLogger(__name__)

SEASONS_COMPLETE = [
    "2017/18", "2018/19", "2019/20", "2020/21",
    "2021/22", "2022/23", "2023/24",
]

# ── SQL ───────────────────────────────────────────────────────────────────────

QUERY = """
SELECT
    r.referee_id,
    r.name                          AS arbitro,
    s.label                         AS epoca,
    t.team_id,
    t.name                          AS equipa,
    COUNT(*)                        AS jogos,
    SUM(ms.yellow_cards)            AS amarelos_total,
    SUM(ms.red_cards)               AS vermelhos_total,
    SUM(ms.fouls)                   AS faltas_total,
    ROUND(AVG(ms.yellow_cards),3)   AS media_amarelos,
    ROUND(AVG(ms.red_cards),4)      AS media_vermelhos,
    ROUND(AVG(ms.fouls),3)          AS media_faltas,
    -- médias globais do árbitro nessa época (para calcular desvios)
    AVG(AVG(ms.yellow_cards)) OVER (
        PARTITION BY r.referee_id, s.label
    )                               AS media_geral_amarelos_epoca,
    AVG(AVG(ms.fouls)) OVER (
        PARTITION BY r.referee_id, s.label
    )                               AS media_geral_faltas_epoca
FROM matches m
JOIN referees r USING (referee_id)
JOIN seasons s USING (season_id)
JOIN match_stats ms USING (match_id)
JOIN teams t ON t.team_id = ms.team_id
WHERE r.referee_id > 1
  AND s.label = ANY(%(seasons)s)
GROUP BY r.referee_id, r.name, s.label, t.team_id, t.name
HAVING COUNT(*) >= 3
ORDER BY r.name, s.label, media_amarelos DESC
"""

# Query sem threshold mínimo — para amarelos_por_falta (inclui 1+ jogos)
QUERY_ALL_TEAMS = """
SELECT
    r.referee_id,
    r.name                          AS arbitro,
    t.team_id,
    t.name                          AS equipa,
    COUNT(*)                        AS jogos,
    SUM(ms.yellow_cards)            AS amarelos_total,
    SUM(ms.fouls)                   AS faltas_total,
    ROUND(AVG(ms.yellow_cards),3)   AS media_amarelos,
    ROUND(AVG(ms.fouls),2)          AS media_faltas,
    -- Amarelos por falta: 1 amarelo a cada X faltas
    ROUND(
        NULLIF(SUM(ms.fouls), 0)::numeric / NULLIF(SUM(ms.yellow_cards), 0),
        2
    )                               AS faltas_por_amarelo
FROM matches m
JOIN referees r USING (referee_id)
JOIN seasons s USING (season_id)
JOIN match_stats ms USING (match_id)
JOIN teams t ON t.team_id = ms.team_id
WHERE r.referee_id > 1
  AND s.label = ANY(%(seasons)s)
GROUP BY r.referee_id, r.name, t.team_id, t.name
ORDER BY r.name, faltas_por_amarelo DESC
"""

# ── Criar tabelas ─────────────────────────────────────────────────────────────

CREATE_BY_SEASON = """
CREATE TABLE IF NOT EXISTS referee_team_cards (
    referee_id          INTEGER NOT NULL,
    team_id             INTEGER NOT NULL,
    epoca               VARCHAR(10) NOT NULL,
    jogos               INTEGER,
    amarelos_total      INTEGER,
    vermelhos_total     INTEGER,
    faltas_total        INTEGER,
    media_amarelos      NUMERIC(6,3),
    media_vermelhos     NUMERIC(6,4),
    media_faltas        NUMERIC(6,3),
    desvio_amarelos     NUMERIC(6,3),
    percentil_amarelos  NUMERIC(6,1),
    desvio_faltas       NUMERIC(6,3),
    percentil_faltas    NUMERIC(6,1),
    amarelos_por_falta  NUMERIC(6,4),   -- amarelos / faltas (severidade)
    PRIMARY KEY (referee_id, team_id, epoca)
);
CREATE INDEX IF NOT EXISTS idx_rtc_referee_epoca
    ON referee_team_cards (referee_id, epoca);
CREATE INDEX IF NOT EXISTS idx_rtc_media
    ON referee_team_cards (media_amarelos DESC);
CREATE INDEX IF NOT EXISTS idx_rtc_faltas
    ON referee_team_cards (media_faltas DESC);
"""

CREATE_TOTAL = """
CREATE TABLE IF NOT EXISTS referee_team_cards_total (
    referee_id              INTEGER NOT NULL,
    team_id                 INTEGER NOT NULL,
    epocas                  INTEGER,      -- nº de épocas com >= 3 jogos juntos
    jogos_total             INTEGER,
    amarelos_total          INTEGER,
    vermelhos_total         INTEGER,
    media_amarelos          NUMERIC(6,3),
    media_vermelhos         NUMERIC(6,4),
    desvio_vs_media_arbitro NUMERIC(6,3), -- diff face à média geral do árbitro
    percentil_amarelos      NUMERIC(6,1), -- percentil entre todas as equipas
    ranking_amarelos        INTEGER,      -- 1 = equipa com mais amarelos deste árbitro
    PRIMARY KEY (referee_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_rtct_ranking
    ON referee_team_cards_total (referee_id, ranking_amarelos);
"""

UPSERT_BY_SEASON = """
INSERT INTO referee_team_cards (
    referee_id, team_id, epoca, jogos,
    amarelos_total, vermelhos_total, faltas_total,
    media_amarelos, media_vermelhos, media_faltas,
    desvio_amarelos, percentil_amarelos,
    desvio_faltas, percentil_faltas, amarelos_por_falta
) VALUES (
    %(referee_id)s, %(team_id)s, %(epoca)s, %(jogos)s,
    %(amarelos_total)s, %(vermelhos_total)s, %(faltas_total)s,
    %(media_amarelos)s, %(media_vermelhos)s, %(media_faltas)s,
    %(desvio_amarelos)s, %(percentil_amarelos)s,
    %(desvio_faltas)s, %(percentil_faltas)s, %(amarelos_por_falta)s
)
ON CONFLICT (referee_id, team_id, epoca) DO UPDATE SET
    jogos               = EXCLUDED.jogos,
    amarelos_total      = EXCLUDED.amarelos_total,
    vermelhos_total     = EXCLUDED.vermelhos_total,
    faltas_total        = EXCLUDED.faltas_total,
    media_amarelos      = EXCLUDED.media_amarelos,
    media_vermelhos     = EXCLUDED.media_vermelhos,
    media_faltas        = EXCLUDED.media_faltas,
    desvio_amarelos     = EXCLUDED.desvio_amarelos,
    percentil_amarelos  = EXCLUDED.percentil_amarelos,
    desvio_faltas       = EXCLUDED.desvio_faltas,
    percentil_faltas    = EXCLUDED.percentil_faltas,
    amarelos_por_falta  = EXCLUDED.amarelos_por_falta
"""

UPSERT_TOTAL = """
INSERT INTO referee_team_cards_total (
    referee_id, team_id, epocas, jogos_total,
    amarelos_total, vermelhos_total, faltas_total,
    media_amarelos, media_vermelhos, media_faltas,
    desvio_vs_media_arbitro, percentil_amarelos, ranking_amarelos,
    desvio_faltas_vs_media, percentil_faltas, ranking_faltas,
    amarelos_por_falta
) VALUES (
    %(referee_id)s, %(team_id)s, %(epocas)s, %(jogos_total)s,
    %(amarelos_total)s, %(vermelhos_total)s, %(faltas_total)s,
    %(media_amarelos)s, %(media_vermelhos)s, %(media_faltas)s,
    %(desvio_vs_media_arbitro)s, %(percentil_amarelos)s, %(ranking_amarelos)s,
    %(desvio_faltas_vs_media)s, %(percentil_faltas)s, %(ranking_faltas)s,
    %(amarelos_por_falta)s
)
ON CONFLICT (referee_id, team_id) DO UPDATE SET
    epocas                  = EXCLUDED.epocas,
    jogos_total             = EXCLUDED.jogos_total,
    amarelos_total          = EXCLUDED.amarelos_total,
    vermelhos_total         = EXCLUDED.vermelhos_total,
    faltas_total            = EXCLUDED.faltas_total,
    media_amarelos          = EXCLUDED.media_amarelos,
    media_vermelhos         = EXCLUDED.media_vermelhos,
    media_faltas            = EXCLUDED.media_faltas,
    desvio_vs_media_arbitro = EXCLUDED.desvio_vs_media_arbitro,
    percentil_amarelos      = EXCLUDED.percentil_amarelos,
    ranking_amarelos        = EXCLUDED.ranking_amarelos,
    desvio_faltas_vs_media  = EXCLUDED.desvio_faltas_vs_media,
    percentil_faltas        = EXCLUDED.percentil_faltas,
    ranking_faltas          = EXCLUDED.ranking_faltas,
    amarelos_por_falta      = EXCLUDED.amarelos_por_falta
"""


# ── Cálculo ───────────────────────────────────────────────────────────────────

def compute_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Por árbitro+equipa+época: adiciona desvio e percentil face
    à média do árbitro nessa época (para amarelos e faltas).
    """
    rows = []
    for (ref_id, epoca), grp in df.groupby(["referee_id", "epoca"]):
        media_arb_y = grp["media_amarelos"].mean()
        media_arb_f = grp["media_faltas"].mean()
        amarelos_vals = grp["media_amarelos"].values
        faltas_vals   = grp["media_faltas"].values

        for _, row in grp.iterrows():
            desvio_y = round(float(row["media_amarelos"]) - float(media_arb_y), 3)
            desvio_f = round(float(row["media_faltas"])   - float(media_arb_f), 3)
            pct_y = round(float((amarelos_vals < float(row["media_amarelos"])).mean()) * 100, 1)
            pct_f = round(float((faltas_vals   < float(row["media_faltas"])).mean())   * 100, 1)
            rows.append({
                "referee_id":         int(row["referee_id"]),
                "team_id":            int(row["team_id"]),
                "epoca":              row["epoca"],
                "jogos":              int(row["jogos"]),
                "amarelos_total":     int(row["amarelos_total"]),
                "vermelhos_total":    int(row["vermelhos_total"]),
                "faltas_total":       int(row["faltas_total"] or 0),
                "media_amarelos":     float(row["media_amarelos"]),
                "media_vermelhos":    float(row["media_vermelhos"]),
                "media_faltas":       float(row["media_faltas"] or 0),
                "desvio_amarelos":    desvio_y,
                "percentil_amarelos": pct_y,
                "desvio_faltas":      desvio_f,
                "percentil_faltas":   pct_f,
                "amarelos_por_falta": round(float(row["amarelos_total"]) / float(row["faltas_total"]), 4)
                                      if float(row["faltas_total"]) > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def compute_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acumulado 8 épocas: agrega por árbitro+equipa,
    adiciona desvio face à média geral do árbitro e ranking.
    """
    # Agregar por árbitro+equipa
    agg = (
        df.groupby(["referee_id", "arbitro", "team_id", "equipa"])
        .agg(
            epocas=("epoca", "count"),
            jogos_total=("jogos", "sum"),
            amarelos_total=("amarelos_total", "sum"),
            vermelhos_total=("vermelhos_total", "sum"),
            faltas_total=("faltas_total", "sum"),
        )
        .reset_index()
    )
    agg["media_amarelos"]  = agg["amarelos_total"] / agg["jogos_total"]
    agg["media_vermelhos"] = agg["vermelhos_total"] / agg["jogos_total"]
    agg["media_faltas"]    = agg["faltas_total"]    / agg["jogos_total"]

    rows = []
    for ref_id, grp in agg.groupby("referee_id"):
        media_geral_y = grp["media_amarelos"].mean()
        media_geral_f = grp["media_faltas"].mean()
        vals_y = grp["media_amarelos"].values
        vals_f = grp["media_faltas"].values
        grp = grp.copy()
        grp["ranking_amarelos"] = grp["media_amarelos"].rank(ascending=False, method="min").astype(int)
        grp["ranking_faltas"]   = grp["media_faltas"].rank(ascending=False, method="min").astype(int)

        for _, row in grp.iterrows():
            desvio_y = round(float(row["media_amarelos"]) - float(media_geral_y), 3)
            desvio_f = round(float(row["media_faltas"])   - float(media_geral_f), 3)
            pct_y    = round(float((vals_y < float(row["media_amarelos"])).mean()) * 100, 1)
            pct_f    = round(float((vals_f < float(row["media_faltas"])).mean())   * 100, 1)
            rows.append({
                "referee_id":              int(row["referee_id"]),
                "team_id":                 int(row["team_id"]),
                "arbitro":                 row["arbitro"],
                "equipa":                  row["equipa"],
                "epocas":                  int(row["epocas"]),
                "jogos_total":             int(row["jogos_total"]),
                "amarelos_total":          int(row["amarelos_total"]),
                "vermelhos_total":         int(row["vermelhos_total"]),
                "faltas_total":            int(row["faltas_total"]),
                "media_amarelos":          round(float(row["media_amarelos"]), 3),
                "media_vermelhos":         round(float(row["media_vermelhos"]), 4),
                "media_faltas":            round(float(row["media_faltas"]), 3),
                "desvio_vs_media_arbitro": desvio_y,
                "percentil_amarelos":      pct_y,
                "ranking_amarelos":        int(row["ranking_amarelos"]),
                "desvio_faltas_vs_media":  desvio_f,
                "percentil_faltas":        pct_f,
                "ranking_faltas":          int(row["ranking_faltas"]),
                "amarelos_por_falta":      round(float(row["amarelos_total"]) / float(row["faltas_total"]), 4)
                                           if float(row["faltas_total"]) > 0 else 0.0,
            })
    return pd.DataFrame(rows)


# ── Guardar na DB ─────────────────────────────────────────────────────────────

def save_by_season(df: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_team_cards (
                referee_id          INTEGER NOT NULL,
                team_id             INTEGER NOT NULL,
                epoca               VARCHAR(10) NOT NULL,
                jogos               INTEGER,
                amarelos_total      INTEGER,
                vermelhos_total     INTEGER,
                faltas_total        INTEGER,
                media_amarelos      NUMERIC(6,3),
                media_vermelhos     NUMERIC(6,4),
                media_faltas        NUMERIC(6,3),
                desvio_amarelos     NUMERIC(6,3),
                percentil_amarelos  NUMERIC(6,1),
                desvio_faltas       NUMERIC(6,3),
                percentil_faltas    NUMERIC(6,1),
                amarelos_por_falta  NUMERIC(6,4),
                PRIMARY KEY (referee_id, team_id, epoca)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtc_referee_epoca ON referee_team_cards (referee_id, epoca)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtc_media ON referee_team_cards (media_amarelos DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtc_faltas ON referee_team_cards (media_faltas DESC)")
        conn.commit()
        records = df.drop(columns=["arbitro","equipa"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_BY_SEASON, records)
        conn.commit()
    log.info("Guardados %d registos em referee_team_cards", len(df))


def save_total(df: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_team_cards_total (
                referee_id                  INTEGER NOT NULL,
                team_id                     INTEGER NOT NULL,
                epocas                      INTEGER,
                jogos_total                 INTEGER,
                amarelos_total              INTEGER,
                vermelhos_total             INTEGER,
                faltas_total                INTEGER,
                media_amarelos              NUMERIC(6,3),
                media_vermelhos             NUMERIC(6,4),
                media_faltas                NUMERIC(6,3),
                desvio_vs_media_arbitro     NUMERIC(6,3),
                percentil_amarelos          NUMERIC(6,1),
                ranking_amarelos            INTEGER,
                desvio_faltas_vs_media      NUMERIC(6,3),
                percentil_faltas            NUMERIC(6,1),
                ranking_faltas              INTEGER,
                amarelos_por_falta          NUMERIC(6,4),
                PRIMARY KEY (referee_id, team_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtct_ranking ON referee_team_cards_total (referee_id, ranking_amarelos)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtct_ranking_faltas ON referee_team_cards_total (referee_id, ranking_faltas)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtct_apf ON referee_team_cards_total (amarelos_por_falta DESC)")
        conn.commit()
        records = df.drop(columns=["arbitro","equipa"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_TOTAL, records)
        conn.commit()
    log.info("Guardados %d registos em referee_team_cards_total", len(df))


# ── Preview ───────────────────────────────────────────────────────────────────

def print_preview(total: pd.DataFrame, by_season: pd.DataFrame) -> None:
    """Mostra tabelas de resumo no terminal — amarelos e faltas."""

    print("\n" + "═"*90)
    print("TOP 3 EQUIPAS COM MAIS AMARELOS E MAIS FALTAS POR ÁRBITRO (acumulado 7 épocas)")
    print("═"*90)

    for ref_id, grp in total.sort_values(
        ["arbitro","ranking_amarelos"]
    ).groupby(["referee_id","arbitro"], sort=False):
        ref_name    = grp["arbitro"].iloc[0]
        media_y     = grp["media_amarelos"].mean()
        media_f     = grp["media_faltas"].mean()
        top3_y      = grp.nsmallest(3, "ranking_amarelos")
        top3_f      = grp.nsmallest(3, "ranking_faltas")

        print(f"\n  {ref_name}  (médias gerais: {media_y:.2f} amarelos/jogo | {media_f:.1f} faltas/jogo)")
        print(f"  {'─'*88}")
        print(f"  {'AMARELOS — mais cartejadas':<44}  {'FALTAS — mais faltas assinaladas':<44}")
        print(f"  {'Equipa':<20} {'Jogos':>5} {'Média':>7} {'Desvio':>8}    {'Equipa':<20} {'Jogos':>5} {'Média':>7} {'Desvio':>8}")
        for (_, ry), (_, rf) in zip(top3_y.iterrows(), top3_f.iterrows()):
            print(f"  {ry['equipa']:<20} {ry['jogos_total']:>5} {ry['media_amarelos']:>7.3f} {ry['desvio_vs_media_arbitro']:>+8.3f}"
                  f"    {rf['equipa']:<20} {rf['jogos_total']:>5} {rf['media_faltas']:>7.1f} {rf['desvio_faltas_vs_media']:>+8.2f}")

    # Desvios extremos — faltas
    print("\n" + "═"*90)
    print("DESVIOS EXTREMOS DE FALTAS — pares árbitro×equipa com desvio > +2 faltas/jogo")
    print("═"*90)
    ext_f = total[total["desvio_faltas_vs_media"] > 2.0].sort_values(
        "desvio_faltas_vs_media", ascending=False
    )
    if not ext_f.empty:
        print(f"\n  {'Árbitro':<32} {'Equipa':<22} {'Jogos':>5} {'Média faltas':>13} {'Desvio':>8} {'Pct':>5}")
        print(f"  {'─'*32} {'─'*22} {'─'*5} {'─'*13} {'─'*8} {'─'*5}")
        for _, r in ext_f.iterrows():
            print(f"  {r['arbitro']:<32} {r['equipa']:<22} "
                  f"{r['jogos_total']:>5} {r['media_faltas']:>13.1f} "
                  f"{r['desvio_faltas_vs_media']:>+8.2f} {r['percentil_faltas']:>4.0f}%")

    # Desvios extremos — amarelos
    print("\n" + "═"*90)
    print("DESVIOS EXTREMOS DE AMARELOS — pares árbitro×equipa com desvio > +0.8 amarelos/jogo")
    print("═"*90)
    ext_y = total[total["desvio_vs_media_arbitro"] > 0.8].sort_values(
        "desvio_vs_media_arbitro", ascending=False
    )
    if not ext_y.empty:
        print(f"\n  {'Árbitro':<32} {'Equipa':<22} {'Jogos':>5} {'Média':>7} {'Desvio':>8} {'Pct':>5}")
        print(f"  {'─'*32} {'─'*22} {'─'*5} {'─'*7} {'─'*8} {'─'*5}")
        for _, r in ext_y.iterrows():
            print(f"  {r['arbitro']:<32} {r['equipa']:<22} "
                  f"{r['jogos_total']:>5} {r['media_amarelos']:>7.3f} "
                  f"{r['desvio_vs_media_arbitro']:>+8.3f} {r['percentil_amarelos']:>4.0f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def save_faltas_por_amarelo(df: pd.DataFrame) -> None:
    """Guarda a tabela com faltas_por_amarelo para todas as equipas (sem threshold)."""
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_team_severity (
                referee_id          INTEGER NOT NULL,
                team_id             INTEGER NOT NULL,
                jogos               INTEGER,
                amarelos_total      INTEGER,
                faltas_total        INTEGER,
                media_amarelos      NUMERIC(6,3),
                media_faltas        NUMERIC(6,2),
                faltas_por_amarelo  NUMERIC(8,2),  -- 1 amarelo a cada X faltas
                PRIMARY KEY (referee_id, team_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rts_fpa
                ON referee_team_severity (referee_id, faltas_por_amarelo DESC)
        """)
        conn.commit()
        records = df.drop(columns=["arbitro","equipa"], errors="ignore").to_dict("records")
        cur.executemany("""
            INSERT INTO referee_team_severity
                (referee_id, team_id, jogos, amarelos_total, faltas_total,
                 media_amarelos, media_faltas, faltas_por_amarelo)
            VALUES
                (%(referee_id)s, %(team_id)s, %(jogos)s, %(amarelos_total)s,
                 %(faltas_total)s, %(media_amarelos)s, %(media_faltas)s,
                 %(faltas_por_amarelo)s)
            ON CONFLICT (referee_id, team_id) DO UPDATE SET
                jogos               = EXCLUDED.jogos,
                amarelos_total      = EXCLUDED.amarelos_total,
                faltas_total        = EXCLUDED.faltas_total,
                media_amarelos      = EXCLUDED.media_amarelos,
                media_faltas        = EXCLUDED.media_faltas,
                faltas_por_amarelo  = EXCLUDED.faltas_por_amarelo
        """, records)
        conn.commit()
    log.info("Guardados %d registos em referee_team_severity", len(df))


def run() -> None:
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY, {"seasons": SEASONS_COMPLETE})
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    log.info("Dados carregados: %d linhas (árbitro×equipa×época)", len(df))

    for col in ["media_amarelos", "media_vermelhos", "media_faltas",
                "amarelos_total", "vermelhos_total", "faltas_total",
                "jogos", "media_geral_amarelos_epoca", "media_geral_faltas_epoca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    log.info("A calcular estatísticas por época...")
    by_season = compute_by_season(df)
    save_by_season(by_season)

    log.info("A calcular estatísticas acumuladas (7 épocas completas)...")
    total = compute_total(df)
    save_total(total)

    # Calcular faltas_por_amarelo para TODAS as equipas (sem threshold mínimo)
    log.info("A calcular severidade (faltas/amarelo) para todas as equipas...")
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY_ALL_TEAMS, {"seasons": SEASONS_COMPLETE})
        rows_all = cur.fetchall()
    df_all = pd.DataFrame(rows_all)
    for col in ["jogos","amarelos_total","faltas_total","media_amarelos",
                "media_faltas","faltas_por_amarelo"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
    save_faltas_por_amarelo(df_all)

    print_preview(total, by_season)


if __name__ == "__main__":
    import dotenv
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])
    run()
