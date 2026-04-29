"""
cal/features/engineering.py
Fase 2 — Feature Engineering

Constrói a feature matrix por (match_id, team_id) a partir dos dados
já na base de dados. Guarda o resultado na tabela match_features.

Features geradas:
  Contexto do jogo:
    - is_home            : 1 se equipa da casa, 0 se visitante
    - matchday           : jornada (1-34)
    - ht_score_diff      : golos_casa - golos_fora ao intervalo (perspectiva da equipa)
    - ft_score_diff      : idem no final (para análise retrospectiva)

  Força das equipas (rolling, últimos 10 jogos antes do jogo):
    - team_points_r10    : pontos acumulados pela equipa nos últimos 10 jogos
    - opp_points_r10     : pontos acumulados pelo adversário nos últimos 10 jogos
    - team_goals_scored_r10  : média de golos marcados
    - team_goals_conceded_r10: média de golos sofridos

  Contexto do árbitro (rolling, últimos 20 jogos antes do jogo):
    - ref_yellows_pg_r20 : média de amarelos dados pelo árbitro por jogo
    - ref_reds_pg_r20    : média de vermelhos dados pelo árbitro por jogo
    - ref_fouls_pg_r20   : média de faltas assinaladas pelo árbitro por jogo
    - ref_games_total    : total de jogos do árbitro na base de dados até à data

  Variáveis alvo (observadas — para treino dos modelos):
    - yellow_cards       : cartões amarelos observados (da equipa neste jogo)
    - red_cards          : cartões vermelhos observados
    - fouls              : faltas observadas

Uso:
    python -m cal.features.engineering          # calcular todas as temporadas
    python -m cal.features.engineering 2023/24  # temporada específica
"""

import logging
import math
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np

from cal.db import get_cursor

log = logging.getLogger(__name__)


# ── SQL: carregar dados base ──────────────────────────────────────────────────

QUERY_MATCHES = """
SELECT
    m.match_id,
    m.match_date,
    s.label          AS season,
    m.season_id,
    m.home_team_id,
    m.away_team_id,
    m.referee_id,
    m.home_goals,
    m.away_goals,
    m.ht_home_goals,
    m.ht_away_goals,
    th.name          AS home_team,
    ta.name          AS away_team,
    r.name           AS referee,
    -- home stats
    ms_h.fouls       AS home_fouls,
    ms_h.yellow_cards AS home_yellows,
    ms_h.red_cards   AS home_reds,
    -- away stats
    ms_a.fouls       AS away_fouls,
    ms_a.yellow_cards AS away_yellows,
    ms_a.red_cards   AS away_reds
FROM matches m
JOIN seasons s USING (season_id)
JOIN teams th ON th.team_id = m.home_team_id
JOIN teams ta ON ta.team_id = m.away_team_id
LEFT JOIN referees r USING (referee_id)
LEFT JOIN match_stats ms_h
    ON ms_h.match_id = m.match_id AND ms_h.team_id = m.home_team_id
LEFT JOIN match_stats ms_a
    ON ms_a.match_id = m.match_id AND ms_a.team_id = m.away_team_id
ORDER BY m.match_date, m.match_id
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _points(home_goals: int, away_goals: int) -> tuple[int, int]:
    """Devolve (pontos_casa, pontos_fora) para um resultado."""
    if home_goals > away_goals:
        return 3, 0
    elif home_goals == away_goals:
        return 1, 1
    else:
        return 0, 3


def _rolling_team_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Para cada (match_id, team_id), calcula rolling stats dos últimos `window`
    jogos ANTERIORES ao jogo actual.

    Devolve DataFrame com colunas:
        match_id, team_id, team_points_r10, opp_points_r10,
        team_goals_scored_r10, team_goals_conceded_r10
    """
    # Construir série temporal por equipa — uma linha por (jogo, equipa)
    rows = []
    for _, m in df.iterrows():
        # Perspectiva equipa da casa
        hp, ap = _points(m['home_goals'] or 0, m['away_goals'] or 0)
        rows.append({
            'match_id':      m['match_id'],
            'match_date':    m['match_date'],
            'team_id':       m['home_team_id'],
            'opp_id':        m['away_team_id'],
            'points':        hp,
            'opp_points':    ap,
            'goals_scored':  m['home_goals'] or 0,
            'goals_conceded':m['away_goals'] or 0,
        })
        # Perspectiva equipa visitante
        rows.append({
            'match_id':      m['match_id'],
            'match_date':    m['match_date'],
            'team_id':       m['away_team_id'],
            'opp_id':        m['home_team_id'],
            'points':        ap,
            'opp_points':    hp,
            'goals_scored':  m['away_goals'] or 0,
            'goals_conceded':m['home_goals'] or 0,
        })

    ts = pd.DataFrame(rows).sort_values(['team_id', 'match_date', 'match_id'])

    # Rolling shift(1) garante que só usamos jogos ANTERIORES
    result_rows = []
    for team_id, grp in ts.groupby('team_id'):
        grp = grp.sort_values('match_date').reset_index(drop=True)
        grp['team_points_r10'] = (
            grp['points'].shift(1).rolling(window, min_periods=1).sum()
        )
        grp['opp_points_r10'] = (
            grp['opp_points'].shift(1).rolling(window, min_periods=1).sum()
        )
        grp['team_goals_scored_r10'] = (
            grp['goals_scored'].shift(1).rolling(window, min_periods=1).mean()
        )
        grp['team_goals_conceded_r10'] = (
            grp['goals_conceded'].shift(1).rolling(window, min_periods=1).mean()
        )
        result_rows.append(grp[[
            'match_id', 'team_id',
            'team_points_r10', 'opp_points_r10',
            'team_goals_scored_r10', 'team_goals_conceded_r10',
        ]])

    return pd.concat(result_rows, ignore_index=True)


def _rolling_referee_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Para cada match_id com árbitro, calcula rolling stats dos últimos `window`
    jogos do árbitro ANTERIORES ao jogo actual.

    Devolve DataFrame com colunas:
        match_id, ref_yellows_pg_r20, ref_reds_pg_r20,
        ref_fouls_pg_r20, ref_games_total
    """
    # Total de cartões e faltas por jogo (ambas as equipas)
    ref_df = df[df['referee_id'].notna() & (df['referee_id'] > 1)].copy()
    ref_df['total_yellows'] = (
        ref_df['home_yellows'].fillna(0) + ref_df['away_yellows'].fillna(0)
    )
    ref_df['total_reds'] = (
        ref_df['home_reds'].fillna(0) + ref_df['away_reds'].fillna(0)
    )
    ref_df['total_fouls'] = (
        ref_df['home_fouls'].fillna(0) + ref_df['away_fouls'].fillna(0)
    )

    ref_df = ref_df.sort_values(['referee_id', 'match_date', 'match_id'])

    result_rows = []
    for ref_id, grp in ref_df.groupby('referee_id'):
        grp = grp.sort_values('match_date').reset_index(drop=True)
        grp['ref_games_total'] = grp.index  # jogos anteriores (shift implícito)
        grp['ref_yellows_pg_r20'] = (
            grp['total_yellows'].shift(1).rolling(window, min_periods=1).mean()
        )
        grp['ref_reds_pg_r20'] = (
            grp['total_reds'].shift(1).rolling(window, min_periods=1).mean()
        )
        grp['ref_fouls_pg_r20'] = (
            grp['total_fouls'].shift(1).rolling(window, min_periods=1).mean()
        )
        result_rows.append(grp[[
            'match_id',
            'ref_yellows_pg_r20', 'ref_reds_pg_r20',
            'ref_fouls_pg_r20', 'ref_games_total',
        ]])

    return pd.concat(result_rows, ignore_index=True)


def _matchday(df: pd.DataFrame) -> pd.Series:
    """
    Estima a jornada por temporada ordenando os jogos por data e
    atribuindo jornada = rank da data dentro da temporada (9 jogos/jornada).
    """
    df = df.sort_values(['season_id', 'match_date', 'match_id'])
    df['_date_rank'] = df.groupby('season_id')['match_date'].rank(
        method='dense'
    ).astype(int)
    # Jornada ≈ ceil(rank / 9) — aproximação; algumas jornadas têm jogos em datas diferentes
    df['matchday'] = np.ceil(df['_date_rank'] / 9).astype(int)
    return df['matchday']


# ── Build feature matrix ──────────────────────────────────────────────────────

def build(seasons: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Constrói a feature matrix completa para as temporadas indicadas.
    Se seasons=None, usa todos os dados disponíveis.

    Devolve DataFrame com uma linha por (match_id, team_id).
    """
    log.info("A carregar dados da base de dados...")

    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY_MATCHES)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    log.info("Jogos carregados: %d", len(df))

    if seasons:
        df = df[df['season'].isin(seasons)].copy()
        log.info("Filtrado para %d jogos (%s)", len(df), seasons)

    # ── Matchday ─────────────────────────────────────────────────────────────
    df['matchday'] = _matchday(df)

    # ── Score diff ao intervalo ───────────────────────────────────────────────
    df['ht_score_diff_home'] = (
        df['ht_home_goals'].fillna(0) - df['ht_away_goals'].fillna(0)
    )
    df['ft_score_diff_home'] = (
        df['home_goals'].fillna(0) - df['away_goals'].fillna(0)
    )

    # ── Rolling team stats ────────────────────────────────────────────────────
    log.info("A calcular rolling team stats (window=10)...")
    # Usar TODOS os jogos (não filtrados) para ter histórico completo
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY_MATCHES)
        all_rows = cur.fetchall()
    df_all = pd.DataFrame(all_rows)

    team_rolling = _rolling_team_stats(df_all, window=10)

    # ── Rolling referee stats ─────────────────────────────────────────────────
    log.info("A calcular rolling referee stats (window=20)...")
    ref_rolling = _rolling_referee_stats(df_all, window=20)

    # ── Expandir para formato (match_id, team_id) ─────────────────────────────
    log.info("A construir feature matrix...")

    feature_rows = []
    for _, m in df.iterrows():
        for is_home in [True, False]:
            team_id = m['home_team_id'] if is_home else m['away_team_id']
            opp_id  = m['away_team_id'] if is_home else m['home_team_id']

            # Score diff na perspectiva desta equipa
            ht_diff = m['ht_score_diff_home'] * (1 if is_home else -1)
            ft_diff = m['ft_score_diff_home'] * (1 if is_home else -1)

            # Stats observadas
            yellows = m['home_yellows'] if is_home else m['away_yellows']
            reds    = m['home_reds']    if is_home else m['away_reds']
            fouls   = m['home_fouls']   if is_home else m['away_fouls']

            row = {
                'match_id':    m['match_id'],
                'team_id':     team_id,
                'opp_id':      opp_id,
                'season':      m['season'],
                'match_date':  m['match_date'],
                'referee_id':  m['referee_id'],
                'matchday':    m['matchday'],
                'is_home':     int(is_home),
                'ht_score_diff': ht_diff,
                'ft_score_diff': ft_diff,
                # alvo
                'yellow_cards': yellows,
                'red_cards':    reds,
                'fouls':        fouls,
            }
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows)

    # Merge team rolling stats
    features = features.merge(
        team_rolling.rename(columns={'team_id': 'team_id'}),
        on=['match_id', 'team_id'],
        how='left',
    )

    # Merge ref rolling stats
    features = features.merge(
        ref_rolling,
        on='match_id',
        how='left',
    )

    # Valores em falta nas rolling (primeiros jogos de cada equipa/árbitro)
    rolling_cols = [
        'team_points_r10', 'opp_points_r10',
        'team_goals_scored_r10', 'team_goals_conceded_r10',
        'ref_yellows_pg_r20', 'ref_reds_pg_r20',
        'ref_fouls_pg_r20', 'ref_games_total',
    ]
    for col in rolling_cols:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())

    log.info(
        "Feature matrix construída: %d linhas × %d colunas",
        len(features), len(features.columns)
    )
    return features


def save_to_db(features: pd.DataFrame) -> None:
    """
    Guarda a feature matrix na tabela match_features.
    Cria a tabela se não existir.
    """
    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS match_features (
        match_id                  INTEGER NOT NULL,
        team_id                   INTEGER NOT NULL,
        opp_id                    INTEGER,
        season                    VARCHAR(10),
        match_date                DATE,
        referee_id                INTEGER,
        matchday                  SMALLINT,
        is_home                   SMALLINT,
        ht_score_diff             NUMERIC(4,1),
        ft_score_diff             NUMERIC(4,1),
        team_points_r10           NUMERIC(6,2),
        opp_points_r10            NUMERIC(6,2),
        team_goals_scored_r10     NUMERIC(6,3),
        team_goals_conceded_r10   NUMERIC(6,3),
        ref_yellows_pg_r20        NUMERIC(6,3),
        ref_reds_pg_r20           NUMERIC(6,3),
        ref_fouls_pg_r20          NUMERIC(6,3),
        ref_games_total           INTEGER,
        yellow_cards              SMALLINT,
        red_cards                 SMALLINT,
        fouls                     SMALLINT,
        PRIMARY KEY (match_id, team_id)
    );
    CREATE INDEX IF NOT EXISTS idx_mf_referee
        ON match_features (referee_id, match_date);
    CREATE INDEX IF NOT EXISTS idx_mf_team
        ON match_features (team_id, match_date);
    """

    UPSERT = """
    INSERT INTO match_features (
        match_id, team_id, opp_id, season, match_date, referee_id,
        matchday, is_home, ht_score_diff, ft_score_diff,
        team_points_r10, opp_points_r10,
        team_goals_scored_r10, team_goals_conceded_r10,
        ref_yellows_pg_r20, ref_reds_pg_r20,
        ref_fouls_pg_r20, ref_games_total,
        yellow_cards, red_cards, fouls
    ) VALUES (
        %(match_id)s, %(team_id)s, %(opp_id)s, %(season)s, %(match_date)s,
        %(referee_id)s, %(matchday)s, %(is_home)s,
        %(ht_score_diff)s, %(ft_score_diff)s,
        %(team_points_r10)s, %(opp_points_r10)s,
        %(team_goals_scored_r10)s, %(team_goals_conceded_r10)s,
        %(ref_yellows_pg_r20)s, %(ref_reds_pg_r20)s,
        %(ref_fouls_pg_r20)s, %(ref_games_total)s,
        %(yellow_cards)s, %(red_cards)s, %(fouls)s
    )
    ON CONFLICT (match_id, team_id) DO UPDATE SET
        team_points_r10         = EXCLUDED.team_points_r10,
        opp_points_r10          = EXCLUDED.opp_points_r10,
        team_goals_scored_r10   = EXCLUDED.team_goals_scored_r10,
        team_goals_conceded_r10 = EXCLUDED.team_goals_conceded_r10,
        ref_yellows_pg_r20      = EXCLUDED.ref_yellows_pg_r20,
        ref_reds_pg_r20         = EXCLUDED.ref_reds_pg_r20,
        ref_fouls_pg_r20        = EXCLUDED.ref_fouls_pg_r20,
        ref_games_total         = EXCLUDED.ref_games_total
    """

    with get_cursor() as (conn, cur):
        cur.execute(CREATE_TABLE)
        conn.commit()

        int_cols = ['match_id', 'team_id', 'opp_id', 'referee_id',
                    'matchday', 'is_home', 'yellow_cards', 'red_cards', 'fouls',
                    'ref_games_total']
        for col in int_cols:
            if col in features.columns:
                features[col] = features[col].apply(
                    lambda x: None if (
                        x is None or (isinstance(x, float) and math.isnan(x))
                    ) else int(x)
                )
        records = features.where(pd.notnull(features), None).to_dict('records')
        cur.executemany(UPSERT, records)
        conn.commit()

    log.info("Guardadas %d linhas em match_features", len(features))


if __name__ == "__main__":
    import sys
    import dotenv
    import logging
    from rich.logging import RichHandler

    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False)],
    )

    seasons_arg = sys.argv[1:] or None
    features = build(seasons=seasons_arg)
    save_to_db(features)

    # Preview
    print("\nPrimeiras 3 linhas da feature matrix:")
    print(features[[
        'match_id', 'season', 'match_date', 'is_home', 'matchday',
        'ht_score_diff', 'team_points_r10', 'ref_yellows_pg_r20',
        'yellow_cards', 'fouls'
    ]].head(3).to_string(index=False))
