-- =============================================================
-- CAL — Criticar a Arbitragem Legalmente
-- PostgreSQL Schema v1.0
-- Primeira Liga Portuguesa
-- =============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS unaccent;

-- =============================================================
-- LOOKUP TABLES
-- =============================================================

CREATE TABLE IF NOT EXISTS seasons (
    season_id   SERIAL PRIMARY KEY,
    label       VARCHAR(10) NOT NULL UNIQUE,   -- '2023/24'
    start_year  SMALLINT NOT NULL,
    end_year    SMALLINT NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams (
    team_id     SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    short_name  VARCHAR(30),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Normalização de nomes de equipas entre fontes
CREATE TABLE IF NOT EXISTS team_aliases (
    alias_id    SERIAL PRIMARY KEY,
    team_id     INTEGER NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    source      VARCHAR(30) NOT NULL,           -- 'football-data', 'fbref', 'sofascore'
    alias_name  VARCHAR(100) NOT NULL,
    UNIQUE (source, alias_name)
);

CREATE TABLE IF NOT EXISTS referees (
    referee_id  SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,   -- nome canónico (normalizado)
    nationality VARCHAR(50) DEFAULT 'Portuguese',
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Normalização de nomes de árbitros entre fontes
CREATE TABLE IF NOT EXISTS referee_aliases (
    alias_id    SERIAL PRIMARY KEY,
    referee_id  INTEGER NOT NULL REFERENCES referees(referee_id) ON DELETE CASCADE,
    source      VARCHAR(30) NOT NULL,
    alias_name  VARCHAR(100) NOT NULL,
    UNIQUE (source, alias_name)
);

-- =============================================================
-- MATCH DATA
-- =============================================================

CREATE TABLE IF NOT EXISTS matches (
    match_id        SERIAL PRIMARY KEY,
    season_id       INTEGER NOT NULL REFERENCES seasons(season_id),
    match_date      DATE NOT NULL,
    matchday        SMALLINT,
    home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals      SMALLINT,
    away_goals      SMALLINT,
    ht_home_goals   SMALLINT,                  -- intervalo
    ht_away_goals   SMALLINT,
    referee_id      INTEGER REFERENCES referees(referee_id),
    stadium         VARCHAR(100),
    source          VARCHAR(30) NOT NULL,       -- 'football-data', 'fbref'
    source_row_hash VARCHAR(64),               -- SHA256 da linha raw (deduplicação)
    ingested_at     TIMESTAMP DEFAULT NOW(),
    UNIQUE (match_date, home_team_id, away_team_id)
);

-- =============================================================
-- MATCH STATISTICS (aggregate por equipa por jogo)
-- =============================================================

CREATE TABLE IF NOT EXISTS match_stats (
    stat_id         SERIAL PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    is_home         BOOLEAN NOT NULL,
    -- Disciplina
    fouls           SMALLINT,
    yellow_cards    SMALLINT,
    red_cards       SMALLINT,
    -- Jogo
    shots           SMALLINT,
    shots_on_target SMALLINT,
    corners         SMALLINT,
    -- Expected (quando disponível via FBref)
    xg              NUMERIC(5, 2),
    -- Resultado (calculado)
    goals           SMALLINT,
    UNIQUE (match_id, team_id)
);

-- =============================================================
-- REFEREE DECISIONS (decisão individual por jogo)
-- Fonte primária: football-data.co.uk agrega por jogo;
-- Linhas aqui resultam do unpack dos totais de match_stats.
-- Quando disponível event-level (Sportmonks), inserir directo.
-- =============================================================

CREATE TABLE IF NOT EXISTS referee_decisions (
    decision_id     SERIAL PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    referee_id      INTEGER NOT NULL REFERENCES referees(referee_id),
    decision_type   VARCHAR(30) NOT NULL,   -- 'yellow_card','red_card','penalty','foul'
    team_id         INTEGER REFERENCES teams(team_id),  -- equipa penalizada
    minute          SMALLINT,               -- NULL quando só temos agregados
    var_involved    BOOLEAN DEFAULT FALSE,
    notes           TEXT,
    source          VARCHAR(30),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- =============================================================
-- EXPECTED PROBABILITIES (output dos modelos)
-- =============================================================

CREATE TABLE IF NOT EXISTS expected_probabilities (
    prob_id             SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id             INTEGER NOT NULL REFERENCES teams(team_id),
    is_home             BOOLEAN NOT NULL,
    model_version       VARCHAR(20) NOT NULL DEFAULT 'v1.0',
    -- Probabilidades esperadas (soma ao longo do jogo)
    expected_yellows    NUMERIC(8, 4),
    expected_reds       NUMERIC(8, 4),
    expected_penalties  NUMERIC(8, 4),
    expected_fouls      NUMERIC(8, 4),
    computed_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE (match_id, team_id, model_version)
);

-- =============================================================
-- REFEREE BIAS SCORES (output do bias engine)
-- =============================================================

CREATE TABLE IF NOT EXISTS referee_bias_scores (
    score_id                SERIAL PRIMARY KEY,
    referee_id              INTEGER NOT NULL REFERENCES referees(referee_id),
    season_id               INTEGER NOT NULL REFERENCES seasons(season_id),
    model_version           VARCHAR(20) NOT NULL DEFAULT 'v1.0',
    matches_refereed        SMALLINT,
    -- Cartões amarelos
    expected_home_yellows   NUMERIC(8, 4),
    actual_home_yellows     SMALLINT,
    expected_away_yellows   NUMERIC(8, 4),
    actual_away_yellows     SMALLINT,
    yellow_home_bias_z      NUMERIC(8, 4),
    yellow_away_bias_z      NUMERIC(8, 4),
    yellow_diff_bias_z      NUMERIC(8, 4),   -- (home - away) vs esperado
    -- Cartões vermelhos
    expected_home_reds      NUMERIC(8, 4),
    actual_home_reds        SMALLINT,
    expected_away_reds      NUMERIC(8, 4),
    actual_away_reds        SMALLINT,
    red_home_bias_z         NUMERIC(8, 4),
    red_away_bias_z         NUMERIC(8, 4),
    red_diff_bias_z         NUMERIC(8, 4),
    -- Penáltis (quando disponível)
    expected_home_penalties NUMERIC(8, 4),
    actual_home_penalties   SMALLINT,
    expected_away_penalties NUMERIC(8, 4),
    actual_away_penalties   SMALLINT,
    penalty_home_bias_z     NUMERIC(8, 4),
    penalty_away_bias_z     NUMERIC(8, 4),
    -- Score composto de suspeição
    suspicion_score         NUMERIC(8, 4),
    computed_at             TIMESTAMP DEFAULT NOW(),
    UNIQUE (referee_id, season_id, model_version)
);

-- =============================================================
-- CLUB-REFEREE INTERACTION BIAS
-- Coeficientes do modelo de regressão: árbitro X equipa
-- =============================================================

CREATE TABLE IF NOT EXISTS referee_team_bias (
    bias_id         SERIAL PRIMARY KEY,
    referee_id      INTEGER NOT NULL REFERENCES referees(referee_id),
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    season_id       INTEGER REFERENCES seasons(season_id),   -- NULL = all seasons
    decision_type   VARCHAR(30) NOT NULL,
    interaction_coef NUMERIC(10, 6),        -- coeficiente do termo de interação
    p_value         NUMERIC(10, 6),         -- p-value do coeficiente
    n_observations  SMALLINT,
    significant     BOOLEAN GENERATED ALWAYS AS (p_value < 0.05) STORED,
    model_version   VARCHAR(20) NOT NULL DEFAULT 'v1.0',
    computed_at     TIMESTAMP DEFAULT NOW(),
    UNIQUE (referee_id, team_id, season_id, decision_type, model_version)
);

-- =============================================================
-- INGEST LOG (auditoria)
-- =============================================================

CREATE TABLE IF NOT EXISTS ingest_log (
    log_id          SERIAL PRIMARY KEY,
    source          VARCHAR(30) NOT NULL,
    season_label    VARCHAR(10),
    rows_fetched    INTEGER,
    rows_inserted   INTEGER,
    rows_skipped    INTEGER,
    status          VARCHAR(20) NOT NULL,   -- 'success', 'error', 'partial'
    error_msg       TEXT,
    started_at      TIMESTAMP DEFAULT NOW(),
    finished_at     TIMESTAMP
);

-- =============================================================
-- INDEXES
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_matches_season     ON matches (season_id);
CREATE INDEX IF NOT EXISTS idx_matches_referee    ON matches (referee_id);
CREATE INDEX IF NOT EXISTS idx_matches_home_team  ON matches (home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team  ON matches (away_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_date       ON matches (match_date);

CREATE INDEX IF NOT EXISTS idx_match_stats_match  ON match_stats (match_id);
CREATE INDEX IF NOT EXISTS idx_match_stats_team   ON match_stats (team_id);

CREATE INDEX IF NOT EXISTS idx_decisions_match    ON referee_decisions (match_id);
CREATE INDEX IF NOT EXISTS idx_decisions_referee  ON referee_decisions (referee_id);
CREATE INDEX IF NOT EXISTS idx_decisions_type     ON referee_decisions (decision_type);

CREATE INDEX IF NOT EXISTS idx_bias_referee       ON referee_bias_scores (referee_id);
CREATE INDEX IF NOT EXISTS idx_bias_season        ON referee_bias_scores (season_id);
CREATE INDEX IF NOT EXISTS idx_bias_suspicion     ON referee_bias_scores (suspicion_score DESC);

CREATE INDEX IF NOT EXISTS idx_team_bias_sig      ON referee_team_bias (significant, p_value);

-- =============================================================
-- USEFUL VIEWS
-- =============================================================

-- Resumo de jogo com nomes legíveis
CREATE OR REPLACE VIEW v_matches AS
SELECT
    m.match_id,
    s.label             AS season,
    m.match_date,
    m.matchday,
    ht.name             AS home_team,
    at.name             AS away_team,
    m.home_goals,
    m.away_goals,
    r.name              AS referee,
    m.source
FROM matches m
JOIN seasons s        ON s.season_id    = m.season_id
JOIN teams   ht       ON ht.team_id     = m.home_team_id
JOIN teams   at       ON at.team_id     = m.away_team_id
LEFT JOIN referees r  ON r.referee_id   = m.referee_id;

-- Estatísticas de árbitro por temporada (sem modelos)
CREATE OR REPLACE VIEW v_referee_raw_stats AS
SELECT
    r.name                          AS referee,
    s.label                         AS season,
    COUNT(DISTINCT m.match_id)      AS matches,
    SUM(ms_h.yellow_cards)          AS total_home_yellows,
    SUM(ms_a.yellow_cards)          AS total_away_yellows,
    SUM(ms_h.red_cards)             AS total_home_reds,
    SUM(ms_a.red_cards)             AS total_away_reds,
    SUM(ms_h.fouls)                 AS total_home_fouls,
    SUM(ms_a.fouls)                 AS total_away_fouls,
    ROUND(AVG(ms_h.yellow_cards), 2) AS avg_home_yellows,
    ROUND(AVG(ms_a.yellow_cards), 2) AS avg_away_yellows,
    ROUND(AVG(ms_h.red_cards + ms_a.red_cards), 2) AS avg_reds_per_game
FROM matches m
JOIN seasons   s     ON s.season_id  = m.season_id
JOIN referees  r     ON r.referee_id = m.referee_id
LEFT JOIN match_stats ms_h ON ms_h.match_id = m.match_id AND ms_h.is_home = TRUE
LEFT JOIN match_stats ms_a ON ms_a.match_id = m.match_id AND ms_a.is_home = FALSE
GROUP BY r.name, s.label
ORDER BY r.name, s.label;

-- Ranking de suspeição (última temporada com scores calculados)
CREATE OR REPLACE VIEW v_suspicion_ranking AS
SELECT
    r.name              AS referee,
    s.label             AS season,
    bs.matches_refereed,
    bs.yellow_diff_bias_z,
    bs.red_diff_bias_z,
    bs.penalty_home_bias_z,
    bs.suspicion_score,
    RANK() OVER (PARTITION BY bs.season_id ORDER BY bs.suspicion_score DESC NULLS LAST) AS rank_in_season
FROM referee_bias_scores bs
JOIN referees r ON r.referee_id = bs.referee_id
JOIN seasons  s ON s.season_id  = bs.season_id
ORDER BY bs.season_id DESC, bs.suspicion_score DESC NULLS LAST;
