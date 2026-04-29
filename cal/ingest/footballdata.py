"""
cal/ingest/footballdata.py
Ingestion pipeline — football-data.co.uk
Primeira Liga Portuguesa (P1.csv)

Disponibiliza por jogo:
  - árbitro
  - golos (FT + HT)
  - faltas (HF / AF)
  - cartões amarelos (HY / AY)
  - cartões vermelhos (HR / AR)
  - remates, remates enquadrados, cantos

Seasons disponíveis: 2007/08 → presente
URL: https://www.football-data.co.uk/mmz4281/{SSYY}/P1.csv
     Ex: 2324 = 2023/24
"""

import io
import logging
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

from cal.db import (
    get_cursor,
    log_ingest,
    row_hash,
    upsert_referee,
    upsert_season,
    upsert_team,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/P1.csv"
SOURCE = "football-data"

# Temporadas disponíveis com dados de árbitro
# (anteriores a 1718 têm dados incompletos de disciplina)
AVAILABLE_SEASONS = {
    "2017/18": "1718",
    "2018/19": "1819",
    "2019/20": "1920",
    "2020/21": "2021",
    "2021/22": "2122",
    "2022/23": "2223",
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526",
    # ── Para adicionar épocas futuras ────────────────────────────────────────
    # Padrão do código: "AAAA/BB": "AABB"  (ex: 2026/27 → "2627")
    # O CSV aparece tipicamente em Agosto, algumas semanas após o início.
    # Verificar disponibilidade em:
    #   https://www.football-data.co.uk/portugalm.php
}

# Colunas esperadas e os seus tipos
EXPECTED_COLS = {
    "Date": str,
    "HomeTeam": str,
    "AwayTeam": str,
    "FTHG": "Int64",   # Full-time home goals
    "FTAG": "Int64",   # Full-time away goals
    "HTHG": "Int64",   # Half-time home goals
    "HTAG": "Int64",   # Half-time away goals
    "Referee": str,
    "HF": "Int64",     # Home fouls
    "AF": "Int64",     # Away fouls
    "HY": "Int64",     # Home yellow cards
    "AY": "Int64",     # Away yellow cards
    "HR": "Int64",     # Home red cards
    "AR": "Int64",     # Away red cards
    "HS": "Int64",     # Home shots
    "AS": "Int64",     # Away shots
    "HST": "Int64",    # Home shots on target
    "AST": "Int64",    # Away shots on target
    "HC": "Int64",     # Home corners
    "AC": "Int64",     # Away corners
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_season_csv(season_code: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Download CSV for one season; return DataFrame or None on failure."""
    url = BASE_URL.format(code=season_code)
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching {url} (attempt {attempt})")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # football-data sometimes returns empty file for future seasons
            if len(resp.content) < 200:
                logger.warning(f"Response too small for {url}, skipping.")
                return None
            df = pd.read_csv(
                io.StringIO(resp.text),
                encoding="utf-8",
                on_bad_lines="warn",
            )
            # Drop fully empty rows (common at end of in-progress seasons)
            df = df.dropna(how="all")
            logger.info(f"Fetched {len(df)} rows for season code {season_code}")
            return df
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Season {season_code} not yet available (404).")
                return None
            logger.error(f"HTTP error {e.response.status_code} for {url}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_date(raw: str) -> Optional[date]:
    """Parse DD/MM/YY or DD/MM/YYYY."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    logger.warning(f"Unrecognised date format: {raw!r}")
    return None


def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and cast expected columns; add missing optional ones as NaN.
    football-data.co.uk sometimes omits columns for older seasons.
    """
    for col, dtype in EXPECTED_COLS.items():
        if col not in df.columns:
            df[col] = pd.NA
        elif dtype != str:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Filter rows with minimum required fields
    required = ["Date", "HomeTeam", "AwayTeam"]
    df = df.dropna(subset=required)
    df = df[df["HomeTeam"].str.strip() != ""]
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_season(df: pd.DataFrame, season_label: str) -> dict:
    """Insert one season's data into the DB. Returns stats dict."""
    stats = {"fetched": len(df), "inserted": 0, "skipped": 0}

    with get_cursor() as (conn, cur):
            season_id = upsert_season(cur, season_label)

            for _, row in df.iterrows():
                match_date = parse_date(str(row["Date"]))
                if match_date is None:
                    stats["skipped"] += 1
                    continue

                home_name = str(row["HomeTeam"]).strip()
                away_name = str(row["AwayTeam"]).strip()

                # Skip rows without team names
                if not home_name or not away_name:
                    stats["skipped"] += 1
                    continue

                home_id = upsert_team(cur, home_name, SOURCE)
                away_id = upsert_team(cur, away_name, SOURCE)

                # Referee (may be missing for some older rows)
                referee_raw = str(row.get("Referee", "")).strip()
                referee_id = None
                if referee_raw and referee_raw.lower() not in ("nan", ""):
                    referee_id = upsert_referee(cur, referee_raw, SOURCE)

                # Deduplication hash
                hash_dict = {
                    "date": str(match_date),
                    "home": home_name,
                    "away": away_name,
                    "fthg": str(row.get("FTHG")),
                    "ftag": str(row.get("FTAG")),
                }
                rhash = row_hash(hash_dict)

                # Upsert match
                cur.execute(
                    """
                    INSERT INTO matches
                        (season_id, match_date, home_team_id, away_team_id,
                         home_goals, away_goals, ht_home_goals, ht_away_goals,
                         referee_id, source, source_row_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_date, home_team_id, away_team_id)
                    DO UPDATE SET
                        referee_id      = EXCLUDED.referee_id,
                        home_goals      = EXCLUDED.home_goals,
                        away_goals      = EXCLUDED.away_goals,
                        ht_home_goals   = EXCLUDED.ht_home_goals,
                        ht_away_goals   = EXCLUDED.ht_away_goals,
                        source_row_hash = EXCLUDED.source_row_hash
                    RETURNING match_id, (xmax = 0) AS was_inserted
                    """,
                    (
                        season_id,
                        match_date,
                        home_id,
                        away_id,
                        _int(row.get("FTHG")),
                        _int(row.get("FTAG")),
                        _int(row.get("HTHG")),
                        _int(row.get("HTAG")),
                        referee_id,
                        SOURCE,
                        rhash,
                    ),
                )
                match_id, was_inserted = cur.fetchone()

                if was_inserted:
                    stats["inserted"] += 1
                else:
                    stats["skipped"] += 1

                # Upsert match_stats (home)
                _upsert_stats(
                    cur, match_id, home_id, is_home=True,
                    fouls=row.get("HF"), yellows=row.get("HY"),
                    reds=row.get("HR"), shots=row.get("HS"),
                    shots_ot=row.get("HST"), corners=row.get("HC"),
                    goals=row.get("FTHG"),
                )
                # Upsert match_stats (away)
                _upsert_stats(
                    cur, match_id, away_id, is_home=False,
                    fouls=row.get("AF"), yellows=row.get("AY"),
                    reds=row.get("AR"), shots=row.get("AS"),
                    shots_ot=row.get("AST"), corners=row.get("AC"),
                    goals=row.get("FTAG"),
                )

            log_ingest(
                source=SOURCE, season_label=season_label,
                rows_fetched=stats["fetched"], rows_inserted=stats["inserted"],
                rows_skipped=stats["skipped"], status="success", cur=cur,
            )

    return stats


def _upsert_stats(cur, match_id, team_id, is_home,
                  fouls, yellows, reds, shots, shots_ot, corners, goals):
    cur.execute(
        """
        INSERT INTO match_stats
            (match_id, team_id, is_home, fouls, yellow_cards, red_cards,
             shots, shots_on_target, corners, goals)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id, team_id) DO UPDATE SET
            fouls           = EXCLUDED.fouls,
            yellow_cards    = EXCLUDED.yellow_cards,
            red_cards       = EXCLUDED.red_cards,
            shots           = EXCLUDED.shots,
            shots_on_target = EXCLUDED.shots_on_target,
            corners         = EXCLUDED.corners,
            goals           = EXCLUDED.goals
        """,
        (
            match_id, team_id, is_home,
            _int(fouls), _int(yellows), _int(reds),
            _int(shots), _int(shots_ot), _int(corners), _int(goals),
        ),
    )


def _int(val) -> Optional[int]:
    """Safely convert pandas value to int, returning None for NaN."""
    try:
        v = int(val)
        return v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seasons: list[str] | None = None) -> None:
    """
    Ingest one or more seasons.

    Args:
        seasons: list of season labels like ['2023/24', '2022/23'].
                 If None, ingest ALL available seasons.
    """
    target = seasons or list(AVAILABLE_SEASONS.keys())
    logger.info(f"Starting football-data ingestion for {len(target)} season(s).")

    for label in target:
        code = AVAILABLE_SEASONS.get(label)
        if not code:
            logger.warning(f"Unknown season label: {label!r}. Skipping.")
            continue

        df = fetch_season_csv(code)
        if df is None:
            logger.warning(f"No data for {label}, skipping.")
            log_ingest(source=SOURCE, season_label=label,
                       rows_fetched=0, rows_inserted=0, rows_skipped=0,
                       status="error", error_msg="fetch returned None")
            continue

        df = normalise_df(df)
        stats = load_season(df, label)
        logger.info(
            f"[{label}] fetched={stats['fetched']} "
            f"inserted={stats['inserted']} skipped={stats['skipped']}"
        )

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    import sys
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    seasons_arg = sys.argv[1:] or None   # e.g. python -m cal.ingest.footballdata "2023/24"
    run(seasons_arg)
