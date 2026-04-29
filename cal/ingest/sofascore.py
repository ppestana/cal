# cal/ingest/sofascore.py
# Enriquecimento de árbitros via Sofascore API não-oficial.
#
# Estratégia:
#   Para cada temporada → para cada jornada (1-34) → para cada jogo:
#     1. Obter event_id e árbitro via GET /event/{id}
#     2. Fazer match com o jogo já na DB por (data, equipa_casa, equipa_fora)
#     3. Upsert referee na tabela referees + referee_aliases
#     4. Actualizar matches.referee_id
#
# Uso:
#   from cal.ingest.sofascore import run
#   run()                    # todas as temporadas
#   run(['2023/24'])         # temporada específica

import time
import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from cal.db import get_cursor, upsert_referee, log_ingest

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

BASE_URL  = "https://api.sofascore.com/api/v1"
SOURCE    = "sofascore"
HEADERS   = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.sofascore.com",
}

# tournament_id=238 — Liga Portugal / Primeira Liga
TOURNAMENT_ID = 238
ROUNDS        = 34

# Mapa season_label → sofascore season_id
SEASON_IDS: dict[str, int] = {
    "2017/18": 13539,
    "2018/19": 17714,
    "2019/20": 24150,
    "2020/21": 32456,
    "2021/22": 37358,
    "2022/23": 42655,
    "2023/24": 52769,
    "2024/25": 63670,
    "2025/26": 77806,
    # ── Para adicionar épocas futuras ────────────────────────────────────────
    # Obter o season_id do Sofascore com o comando:
    #
    #   curl -s -H "Referer: https://www.sofascore.com" \
    #     "https://api.sofascore.com/api/v1/unique-tournament/238/seasons" \
    #     | python -m json.tool | grep -A2 '"name"'
    #
    # O tournament_id=238 é fixo para a Liga Portugal.
    # O season_id mais recente aparece primeiro na lista.
}

# Normalização de nomes de equipas Sofascore → nomes no football-data.co.uk
# Adicionar entradas sempre que surjam novas variações.
TEAM_NAME_MAP: dict[str, str] = {
    "Sporting CP":          "Sporting",
    "FC Porto":             "Porto",
    "SL Benfica":           "Benfica",
    "SC Braga":             "Braga",
    "Sporting Braga":       "Braga",
    "Vitória SC":           "Guimaraes",
    "Vitória Guimarães":    "Guimaraes",
    "FC Vizela":            "Vizela",
    "FC Arouca":            "Arouca",
    "GD Chaves":            "Chaves",
    "CD Santa Clara":       "Santa Clara",
    "CF Estrela da Amadora": "Estrela",
    "SC Farense":           "Farense",
    "Gil Vicente FC":       "Gil Vicente",
    "CD Nacional":          "Nacional",
    "Rio Ave FC":           "Rio Ave",
    "Moreirense FC":        "Moreirense",
    "Boavista FC":          "Boavista",
    "FC Famalicão":         "Famalicao",
    "Portimonense SC":      "Portimonense",
    "Portimonense SAD":     "Portimonense",
    "Casa Pia AC":          "Casa Pia",
    "Estoril Praia":        "Estoril",
    "Paços de Ferreira":    "Pacos de Ferreira",
    "CD Tondela":           "Tondela",
    "CD Aves":              "Aves",
    "Belenenses SAD":       "Belenenses",
    "CF Os Belenenses":     "Belenenses",
    "Marítimo":             "Maritimo",
    "CS Marítimo":          "Maritimo",
    "FC Penafiel":          "Penafiel",
    "AVS Futebol SAD":      "AVS",
}

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3, backoff: float = 2.0) -> Optional[dict]:
    """GET com retry e backoff exponencial. Devolve None em caso de erro."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = backoff ** (attempt + 2)
                log.warning("Rate limit (429). Aguardar %.0fs ...", wait)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            log.warning("HTTP %d para %s", r.status_code, url)
        except requests.RequestException as e:
            log.warning("Erro request (tentativa %d/%d): %s", attempt + 1, retries, e)
            time.sleep(backoff ** attempt)
    return None

# ── Normalização ──────────────────────────────────────────────────────────────

def _normalise_team(name: str) -> str:
    """Converte nome de equipa Sofascore para o nome canónico da DB."""
    return TEAM_NAME_MAP.get(name, name)

def _ts_to_date(ts: int) -> str:
    """Converte Unix timestamp para string 'YYYY-MM-DD'."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

# ── Core: jornada → eventos ───────────────────────────────────────────────────

def _fetch_round_event_ids(season_id: int, round_num: int) -> list[int]:
    """Devolve lista de event_ids para uma jornada."""
    url  = f"{BASE_URL}/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/events/round/{round_num}"
    data = _get(url)
    if not data:
        return []
    return [e["id"] for e in data.get("events", [])]


def _fetch_event_detail(event_id: int) -> Optional[dict]:
    """Devolve detalhes completos de um evento (inclui árbitro)."""
    url  = f"{BASE_URL}/event/{event_id}"
    data = _get(url)
    return data.get("event") if data else None

# ── Core: match na DB ─────────────────────────────────────────────────────────

def _find_match_id(cursor, match_date: str, home_norm: str, away_norm: str) -> Optional[int]:
    """
    Encontra o match_id na DB pelo triple (data, equipa_casa, equipa_fora).
    Usa ILIKE para absorver pequenas variações de capitalização.
    """
    cursor.execute("""
        SELECT m.match_id
        FROM   matches m
        JOIN   teams th ON th.team_id = m.home_team_id
        JOIN   teams ta ON ta.team_id = m.away_team_id
        WHERE  m.match_date = %s
        AND    (th.name ILIKE %s OR EXISTS (
                    SELECT 1 FROM team_aliases a
                    WHERE a.team_id = th.team_id AND a.alias_name ILIKE %s
               ))
        AND    (ta.name ILIKE %s OR EXISTS (
                    SELECT 1 FROM team_aliases a
                    WHERE a.team_id = ta.team_id AND a.alias_name ILIKE %s
               ))
    """, (match_date, home_norm, home_norm, away_norm, away_norm))
    row = cursor.fetchone()
    return row[0] if row else None


def _update_match_referee(cursor, match_id: int, referee_id: int) -> None:
    cursor.execute(
        "UPDATE matches SET referee_id = %s WHERE match_id = %s AND (referee_id IS NULL OR referee_id = 1)",
        (referee_id, match_id)
    )

# ── Processamento de uma temporada ───────────────────────────────────────────

def _process_season(season_label: str, season_id: int) -> dict:
    """
    Processa uma temporada completa.
    Devolve dict com estatísticas: matched, unmatched, no_referee, errors.
    """
    stats = {"matched": 0, "unmatched": 0, "no_referee": 0, "errors": 0}
    log.info("Sofascore — a processar temporada %s (season_id=%d)", season_label, season_id)

    with get_cursor() as (conn, cur):
        for round_num in range(1, ROUNDS + 1):
            event_ids = _fetch_round_event_ids(season_id, round_num)
            if not event_ids:
                log.debug("  Jornada %d: sem eventos", round_num)
                continue

            log.info("  Jornada %2d: %d jogos", round_num, len(event_ids))

            for event_id in event_ids:
                try:
                    event = _fetch_event_detail(event_id)
                    if not event:
                        stats["errors"] += 1
                        continue

                    # ── Árbitro ──────────────────────────────────────────────
                    ref_data = event.get("referee")
                    if not ref_data:
                        stats["no_referee"] += 1
                        continue

                    ref_name  = ref_data["name"]
                    ref_zz_id = str(ref_data["id"])

                    # ── Equipas e data ───────────────────────────────────────
                    home_raw  = event["homeTeam"]["name"]
                    away_raw  = event["awayTeam"]["name"]
                    home_norm = _normalise_team(home_raw)
                    away_norm = _normalise_team(away_raw)
                    ts        = event.get("startTimestamp")
                    if not ts:
                        stats["errors"] += 1
                        continue
                    match_date = _ts_to_date(ts)

                    # ── Match na DB ──────────────────────────────────────────
                    match_id = _find_match_id(cur, match_date, home_norm, away_norm)
                    if not match_id:
                        log.debug(
                            "    [sem match] %s vs %s em %s",
                            home_norm, away_norm, match_date
                        )
                        stats["unmatched"] += 1
                        continue

                    # ── Upsert árbitro ───────────────────────────────────────
                    referee_id = upsert_referee(
                        cur,
                        name=ref_name,
                        source=SOURCE,
                        source_id=ref_zz_id,
                    )

                    # ── Actualizar match ─────────────────────────────────────
                    _update_match_referee(cur, match_id, referee_id)
                    stats["matched"] += 1

                    # Throttle gentil — evitar rate limit
                    time.sleep(0.15)

                except Exception as e:
                    log.exception("    Erro no evento %d: %s", event_id, e)
                    stats["errors"] += 1

            conn.commit()
            time.sleep(0.5)  # pausa entre jornadas

    return stats

# ── Entry point ───────────────────────────────────────────────────────────────

def run(seasons: Optional[list[str]] = None) -> None:
    """
    Enriquecer árbitros via Sofascore para as temporadas indicadas.
    Se seasons=None, processa todas as temporadas disponíveis.
    """
    targets = seasons if seasons else list(SEASON_IDS.keys())
    invalid = [s for s in targets if s not in SEASON_IDS]
    if invalid:
        raise ValueError(f"Temporadas inválidas: {invalid}. Válidas: {list(SEASON_IDS.keys())}")

    for label in targets:
        sid    = SEASON_IDS[label]
        start  = datetime.now()
        stats  = _process_season(label, sid)
        elapsed = (datetime.now() - start).seconds

        log.info(
            "Sofascore %s — matched=%d unmatched=%d no_referee=%d errors=%d [%ds]",
            label, stats["matched"], stats["unmatched"],
            stats["no_referee"], stats["errors"], elapsed
        )
        log_ingest(
            source=SOURCE,
            season_label=label,
            rows_fetched=stats["matched"] + stats["unmatched"] + stats["no_referee"],
            rows_inserted=stats["matched"],
            status="success" if stats["errors"] == 0 else "partial",
        )
