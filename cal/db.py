"""
cal/db.py — Database connection pool and helpers
"""
import os
import hashlib
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool (lazy init)
# ---------------------------------------------------------------------------
_pool: pool.SimpleConnectionPool | None = None


def _get_dsn() -> str:
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('DB_NAME', 'cal')} "
        f"user={os.getenv('DB_USER', 'cal_user')} "
        f"password={os.getenv('DB_PASSWORD', '')} "
        f"sslmode={os.getenv('DB_SSLMODE', 'prefer')}"
    )


def init_pool(minconn: int = 1, maxconn: int = 5) -> None:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(minconn, maxconn, dsn=_get_dsn())
        logger.info("DB connection pool initialised.")


@contextmanager
def get_conn():
    """Context manager: get a connection from the pool, auto-return on exit."""
    global _pool
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = False):
    """Context manager returning (conn, cur). Commit is manual; rollback on exception."""
    global _pool
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    factory = RealDictCursor if dict_cursor else None
    try:
        with conn.cursor(cursor_factory=factory) as cur:
            yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upsert_season(cur, label: str) -> int:
    """Insert season if not exists; return season_id."""
    parts = label.split("/")
    start_year = int("20" + parts[0]) if len(parts[0]) == 2 else int(parts[0])
    end_year = start_year + 1
    cur.execute(
        """
        INSERT INTO seasons (label, start_year, end_year)
        VALUES (%s, %s, %s)
        ON CONFLICT (label) DO NOTHING
        RETURNING season_id
        """,
        (label, start_year, end_year),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("SELECT season_id FROM seasons WHERE label = %s", (label,))
    return cur.fetchone()[0]


def upsert_team(cur, name: str, source: str) -> int:
    """
    Insert team alias; resolve to canonical team_id.
    First call with a new alias creates both the canonical team and the alias.
    Subsequent aliases for the same canonical name can be mapped via
    the team_aliases table (manual mapping step).
    """
    # Check if alias already mapped
    cur.execute(
        "SELECT team_id FROM team_aliases WHERE source = %s AND alias_name = %s",
        (source, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Create canonical team (name = alias if no canonical exists yet)
    cur.execute(
        "INSERT INTO teams (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING team_id",
        (name,),
    )
    row = cur.fetchone()
    if row:
        team_id = row[0]
    else:
        cur.execute("SELECT team_id FROM teams WHERE name = %s", (name,))
        team_id = cur.fetchone()[0]

    # Register alias
    cur.execute(
        """
        INSERT INTO team_aliases (team_id, source, alias_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (source, alias_name) DO NOTHING
        """,
        (team_id, source, name),
    )
    return team_id


def upsert_referee(cur, name: str, source: str, source_id: str | None = None) -> int:
    """
    Insert referee alias; resolve to canonical referee_id.
    source_id: ID externo da fonte (ex: Sofascore ID) — guardado como alias separado.
    """
    # Verificar alias por nome+fonte
    cur.execute(
        "SELECT referee_id FROM referee_aliases WHERE source = %s AND alias_name = %s",
        (source, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Canonical name: strip extra spaces
    canonical = " ".join(name.strip().split())

    cur.execute(
        "INSERT INTO referees (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING referee_id",
        (canonical,),
    )
    row = cur.fetchone()
    if row:
        referee_id = row[0]
    else:
        cur.execute("SELECT referee_id FROM referees WHERE name = %s", (canonical,))
        referee_id = cur.fetchone()[0]

    # Alias por nome
    cur.execute(
        """
        INSERT INTO referee_aliases (referee_id, source, alias_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (source, alias_name) DO NOTHING
        """,
        (referee_id, source, name),
    )

    # Alias por source_id numérico (facilita lookups futuros)
    if source_id and source_id != name:
        cur.execute(
            """
            INSERT INTO referee_aliases (referee_id, source, alias_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (source, alias_name) DO NOTHING
            """,
            (referee_id, f"{source}_id", source_id),
        )

    return referee_id


def row_hash(row_dict: dict) -> str:
    """SHA-256 hash of a dict for deduplication."""
    payload = "|".join(f"{k}={v}" for k, v in sorted(row_dict.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def log_ingest(
    source: str,
    season_label: str,
    rows_fetched: int,
    rows_inserted: int,
    status: str,
    rows_skipped: int = 0,
    error_msg: str | None = None,
    cur=None,
) -> None:
    """
    Registar resultado de ingestão em ingest_log.
    Se cur=None, abre a própria conexão (útil para chamar fora de get_cursor).
    """
    sql = """
        INSERT INTO ingest_log
            (source, season_label, rows_fetched, rows_inserted, rows_skipped, status, error_msg, finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
    """
    params = (source, season_label, rows_fetched, rows_inserted, rows_skipped, status, error_msg)

    if cur is not None:
        cur.execute(sql, params)
    else:
        with get_cursor() as (conn, c):
            c.execute(sql, params)
