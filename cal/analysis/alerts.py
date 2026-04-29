"""
cal/analysis/alerts.py
Fase 6 — Alertas Automáticos Pós-Jornada

Após cada actualização, detecta padrões anómalos e gera um resumo
com três níveis de threshold:

  Nível 1 — MONITORIZAR  : |Z| > 1.0  ou desvio amarelos > +0.4
  Nível 2 — SUSPEITO     : |Z| > 2.5  ou suspicion_score > 4.0
  Nível 3 — ANOMALIA     : |Z| > 3.0  ou suspicion_score > 6.0

Adicionalmente detecta:
  - Top 3 árbitros com maior SuspicionScore na época actual
  - Árbitros com desvio de faltas > +1.0 numa equipa específica
  - Padrões multi-época (mesmo árbitro, mesmo sentido, 2+ épocas)

Uso:
    python run_alerts.py                    # alertas da época actual
    python run_alerts.py "2023/24"          # época específica
    python run_alerts.py --save             # guarda em referee_alerts
"""

import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from cal.db import get_cursor

log = logging.getLogger(__name__)

CURRENT_SEASON = "2024/25"

THRESHOLDS = {
    "monitorizar": {"z": 1.0,  "suspicion": 2.0, "label": "⚡ MONITORIZAR",  "color": "yellow"},
    "suspeito":    {"z": 2.5,  "suspicion": 4.0, "label": "⚠️  SUSPEITO",    "color": "orange"},
    "anomalia":    {"z": 3.0,  "suspicion": 6.0, "label": "🚨 ANOMALIA",     "color": "red"},
}

# Nota metodológica: os Z-scores usam a fórmula corrigida Σ pᵢ(1-pᵢ)
# Os alertas de nível SUSPEITO e ANOMALIA devem ser lidos em conjunto
# com sig_yellow_fdr / sig_fouls_fdr para confirmar significância
# após correcção de múltiplos testes (Benjamini-Hochberg, FDR 5%).


# ── Queries ────────────────────────────────────────────────────────────────────

def load_current_bias(season: str) -> pd.DataFrame:
    """Bias scores actuais com p-values ajustados FDR."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT s.referee_id, r.name AS referee, s.season,
                   s.n_games, s.yellow_diff_bias_z, s.red_diff_bias_z,
                   s.fouls_diff_bias_z, s.suspicion_score,
                   s.p_adj_yellow, s.p_adj_fouls,
                   s.sig_yellow_fdr, s.sig_fouls_fdr
            FROM referee_bias_scores s
            JOIN referees r USING (referee_id)
            WHERE s.season = %s
            ORDER BY s.suspicion_score DESC
        """, (season,))
        return pd.DataFrame(cur.fetchall())


def load_history_last_n(season: str, last_n: int = 3) -> pd.DataFrame:
    """Últimas N jornadas do histórico — para detectar tendências recentes."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT h.referee_id, r.name AS referee, h.season,
                   h.matchday, h.n_games, h.yellow_diff_bias_z,
                   h.red_diff_bias_z, h.fouls_diff_bias_z, h.suspicion_score
            FROM referee_bias_history h
            JOIN referees r USING (referee_id)
            WHERE h.season = %s
              AND h.matchday >= (
                  SELECT MAX(matchday) - %s
                  FROM referee_bias_history
                  WHERE season = %s
              )
            ORDER BY h.referee_id, h.matchday
        """, (season, last_n, season))
        return pd.DataFrame(cur.fetchall())


def load_multi_season_bias() -> pd.DataFrame:
    """Bias scores das últimas 3 épocas para detectar padrões multi-época."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT s.referee_id, r.name AS referee, s.season,
                   s.yellow_diff_bias_z, s.fouls_diff_bias_z, s.suspicion_score
            FROM referee_bias_scores s
            JOIN referees r USING (referee_id)
            WHERE s.season IN ('2022/23','2023/24','2024/25')
            ORDER BY s.referee_id, s.season
        """)
        return pd.DataFrame(cur.fetchall())


def load_extreme_pairs(season: str) -> pd.DataFrame:
    """Pares árbitro×equipa com desvio extremo na época actual."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT c.*, r.name AS arbitro, t.name AS equipa
            FROM referee_team_cards c
            JOIN referees r USING (referee_id)
            JOIN teams t USING (team_id)
            WHERE c.epoca = %s
              AND (c.desvio_amarelos > 0.8 OR c.desvio_faltas > 2.0)
            ORDER BY c.desvio_amarelos DESC
        """, (season,))
        return pd.DataFrame(cur.fetchall())


# ── Detecção de padrões ────────────────────────────────────────────────────────

def detect_threshold_alerts(bias_df: pd.DataFrame) -> list[dict]:
    """Detecta árbitros que ultrapassaram os limiares de Z-score."""
    alerts = []
    for _, row in bias_df.iterrows():
        max_z = max(abs(float(row["yellow_diff_bias_z"])),
                    abs(float(row["red_diff_bias_z"])),
                    abs(float(row["fouls_diff_bias_z"])))
        suspicion = float(row["suspicion_score"])

        level = None
        if max_z >= 3.0 or suspicion >= 6.0:
            level = "anomalia"
        elif max_z >= 2.5 or suspicion >= 4.0:
            level = "suspeito"
        elif max_z >= 1.0 or suspicion >= 2.0:
            level = "monitorizar"

        if level:
            z_vals = {
                "amarelos": abs(float(row["yellow_diff_bias_z"])),
                "vermelhos": abs(float(row["red_diff_bias_z"])),
                "faltas":   abs(float(row["fouls_diff_bias_z"])),
            }
            col_map = {
                "amarelos":  "yellow_diff_bias_z",
                "vermelhos": "red_diff_bias_z",
                "faltas":    "fouls_diff_bias_z",
            }
            dominant = max(z_vals, key=z_vals.get)
            z_sign   = "+" if float(row[col_map[dominant]]) > 0 else "-"

            alerts.append({
                "level":      level,
                "referee":    row["referee"],
                "referee_id": row["referee_id"],
                "season":     row["season"],
                "n_games":    int(row["n_games"]),
                "suspicion":  suspicion,
                "max_z":      round(max_z, 3),
                "dominant":   dominant,
                "z_sign":     z_sign,
                "z_y":        float(row["yellow_diff_bias_z"]),
                "z_r":        float(row["red_diff_bias_z"]),
                "z_f":        float(row["fouls_diff_bias_z"]),
            })
    return alerts


def detect_top3(bias_df: pd.DataFrame) -> list[dict]:
    """Top 3 árbitros por SuspicionScore."""
    top = bias_df.nlargest(3, "suspicion_score")
    return [
        {
            "rank":      i + 1,
            "referee":   row["referee"],
            "suspicion": float(row["suspicion_score"]),
            "n_games":   int(row["n_games"]),
        }
        for i, (_, row) in enumerate(top.iterrows())
    ]


def detect_trend_alerts(history_df: pd.DataFrame) -> list[dict]:
    """
    Detecta árbitros cujo SuspicionScore tem subido nas últimas 3 jornadas.
    """
    alerts = []
    if history_df.empty:
        return alerts
    for ref_id, grp in history_df.groupby("referee_id"):
        grp = grp.sort_values("matchday")
        if len(grp) < 3:
            continue
        scores = grp["suspicion_score"].astype(float).values
        # Tendência crescente nas últimas 3 jornadas
        if scores[-1] > scores[-2] > scores[-3] and scores[-1] > 3.0:
            alerts.append({
                "referee":    grp["referee"].iloc[0],
                "referee_id": ref_id,
                "season":     grp["season"].iloc[-1],
                "matchday":   int(grp["matchday"].iloc[-1]),
                "suspicion_current": round(float(scores[-1]), 3),
                "suspicion_3j_ago":  round(float(scores[-3]), 3),
                "trend": "crescente",
            })
    return alerts


def detect_multi_season(multi_df: pd.DataFrame) -> list[dict]:
    """
    Detecta árbitros com |Z| > 1.5 em amarelos nas últimas 3 épocas
    sempre no mesmo sentido.
    """
    alerts = []
    if multi_df.empty:
        return alerts
    for ref_id, grp in multi_df.groupby("referee_id"):
        grp = grp.sort_values("season")
        if len(grp) < 2:
            continue
        z_y = grp["yellow_diff_bias_z"].astype(float).values
        biased = np.abs(z_y) > 1.5
        if biased.sum() >= 2:
            signs = np.sign(z_y[biased])
            if len(set(signs)) == 1:  # mesmo sentido
                alerts.append({
                    "referee":    grp["referee"].iloc[0],
                    "referee_id": ref_id,
                    "epocas":     grp["season"].tolist(),
                    "z_values":   [round(z, 3) for z in z_y.tolist()],
                    "sentido":    "mais amarelos" if signs[0] > 0 else "menos amarelos",
                    "n_epocas":   int(biased.sum()),
                })
    return alerts


def detect_fouls_pairs(pairs_df: pd.DataFrame) -> list[dict]:
    """Pares árbitro×equipa com desvio de faltas > +1 na época actual."""
    alerts = []
    if pairs_df.empty:
        return alerts
    for _, row in pairs_df.iterrows():
        df_val = float(row.get("desvio_faltas", 0) or 0)
        dy_val = float(row.get("desvio_amarelos", 0) or 0)
        if df_val > 1.0 or dy_val > 0.5:
            alerts.append({
                "arbitro":  row["arbitro"],
                "equipa":   row["equipa"],
                "jogos":    int(row["jogos"]),
                "desvio_f": round(df_val, 2),
                "desvio_y": round(dy_val, 3),
                "epoca":    row["epoca"],
            })
    return sorted(alerts, key=lambda x: x["desvio_f"], reverse=True)


# ── Formatar relatório de alertas ─────────────────────────────────────────────

def format_report(season: str, threshold_alerts: list, top3: list,
                  trend_alerts: list, multi_alerts: list,
                  fouls_pairs: list) -> str:
    """Gera relatório textual de alertas."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  CAL — RELATÓRIO DE ALERTAS")
    lines.append(f"  Época: {season}  |  Gerado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # Top 3
    lines.append("\n🏆 TOP 3 — MAIOR SUSPICION SCORE NA ÉPOCA")
    lines.append("─" * 50)
    for t in top3:
        lines.append(f"  #{t['rank']}  {t['referee']:<32}  Score={t['suspicion']:.3f}  ({t['n_games']} jogos)")

    # Threshold alerts por nível
    for level_key in ["anomalia", "suspeito", "monitorizar"]:
        level_alerts = [a for a in threshold_alerts if a["level"] == level_key]
        if not level_alerts:
            continue
        cfg = THRESHOLDS[level_key]
        lines.append(f"\n{cfg['label']}  (|Z| > {cfg['z']} ou SuspicionScore > {cfg['suspicion']})")
        lines.append("─" * 50)
        for a in level_alerts:
            sign_y = f"{a['z_y']:+.2f}"
            sign_r = f"{a['z_r']:+.2f}"
            sign_f = f"{a['z_f']:+.2f}"
            lines.append(
                f"  {a['referee']:<32}  "
                f"Score={a['suspicion']:.3f}  "
                f"Z: amarelos={sign_y} vermelhos={sign_r} faltas={sign_f}  "
                f"({a['n_games']}j)"
            )

    # Tendência crescente
    if trend_alerts:
        lines.append("\n📈 TENDÊNCIA CRESCENTE (SuspicionScore a subir 3 jornadas)")
        lines.append("─" * 50)
        for t in trend_alerts:
            lines.append(
                f"  {t['referee']:<32}  "
                f"Jornada {t['matchday']}  "
                f"Score: {t['suspicion_3j_ago']:.2f} → {t['suspicion_current']:.2f}"
            )

    # Multi-época
    if multi_alerts:
        lines.append("\n🔄 PADRÃO MULTI-ÉPOCA (mesmo sentido em 2+ épocas)")
        lines.append("─" * 50)
        for m in multi_alerts:
            epocas_str = " | ".join(
                f"{e}: Z={z:+.2f}"
                for e, z in zip(m["epocas"], m["z_values"])
            )
            lines.append(
                f"  {m['referee']:<32}  "
                f"{m['sentido']}  —  {epocas_str}"
            )

    # Pares árbitro×equipa com desvio extremo de faltas
    if fouls_pairs:
        lines.append(f"\n🦵 PARES ÁRBITRO×EQUIPA COM DESVIO EXTREMO ({season})")
        lines.append("─" * 50)
        for p in fouls_pairs[:10]:
            lines.append(
                f"  {p['arbitro']:<30}  {p['equipa']:<18}  "
                f"Δfaltas={p['desvio_f']:+.1f}  Δamarelhos={p['desvio_y']:+.2f}  "
                f"({p['jogos']}j)"
            )

    lines.append("\n" + "=" * 70)
    lines.append("  Nota: alertas indicativos — não constituem prova de parcialidade.")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Guardar alertas na DB ─────────────────────────────────────────────────────

def save_alerts_db(threshold_alerts: list, season: str) -> None:
    with get_cursor() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS referee_alerts (
                id          SERIAL PRIMARY KEY,
                referee_id  INTEGER,
                season      VARCHAR(10),
                alert_level VARCHAR(20),
                suspicion_score NUMERIC(8,4),
                max_z       NUMERIC(8,4),
                dominant_metric VARCHAR(20),
                z_sign      VARCHAR(2),
                n_games     INTEGER,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        for a in threshold_alerts:
            cur.execute("""
                INSERT INTO referee_alerts
                    (referee_id, season, alert_level, suspicion_score,
                     max_z, dominant_metric, z_sign, n_games)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (a["referee_id"], season, a["level"], a["suspicion"],
                  a["max_z"], a["dominant"], a["z_sign"], a["n_games"]))
        conn.commit()
    log.info("Guardados %d alertas em referee_alerts", len(threshold_alerts))


# ── Entry point ───────────────────────────────────────────────────────────────

def run(season: Optional[str] = None, save: bool = False) -> str:
    target = season or CURRENT_SEASON
    log.info("A gerar alertas para %s...", target)

    bias_df   = load_current_bias(target)
    hist_df   = load_history_last_n(target, last_n=3)
    multi_df  = load_multi_season_bias()
    pairs_df  = load_extreme_pairs(target)

    for df in [bias_df, hist_df, multi_df]:
        for col in ["yellow_diff_bias_z","red_diff_bias_z",
                    "fouls_diff_bias_z","suspicion_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    threshold_alerts = detect_threshold_alerts(bias_df)
    top3             = detect_top3(bias_df)
    trend_alerts     = detect_trend_alerts(hist_df)
    multi_alerts     = detect_multi_season(multi_df)
    fouls_pairs      = detect_fouls_pairs(pairs_df)

    report = format_report(
        target, threshold_alerts, top3,
        trend_alerts, multi_alerts, fouls_pairs
    )

    if save:
        save_alerts_db(threshold_alerts, target)

    return report


if __name__ == "__main__":
    import sys, dotenv, logging
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])
    season_arg = next((a for a in sys.argv[1:] if "/" in a), None)
    save_arg   = "--save" in sys.argv
    report = run(season=season_arg, save=save_arg)
    print(report)
