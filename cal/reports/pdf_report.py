"""
cal/reports/pdf_report.py
Fase 6 — Relatório PDF por Árbitro / Época

Gera um relatório PDF completo para um árbitro e época específicos com:
  - Capa com nome, época, métricas principais e veredito
  - Z-scores de viés por época (tabela + gráfico de barras)
  - Top equipas com mais/menos faltas e amarelos
  - Desvios extremos por equipa
  - Nota metodológica

Uso:
    python run_pdf_report.py "Artur Soares Dias" "2023/24"
    python run_pdf_report.py "Luís Godinho"           # todas as épocas
"""

import io
import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns
from cal.db import get_cursor

log = logging.getLogger(__name__)

# ── Paleta de cores ────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1d23")
C_MID     = colors.HexColor("#4e5663")
C_LIGHT   = colors.HexColor("#d4d8de")
C_RED     = colors.HexColor("#d6604d")
C_ORANGE  = colors.HexColor("#f5a623")
C_BLUE    = colors.HexColor("#4393c3")
C_GREEN   = colors.HexColor("#4caf50")
C_BG      = colors.HexColor("#f5f7fa")
C_AMBER   = colors.HexColor("#f5a623")
C_WHITE   = colors.white


# ── Queries ────────────────────────────────────────────────────────────────────

def load_referee_id(name: str) -> Optional[int]:
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT referee_id FROM referees
            WHERE name ILIKE %s LIMIT 1
        """, (f"%{name}%",))
        row = cur.fetchone()
        return row["referee_id"] if row else None


def load_bias_scores(referee_id: int) -> pd.DataFrame:
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("""
            SELECT season, n_games, yellow_diff_bias_z,
                   red_diff_bias_z, fouls_diff_bias_z, suspicion_score,
                   p_adj_yellow, p_adj_fouls,
                   sig_yellow_fdr, sig_fouls_fdr
            FROM referee_bias_scores
            WHERE referee_id = %s
            ORDER BY season
        """, (referee_id,))
        return pd.DataFrame(cur.fetchall())


def load_team_cards(referee_id: int, season: Optional[str] = None) -> pd.DataFrame:
    if season:
        with get_cursor(dict_cursor=True) as (conn, cur):
            cur.execute("""
                SELECT c.*, t.name AS equipa
                FROM referee_team_cards c
                JOIN teams t USING (team_id)
                WHERE c.referee_id = %s AND c.epoca = %s
                ORDER BY c.media_amarelos DESC
            """, (referee_id, season))
            return pd.DataFrame(cur.fetchall())
    else:
        with get_cursor(dict_cursor=True) as (conn, cur):
            cur.execute("""
                SELECT c.*, t.name AS equipa
                FROM referee_team_cards_total c
                JOIN teams t USING (team_id)
                WHERE c.referee_id = %s
                ORDER BY c.media_amarelos DESC
            """, (referee_id,))
            return pd.DataFrame(cur.fetchall())


def load_referee_name(referee_id: int) -> str:
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute("SELECT name FROM referees WHERE referee_id = %s", (referee_id,))
        row = cur.fetchone()
        return row["name"] if row else "Desconhecido"


def load_severity(referee_id: int) -> pd.DataFrame:
    """Faltas por amarelo para todas as equipas (sem threshold mínimo)."""
    with get_cursor(dict_cursor=True) as (conn, cur):
        try:
            cur.execute("""
                SELECT s.*, t.name AS equipa
                FROM referee_team_severity s
                JOIN teams t USING (team_id)
                WHERE s.referee_id = %s
                  AND s.amarelos_total > 0
                ORDER BY s.faltas_por_amarelo DESC
            """, (referee_id,))
            return pd.DataFrame(cur.fetchall())
        except Exception:
            conn.rollback()
            return pd.DataFrame()


# ── Gráficos matplotlib → imagem em memória ───────────────────────────────────

def make_zscore_chart(bias_df: pd.DataFrame, width_cm=16, height_cm=7) -> io.BytesIO:
    """Gráfico de barras dos Z-scores por época."""
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#f5f7fa")

    seasons  = bias_df["season"].tolist()
    z_y      = bias_df["yellow_diff_bias_z"].astype(float).tolist()
    z_f      = bias_df["fouls_diff_bias_z"].astype(float).tolist()

    x = np.arange(len(seasons))
    w = 0.35

    bars_y = ax.bar(x - w/2, z_y, w, label="Z Amarelos",
                    color=["#d6604d" if v > 0 else "#4393c3" for v in z_y], alpha=0.85)
    bars_f = ax.bar(x + w/2, z_f, w, label="Z Faltas",
                    color=["#f5a623" if v > 0 else "#92c5de" for v in z_f], alpha=0.85)

    ax.axhline(0,   color="#888", linewidth=0.8)
    ax.axhline(2.5, color="#f5a623", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-2.5,color="#f5a623", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(3.0, color="#d6604d", linewidth=0.8, linestyle=":",  alpha=0.7)
    ax.axhline(-3.0,color="#d6604d", linewidth=0.8, linestyle=":",  alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=30, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Z-score", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_teams_chart(cards_df: pd.DataFrame, metric: str = "desvio_vs_media_arbitro",
                     label: str = "Desvio amarelos/jogo",
                     width_cm=16, height_cm=9) -> io.BytesIO:
    """Gráfico de barras horizontais por equipa."""
    df = cards_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    df = df.sort_values(metric).tail(20)  # top 20

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#f5f7fa")

    vals   = df[metric].tolist()
    labels = df["equipa"].tolist()
    colors_bar = ["#d6604d" if v > 0.8 else "#f5a623" if v > 0.4
                  else "#4393c3" if v < -0.4 else "#aab4c0"
                  for v in vals]

    ax.barh(labels, vals, color=colors_bar, alpha=0.85)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(label, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Estilos ReportLab ──────────────────────────────────────────────────────────

def get_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Normal"],
            fontSize=16, fontName="Helvetica-Bold",
            textColor=C_DARK, spaceAfter=6, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, fontName="Helvetica",
            textColor=C_MID, spaceAfter=12, alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Normal"],
            fontSize=11, fontName="Helvetica-Bold",
            textColor=C_DARK, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=8.5, fontName="Helvetica",
            textColor=C_DARK, spaceAfter=4, leading=13,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"],
            fontSize=7, fontName="Helvetica",
            textColor=C_MID, spaceAfter=3, leading=10,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=C_MID, spaceAfter=6, alignment=TA_CENTER,
        ),
        "metric_val": ParagraphStyle(
            "metric_val", parent=base["Normal"],
            fontSize=18, fontName="Helvetica-Bold",
            textColor=C_AMBER, alignment=TA_CENTER,
        ),
        "metric_lbl": ParagraphStyle(
            "metric_lbl", parent=base["Normal"],
            fontSize=7, fontName="Helvetica",
            textColor=C_MID, alignment=TA_CENTER,
        ),
        "verdict_green": ParagraphStyle(
            "verdict_green", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#2e7d32"),
        ),
        "verdict_red": ParagraphStyle(
            "verdict_red", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_RED,
        ),
        "verdict_neutral": ParagraphStyle(
            "verdict_neutral", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_MID,
        ),
    }
    return styles


# ── Componentes do relatório ───────────────────────────────────────────────────

def build_cover(styles, referee_name: str, season: Optional[str],
                bias_df: pd.DataFrame, cards_df: pd.DataFrame) -> list:
    """Capa do relatório."""
    story = []

    # Cabeçalho CAL
    story.append(Paragraph("CAL — Criticar a Arbitragem Legalmente", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_AMBER, spaceAfter=16))

    # Título
    story.append(Paragraph(referee_name, styles["title"]))
    epoch_label = season if season else "Todas as épocas (acumulado)"
    story.append(Paragraph(f"Relatório de arbitragem — {epoch_label}", styles["subtitle"]))
    story.append(Spacer(1, 0.4*cm))

    # Métricas de cabeçalho em tabela
    if not bias_df.empty:
        # Filtrar para a época se especificado
        if season:
            b = bias_df[bias_df["season"] == season]
        else:
            b = bias_df

        if not b.empty:
            n_jogos  = int(b["n_games"].sum())
            max_z    = float(b[["yellow_diff_bias_z","red_diff_bias_z",
                                  "fouls_diff_bias_z"]].astype(float).abs().max().max())
            sus_max  = float(b["suspicion_score"].astype(float).max())

            # Determinar nível de alerta
            if max_z >= 3.0 or sus_max >= 6.0:
                nivel = "ANOMALIA"
                nivel_color = C_RED
                nivel_bg    = colors.HexColor("#fdecea")
            elif max_z >= 2.5 or sus_max >= 4.0:
                nivel = "SUSPEITO"
                nivel_color = C_ORANGE
                nivel_bg    = colors.HexColor("#fff3e0")
            elif max_z >= 1.0 or sus_max >= 2.0:
                nivel = "MONITORIZAR"
                nivel_color = C_BLUE
                nivel_bg    = colors.HexColor("#e3f2fd")
            else:
                nivel = "NORMAL"
                nivel_color = C_GREEN
                nivel_bg    = colors.HexColor("#e8f5e9")

            metrics_data = [
                [
                    Paragraph(str(n_jogos),        styles["metric_val"]),
                    Paragraph(f"{max_z:.3f}",       styles["metric_val"]),
                    Paragraph(f"{sus_max:.3f}",     styles["metric_val"]),
                    Paragraph(nivel,
                              ParagraphStyle("nv", parent=getSampleStyleSheet()["Normal"],
                                             fontSize=13, fontName="Helvetica-Bold",
                                             textColor=nivel_color, alignment=TA_CENTER)),
                ],
                [
                    Paragraph("Jogos",              styles["metric_lbl"]),
                    Paragraph("Z-score máximo",     styles["metric_lbl"]),
                    Paragraph("SuspicionScore",     styles["metric_lbl"]),
                    Paragraph("Nivel de alerta",    styles["metric_lbl"]),
                ],
            ]
            t = Table(metrics_data, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), C_BG),
                ("BACKGROUND", (3,0), (3,0),   nivel_bg),
                ("BOX",        (0,0), (-1,-1), 0.5, C_LIGHT),
                ("INNERGRID",  (0,0), (-1,-1), 0.3, C_LIGHT),
                ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*cm))

    # Data de geração
    story.append(Paragraph(
        f"Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} · "
        f"Dados: football-data.co.uk + Sofascore · Liga Portugal 2017/18–2023/24",
        styles["small"]
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LIGHT, spaceAfter=8))

    return story


def build_bias_section(styles, bias_df: pd.DataFrame,
                        season: Optional[str]) -> list:
    """Secção de Z-scores de viés."""
    story = []
    story.append(Paragraph("Z-scores de Viés por Época", styles["section"]))
    story.append(Paragraph(
        "Os Z-scores medem o desvio entre as decisões observadas e as esperadas pelos modelos. "
        "Valores absolutos acima de 2.5 são considerados suspeitos; acima de 3.0, anómalos.",
        styles["body"]
    ))

    if bias_df.empty:
        story.append(Paragraph("Sem dados disponíveis.", styles["small"]))
        return story

    # Filtrar época se necessário
    df = bias_df.copy()
    if season:
        df = df[df["season"] == season]

    for col in ["yellow_diff_bias_z","red_diff_bias_z","fouls_diff_bias_z","suspicion_score","n_games"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["sig_yellow_fdr","sig_fouls_fdr"]:
        if col not in df.columns:
            df[col] = None

    # Gráfico Z-scores
    if len(bias_df) > 1 and not season:
        chart_buf = make_zscore_chart(bias_df)
        img = Image(chart_buf, width=15*cm, height=6*cm)
        story.append(img)
        story.append(Paragraph("Z-scores acumulados por época (barras vermelhas/laranja = positivo, azul = negativo).",
                                styles["caption"]))

    # Tabela Z-scores com coluna FDR
    header = ["Época", "Jogos", "Z Amarelos", "Z Vermelhos", "Z Faltas", "SuspicionScore", "FDR sig."]
    rows   = [header]

    def z_color(val):
        v = abs(float(val))
        if v >= 3.0: return C_RED
        if v >= 2.5: return C_ORANGE
        if v >= 1.5: return C_DARK
        return C_MID

    for _, row in df.iterrows():
        sig_y = row.get("sig_yellow_fdr")
        sig_f = row.get("sig_fouls_fdr")
        fdr_parts = []
        if sig_f is True: fdr_parts.append("faltas")
        if sig_y is True: fdr_parts.append("amarelos")
        fdr_cell = ", ".join(fdr_parts) if fdr_parts else "—"
        rows.append([
            row["season"],
            str(int(row["n_games"])),
            f"{float(row['yellow_diff_bias_z']):+.3f}",
            f"{float(row['red_diff_bias_z']):+.3f}",
            f"{float(row['fouls_diff_bias_z']):+.3f}",
            f"{float(row['suspicion_score']):.3f}",
            fdr_cell,
        ])

    t = Table(rows, colWidths=[2.2*cm, 1.3*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.4*cm])
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]

    # Colorir células Z extremas e FDR
    for i, row in enumerate(rows[1:], start=1):
        for j, val in enumerate(row[2:5], start=2):
            try:
                v = abs(float(val))
                if v >= 3.0:
                    style_cmds.append(("TEXTCOLOR", (j,i), (j,i), C_RED))
                    style_cmds.append(("FONTNAME",  (j,i), (j,i), "Helvetica-Bold"))
                elif v >= 2.5:
                    style_cmds.append(("TEXTCOLOR", (j,i), (j,i), C_ORANGE))
                    style_cmds.append(("FONTNAME",  (j,i), (j,i), "Helvetica-Bold"))
            except: pass
        # FDR column (col 6)
        fdr_val = row[6]
        if fdr_val and fdr_val != "—":
            style_cmds.append(("TEXTCOLOR", (6,i), (6,i), colors.HexColor("#2e7d32")))
            style_cmds.append(("FONTNAME",  (6,i), (6,i), "Helvetica-Bold"))
        else:
            style_cmds.append(("TEXTCOLOR", (6,i), (6,i), colors.HexColor("#999999")))

    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Limiares: |Z| >= 1.0 monitorizar  |  |Z| >= 2.5 suspeito  |  |Z| >= 3.0 anomalia  |  "
        "FDR sig.: significativo apos correccao Benjamini-Hochberg (FDR 5%).",
        styles["small"]
    ))
    return story


def build_teams_section(styles, cards_df: pd.DataFrame,
                         referee_name: str, season_label: str) -> list:
    """Secção de cartões e faltas por equipa."""
    story = []
    story.append(Paragraph(f"Cartões e Faltas por Equipa — {season_label}", styles["section"]))

    if cards_df.empty:
        story.append(Paragraph("Sem dados disponíveis.", styles["small"]))
        return story

    # Converter colunas
    for col in ["media_amarelos","media_faltas","desvio_vs_media_arbitro",
                "desvio_faltas_vs_media","jogos_total","jogos",
                "percentil_amarelos","percentil_faltas"]:
        if col in cards_df.columns:
            cards_df[col] = pd.to_numeric(cards_df[col], errors="coerce").fillna(0)

    jogos_col  = "jogos_total" if "jogos_total" in cards_df.columns else "jogos"
    desvio_col = "desvio_vs_media_arbitro" if "desvio_vs_media_arbitro" in cards_df.columns else "desvio_amarelos"
    faltas_col = "desvio_faltas_vs_media" if "desvio_faltas_vs_media" in cards_df.columns else "desvio_faltas"

    # Gráfico desvio amarelos
    if desvio_col in cards_df.columns:
        chart_y = make_teams_chart(
            cards_df.rename(columns={desvio_col: "desvio_vs_media_arbitro"}),
            metric="desvio_vs_media_arbitro",
            label="Desvio amarelos/jogo vs média árbitro"
        )
        story.append(Image(chart_y, width=15*cm, height=8*cm))
        story.append(Paragraph(
            "Desvio de amarelos por equipa face à média geral do árbitro. "
            "Vermelho = mais cartões; azul = menos cartões.",
            styles["caption"]
        ))

    # Tabela top 10 desvios extremos
    story.append(Paragraph("Top 10 desvios extremos (acumulado)", styles["section"]))

    top = cards_df.sort_values(desvio_col, ascending=False).head(10)

    # Calcular amarelos_por_falta se não existir
    if "amarelos_por_falta" not in cards_df.columns:
        cards_df["amarelos_por_falta"] = (
            cards_df["amarelos_total"].astype(float) /
            cards_df["faltas_total"].replace(0, np.nan).astype(float)
        ).round(4).fillna(0)

    cols_show  = ["equipa", jogos_col, "media_amarelos", desvio_col,
                  "amarelos_por_falta", "media_faltas", faltas_col]
    cols_show  = [c for c in cols_show if c in cards_df.columns]
    col_labels = {
        "equipa": "Equipa", jogos_col: "Jogos",
        "media_amarelos": "Amar./jogo",
        desvio_col: "Desv.Amar.",
        "amarelos_por_falta": "Amar/Falta",
        "media_faltas": "Faltas/jogo",
        faltas_col: "Desv.Faltas",
    }

    header = [col_labels.get(c, c) for c in cols_show]
    rows   = [header]
    for _, row in top.iterrows():
        r = []
        for c in cols_show:
            val = row.get(c, "")
            if c == "equipa":
                r.append(str(val))
            elif c == "percentil_amarelos":
                r.append(f"{float(val):.0f}%")
            elif c == "amarelos_por_falta":
                r.append(f"{float(val):.4f}")
            elif c in [desvio_col, faltas_col]:
                r.append(f"{float(val):+.2f}")
            else:
                r.append(f"{float(val):.2f}" if isinstance(val, float) else str(val))
        rows.append(r)

    col_w = [3.5*cm] + [1.8*cm] * (len(cols_show) - 1)
    t = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,0), (0,-1),  "LEFT"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]

    # Cor nas linhas com desvio extremo
    if desvio_col in cols_show:
        d_idx = cols_show.index(desvio_col)
        for i, row in enumerate(rows[1:], start=1):
            try:
                d = float(row[d_idx].replace("+",""))
                if d > 0.8:
                    style_cmds.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fff0ee")))
                elif d > 0.4:
                    style_cmds.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fff8ec")))
            except: pass

    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    return story


def build_severity_section(styles, severity_df: pd.DataFrame,
                            referee_name: str) -> list:
    """Secção: 1 amarelo a cada X faltas — todas as equipas, sem threshold."""
    story = []
    story.append(Paragraph("Severidade — 1 Amarelo a Cada X Faltas (todas as equipas)", styles["section"]))
    story.append(Paragraph(
        "Esta tabela mostra quantas faltas o árbitro assinala, em média, antes de mostrar "
        "um cartão amarelo a cada equipa. Valores mais baixos indicam maior severidade com essa equipa. "
        "Inclui todas as equipas com pelo menos 1 jogo — sem threshold mínimo.",
        styles["body"]
    ))

    if severity_df.empty:
        story.append(Paragraph(
            "Sem dados. Correr run_cards.py para calcular a tabela referee_team_severity.",
            styles["small"]
        ))
        return story

    for col in ["jogos","amarelos_total","faltas_total","media_amarelos",
                "media_faltas","faltas_por_amarelo"]:
        if col in severity_df.columns:
            severity_df[col] = pd.to_numeric(severity_df[col], errors="coerce").fillna(0)

    # Ordenar por faltas_por_amarelo ascendente = mais severo primeiro
    sev = severity_df.sort_values("faltas_por_amarelo").copy()
    media_global = float(sev["faltas_por_amarelo"].mean()) if not sev.empty else 0

    header = ["Equipa", "Jogos", "Amarelos", "Faltas", "Faltas/jogo", "1 amarelo a cada X faltas"]
    rows   = [header]
    for _, row in sev.iterrows():
        fpa = float(row["faltas_por_amarelo"])
        rows.append([
            str(row["equipa"]),
            str(int(row["jogos"])),
            str(int(row["amarelos_total"])),
            str(int(row["faltas_total"])),
            f"{float(row['media_faltas']):.1f}",
            f"{fpa:.1f}",
        ])

    col_w = [3.5*cm, 1.5*cm, 2*cm, 1.8*cm, 2.5*cm, 4*cm]
    t = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,0), (0,-1),  "LEFT"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    # Destacar equipas muito abaixo da média (mais severo)
    fpa_col = 5
    for i, row in enumerate(rows[1:], start=1):
        try:
            fpa = float(row[fpa_col])
            if fpa < media_global * 0.5:
                style_cmds.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fdecea")))
                style_cmds.append(("TEXTCOLOR",  (fpa_col,i), (fpa_col,i), C_RED))
                style_cmds.append(("FONTNAME",   (fpa_col,i), (fpa_col,i), "Helvetica-Bold"))
            elif fpa > media_global * 1.5:
                style_cmds.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#e3f2fd")))
                style_cmds.append(("TEXTCOLOR",  (fpa_col,i), (fpa_col,i), C_BLUE))
        except: pass

    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Média geral do árbitro: {media_global:.1f} faltas por amarelo. "
        f"Vermelho = muito mais severo que a média; azul = muito menos severo.",
        styles["small"]
    ))
    return story


def build_methodology(styles) -> list:
    """Nota metodológica final."""
    story = []
    story.append(PageBreak())
    story.append(Paragraph("Nota Metodológica", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LIGHT, spaceAfter=8))

    paras = [
        ("Modelos estatísticos",
         "xYC (expected yellow card) e xRC (expected red card): regressão logística que estima "
         "a probabilidade de cartão dado o contexto do jogo — is_home, matchday, força relativa "
         "das equipas (rolling 10 jogos), tendência do árbitro (rolling 20 jogos). "
         "xF (expected fouls): modelo Poisson que estima o número esperado de faltas. "
         "Validação: leave-one-season-out (LOSO) — os modelos são treinados em N-1 épocas "
         "e validados na época excluída."),
        ("Z-score de viés — cartões (fórmula corrigida)",
         "Z = (jogos_com_cartao_obs - sum(P_i)) / sqrt(sum(P_i * (1 - P_i))). "
         "A variância é calculada como soma das variâncias individuais de Bernoulli — "
         "sum(P_i * (1 - P_i)) — e não como exp*(1-exp/n), que seria incorrecta para "
         "probabilidades não identicamente distribuídas. "
         "Ref: Agresti (2002), Categorical Data Analysis. "
         "Distribuição aproximadamente Normal para n >= 5 jogos."),
        ("Z-score de viés — faltas",
         "Z = (media_faltas_obs - media_faltas_esperada) / (desvio_padrao / sqrt(n)). "
         "Teste t-student; para n < 30 interpretar com precaução."),
        ("Correcção de múltiplos testes — FDR",
         "Aplicada correcção Benjamini-Hochberg (FDR 5%) sobre todos os p-values "
         "em simultâneo (todos os árbitros × épocas). A coluna 'FDR sig.' indica "
         "apenas as métricas com evidência estatística robusta após esta correcção. "
         "Z-scores sem confirmação FDR devem ser interpretados como potencial ruído "
         "estatístico. Ref: Benjamini & Hochberg (1995), JRSS-B."),
        ("SuspicionScore",
         "|Z_amarelos| + |Z_vermelhos| + |Z_faltas|. Medida ordinal de intensidade "
         "agregada — não tem distribuição estatística conhecida. "
         "Valores: >= 2.0 monitorizar, >= 4.0 suspeito, >= 6.0 anomalia."),
        ("Desvio por equipa",
         "Diferença entre a média de amarelos/faltas deste árbitro com esta equipa específica "
         "e a média global do árbitro com todas as equipas. "
         "Permite identificar pares árbitro×equipa anómalos independentemente do estilo "
         "geral do árbitro."),
        ("Limitações",
         "O sistema não prova intenção — identifica padrões estatisticamente anómalos. "
         "Equipas com estilos de jogo mais físicos acumulam naturalmente mais faltas e "
         "cartões, o que pode inflar os desvios positivos. "
         "Amostras inferiores a 5 jogos são excluídas dos Z-scores. "
         "A ausência de sinalização não pode ser interpretada como ausência de viés."),
    ]
    for title, text in paras:
        story.append(Paragraph(f"<b>{title}:</b> {text}", styles["body"]))
        story.append(Spacer(1, 0.2*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LIGHT, spaceAfter=6))
    story.append(Paragraph(
        f"CAL v0.8.0 · Dados: football-data.co.uk + Sofascore API · "
        f"Liga Portugal 2017/18–2025/26 · Gerado em {datetime.now().strftime('%d/%m/%Y')}",
        styles["small"]
    ))
    return story


# ── Gerador principal ──────────────────────────────────────────────────────────

def generate_report(referee_name: str, season: Optional[str] = None,
                    output_dir: str = None) -> str:
    """
    Gera o relatório PDF e devolve o caminho do ficheiro.
    Output: /downloads se montado (Downloads do Mac), senão /app/reports.
    """
    if output_dir is None:
        output_dir = "/downloads" if os.path.isdir("/downloads") else "/app/reports"
    os.makedirs(output_dir, exist_ok=True)

    # Carregar dados
    ref_id = load_referee_id(referee_name)
    if ref_id is None:
        raise ValueError(f"Árbitro '{referee_name}' não encontrado na DB.")

    full_name  = load_referee_name(ref_id)
    bias_df    = load_bias_scores(ref_id)
    severity_df = load_severity(ref_id)

    # Para a tabela de equipas: usar dados da época se tiver >= 5 equipas,
    # caso contrário usar o acumulado total (mais informativo)
    cards_season = load_team_cards(ref_id, season)
    cards_total  = load_team_cards(ref_id, None)

    if season and len(cards_season) >= 5:
        cards_df       = cards_season
        cards_label    = season
    else:
        cards_df       = cards_total
        cards_label    = f"acumulado (época {season} tem poucos dados)" if season else "acumulado"

    for col in ["yellow_diff_bias_z","red_diff_bias_z","fouls_diff_bias_z","suspicion_score","n_games"]:
        if col in bias_df.columns:
            bias_df[col] = pd.to_numeric(bias_df[col], errors="coerce").fillna(0)

    # Nome do ficheiro
    safe_name   = full_name.replace(" ", "_").replace("/", "-")
    season_slug = season.replace("/", "-") if season else "todas-epocas"
    filename    = f"CAL_{safe_name}_{season_slug}.pdf"
    filepath    = os.path.join(output_dir, filename)

    # Construir PDF
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=f"CAL — {full_name} — {season or 'Todas as épocas'}",
        author="CAL — Criticar a Arbitragem Legalmente",
    )

    styles = get_styles()
    story  = []

    # Capa
    story += build_cover(styles, full_name, season, bias_df, cards_df)

    # Z-scores
    story.append(Spacer(1, 0.5*cm))
    story += build_bias_section(styles, bias_df, season)

    # Equipas
    story.append(Spacer(1, 0.5*cm))
    story += build_teams_section(styles, cards_df, full_name, cards_label)

    # Severidade — faltas por amarelo (todas as equipas)
    story.append(Spacer(1, 0.5*cm))
    story += build_severity_section(styles, severity_df, full_name)

    # Metodologia
    story += build_methodology(styles)

    doc.build(story)
    log.info("Relatório gerado: %s", filepath)
    return filepath


if __name__ == "__main__":
    import sys, dotenv, logging
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Uso: python -m cal.reports.pdf_report 'Nome Árbitro' ['2023/24']")
        sys.exit(1)

    referee_arg = args[0]
    season_arg  = args[1] if len(args) > 1 else None
    path = generate_report(referee_arg, season_arg)
    print(f"✓ Relatório gerado: {path}")
