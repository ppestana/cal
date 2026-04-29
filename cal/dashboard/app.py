"""
cal/dashboard/app.py
CAL — Dashboard Streamlit
Fase 5 — Módulo 1: Cartões por Árbitro × Equipa

Correr:
    streamlit run cal/dashboard/app.py --server.port 8501
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="CAL — Arbitragem",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
/* Métricas — bordas suaves */
div[data-testid="metric-container"] {
    border: 1px solid #d0d7e0;
    border-radius: 6px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ── Ligação à DB ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "cal"),
        user=os.getenv("DB_USER", "cal_user"),
        password=os.getenv("DB_PASSWORD", ""),
    )


@st.cache_data(ttl=300)
def query(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall())


# ── Carregar dados ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_referees():
    return query("""
        SELECT DISTINCT r.referee_id, r.name AS arbitro
        FROM referees r
        JOIN referee_team_cards_total t USING (referee_id)
        ORDER BY r.name
    """)


@st.cache_data(ttl=300)
def load_teams():
    return query("""
        SELECT DISTINCT t.team_id, t.name AS equipa
        FROM teams t
        JOIN referee_team_cards_total c USING (team_id)
        ORDER BY t.name
    """)


@st.cache_data(ttl=300)
def load_total(referee_id: int = None):
    if referee_id:
        return query("""
            SELECT c.*, r.name AS arbitro, t.name AS equipa
            FROM referee_team_cards_total c
            JOIN referees r USING (referee_id)
            JOIN teams t USING (team_id)
            WHERE c.referee_id = %s
            ORDER BY c.media_amarelos DESC
        """, (referee_id,))
    return query("""
        SELECT c.*, r.name AS arbitro, t.name AS equipa
        FROM referee_team_cards_total c
        JOIN referees r USING (referee_id)
        JOIN teams t USING (team_id)
        ORDER BY c.desvio_vs_media_arbitro DESC
    """)


@st.cache_data(ttl=300)
def load_by_season(referee_id: int):
    return query("""
        SELECT c.*, r.name AS arbitro, t.name AS equipa
        FROM referee_team_cards c
        JOIN referees r USING (referee_id)
        JOIN teams t USING (team_id)
        WHERE c.referee_id = %s
        ORDER BY c.epoca, c.media_amarelos DESC
    """, (referee_id,))


@st.cache_data(ttl=300)
def load_bias_scores(referee_id: int):
    return query("""
        SELECT season, n_games, yellow_diff_bias_z,
               red_diff_bias_z, fouls_diff_bias_z, suspicion_score,
               p_adj_yellow, p_adj_fouls,
               sig_yellow_fdr, sig_fouls_fdr
        FROM referee_bias_scores
        WHERE referee_id = %s
        ORDER BY season
    """, (referee_id,))


@st.cache_data(ttl=300)
def load_extreme_desvios():
    return query("""
        SELECT c.desvio_vs_media_arbitro, c.media_amarelos,
               c.media_vermelhos, c.jogos_total, c.percentil_amarelos,
               c.ranking_amarelos, c.epocas,
               c.media_faltas, c.desvio_faltas_vs_media,
               c.percentil_faltas, c.ranking_faltas,
               r.name AS arbitro, t.name AS equipa
        FROM referee_team_cards_total c
        JOIN referees r USING (referee_id)
        JOIN teams t USING (team_id)
        WHERE c.desvio_vs_media_arbitro > 0.7
           OR c.desvio_vs_media_arbitro < -0.7
           OR c.desvio_faltas_vs_media > 2.0
           OR c.desvio_faltas_vs_media < -2.0
        ORDER BY c.desvio_vs_media_arbitro DESC
    """)

@st.cache_data(ttl=300)
def load_extreme_faltas():
    return query("""
        SELECT c.desvio_faltas_vs_media, c.media_faltas,
               c.jogos_total, c.percentil_faltas,
               c.ranking_faltas, c.epocas,
               r.name AS arbitro, t.name AS equipa
        FROM referee_team_cards_total c
        JOIN referees r USING (referee_id)
        JOIN teams t USING (team_id)
        WHERE c.desvio_faltas_vs_media > 2.0
           OR c.desvio_faltas_vs_media < -2.0
        ORDER BY c.desvio_faltas_vs_media DESC
    """)



@st.cache_data(ttl=300)
def load_bias_history(referee_id: int, season: str = None):
    if season:
        return query("""
            SELECT h.matchday, h.n_games, h.yellow_diff_bias_z,
                   h.red_diff_bias_z, h.fouls_diff_bias_z, h.suspicion_score
            FROM referee_bias_history h
            WHERE h.referee_id = %s AND h.season = %s
            ORDER BY h.matchday
        """, (referee_id, season))
    return query("""
        SELECT h.season, h.matchday, h.n_games, h.yellow_diff_bias_z,
               h.red_diff_bias_z, h.fouls_diff_bias_z, h.suspicion_score
        FROM referee_bias_history h
        WHERE h.referee_id = %s
        ORDER BY h.season, h.matchday
    """, (referee_id,))


@st.cache_data(ttl=300)
def load_bias_scores_all():
    return query("""
        SELECT s.referee_id, r.name AS referee, s.season,
               s.n_games, s.yellow_diff_bias_z, s.red_diff_bias_z,
               s.fouls_diff_bias_z, s.suspicion_score,
               s.p_adj_yellow, s.p_adj_fouls,
               s.sig_yellow_fdr, s.sig_fouls_fdr
        FROM referee_bias_scores s
        JOIN referees r USING (referee_id)
        ORDER BY r.name, s.season
    """)


@st.cache_data(ttl=300)
def load_home_bias_summary():
    """Resumo de home bias por árbitro — pressure_bias_index acumulado."""
    return query("""
        SELECT
            r.name                              AS referee,
            h.season,
            h.score_context,
            h.n_games,
            h.media_fouls_home,
            h.media_fouls_away,
            h.media_fouls_diff,
            h.media_yc_home,
            h.media_yc_away,
            h.media_yc_diff,
            h.home_bias_fouls_z,
            h.home_bias_yellow_z,
            h.pressure_bias_index
        FROM referee_home_bias h
        JOIN referees r USING (referee_id)
        ORDER BY r.name, h.season, h.score_context
    """)


@st.cache_data(ttl=300)
def load_home_bias_referee(referee_id: int):
    """Home bias para um árbitro específico."""
    return query("""
        SELECT season, score_context, n_games,
               media_fouls_home, media_fouls_away, media_fouls_diff,
               media_yc_home, media_yc_away, media_yc_diff,
               home_bias_fouls_z, home_bias_yellow_z, pressure_bias_index
        FROM referee_home_bias
        WHERE referee_id = %s
        ORDER BY season, score_context
    """, (referee_id,))

@st.cache_data(ttl=300)
def load_top5_europe():
    """
    Calcula a classificação final por época e devolve as 5 primeiras equipas
    com as suas estatísticas de faltas e cartões nessa época.
    Pontos: vitória=3, empate=1, derrota=0.
    """
    return query("""
        WITH resultados AS (
            -- Uma linha por equipa por jogo com pontos
            SELECT
                s.label                 AS epoca,
                s.season_id,
                m.home_team_id          AS team_id,
                CASE
                    WHEN m.home_goals > m.away_goals THEN 3
                    WHEN m.home_goals = m.away_goals THEN 1
                    ELSE 0
                END                     AS pontos,
                m.home_goals            AS golos_marcados,
                m.away_goals            AS golos_sofridos,
                m.match_id
            FROM matches m
            JOIN seasons s USING (season_id)
            WHERE s.label IN ('2017/18','2018/19','2019/20','2020/21',
                              '2021/22','2022/23','2023/24')
            UNION ALL
            SELECT
                s.label, s.season_id,
                m.away_team_id,
                CASE
                    WHEN m.away_goals > m.home_goals THEN 3
                    WHEN m.away_goals = m.home_goals THEN 1
                    ELSE 0
                END,
                m.away_goals,
                m.home_goals,
                m.match_id
            FROM matches m
            JOIN seasons s USING (season_id)
            WHERE s.label IN ('2017/18','2018/19','2019/20','2020/21',
                              '2021/22','2022/23','2023/24')
        ),
        classificacao AS (
            SELECT
                r.epoca,
                r.team_id,
                t.name                          AS equipa,
                COUNT(*)                        AS jogos,
                SUM(r.pontos)                   AS pontos,
                SUM(r.golos_marcados)           AS gm,
                SUM(r.golos_sofridos)           AS gs,
                SUM(r.golos_marcados) - SUM(r.golos_sofridos) AS dg,
                RANK() OVER (
                    PARTITION BY r.epoca
                    ORDER BY SUM(r.pontos) DESC,
                             SUM(r.golos_marcados) - SUM(r.golos_sofridos) DESC,
                             SUM(r.golos_marcados) DESC
                )                               AS pos
            FROM resultados r
            JOIN teams t ON t.team_id = r.team_id
            GROUP BY r.epoca, r.team_id, t.name
        ),
        top5 AS (
            SELECT * FROM classificacao WHERE pos <= 5
        )
        SELECT
            c.epoca,
            c.pos,
            c.equipa,
            c.pontos,
            c.jogos,
            c.gm,
            c.gs,
            c.dg,
            ROUND(AVG(ms.fouls)::numeric, 2)       AS media_faltas,
            SUM(ms.fouls)                           AS faltas_total,
            ROUND(AVG(ms.yellow_cards)::numeric, 3) AS media_amarelos,
            SUM(ms.yellow_cards)                    AS amarelos_total,
            SUM(ms.red_cards)                       AS vermelhos_total
        FROM top5 c
        JOIN match_stats ms ON ms.team_id = c.team_id
        JOIN matches m ON m.match_id = ms.match_id
        JOIN seasons s ON s.season_id = m.season_id AND s.label = c.epoca
        GROUP BY c.epoca, c.pos, c.equipa, c.pontos, c.jogos, c.gm, c.gs, c.dg
        ORDER BY c.epoca, c.pos
    """)


@st.cache_data(ttl=300)
def load_team_profile(team_id: int):
    """Agrega todas as perspectivas de viés para uma equipa específica."""
    return query("""
        SELECT
            r.name                              AS arbitro,
            r.referee_id,
            COUNT(DISTINCT ms.match_id)         AS jogos,
            ROUND(AVG(ms.fouls)::numeric,2)     AS media_faltas,
            ROUND(AVG(ms.yellow_cards)::numeric,3) AS media_amarelos,
            ROUND(AVG(ms.red_cards)::numeric,4) AS media_vermelhos,
            -- desvio de faltas face à média geral do árbitro
            ROUND((AVG(ms.fouls) - ref_avg.avg_fouls)::numeric, 2)   AS desvio_faltas,
            -- desvio de amarelos face à média geral do árbitro
            ROUND((AVG(ms.yellow_cards) - ref_avg.avg_yellows)::numeric, 3) AS desvio_amarelos,
            ref_avg.avg_fouls                   AS media_geral_faltas_arbitro,
            ref_avg.avg_yellows                 AS media_geral_amarelos_arbitro
        FROM match_stats ms
        JOIN matches m ON m.match_id = ms.match_id
        JOIN referees r ON r.referee_id = m.referee_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN (
            -- média geral de cada árbitro em todos os jogos
            SELECT
                m2.referee_id,
                AVG(ms2.fouls)       AS avg_fouls,
                AVG(ms2.yellow_cards) AS avg_yellows
            FROM match_stats ms2
            JOIN matches m2 ON m2.match_id = ms2.match_id
            JOIN referees r2 ON r2.referee_id = m2.referee_id
            JOIN seasons s2 ON s2.season_id = m2.season_id
            WHERE r2.referee_id > 1
              AND s2.label IN ('2017/18','2018/19','2019/20','2020/21',
                               '2021/22','2022/23','2023/24')
            GROUP BY m2.referee_id
        ) ref_avg ON ref_avg.referee_id = r.referee_id
        WHERE ms.team_id = %s
          AND r.referee_id > 1
          AND s.label IN ('2017/18','2018/19','2019/20','2020/21',
                          '2021/22','2022/23','2023/24')
        GROUP BY r.name, r.referee_id, ref_avg.avg_fouls, ref_avg.avg_yellows
        HAVING COUNT(DISTINCT ms.match_id) >= 3
        ORDER BY desvio_amarelos DESC
    """, (team_id,))


@st.cache_data(ttl=300)
def load_team_season_stats(team_id: int):
    """Estatísticas da equipa por época."""
    return query("""
        SELECT
            s.label                             AS epoca,
            COUNT(DISTINCT ms.match_id)         AS jogos,
            ROUND(AVG(ms.fouls)::numeric,2)     AS media_faltas,
            ROUND(AVG(ms.yellow_cards)::numeric,3) AS media_amarelos,
            ROUND(AVG(ms.red_cards)::numeric,4) AS media_vermelhos,
            SUM(ms.fouls)                       AS faltas_total,
            SUM(ms.yellow_cards)                AS amarelos_total
        FROM match_stats ms
        JOIN matches m ON m.match_id = ms.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN referees r ON r.referee_id = m.referee_id
        WHERE ms.team_id = %s
          AND r.referee_id > 1
          AND s.label IN ('2017/18','2018/19','2019/20','2020/21',
                          '2021/22','2022/23','2023/24')
        GROUP BY s.label
        ORDER BY s.label
    """, (team_id,))

@st.cache_data(ttl=300)
def load_teams_ranking():
    """Ranking de equipas por média de faltas e cartões sofridos — perspectiva da equipa."""
    return query("""
        SELECT
            t.name                              AS equipa,
            COUNT(DISTINCT ms.match_id)         AS jogos,
            SUM(ms.fouls)                       AS faltas_total,
            ROUND(AVG(ms.fouls)::numeric, 2)    AS media_faltas,
            SUM(ms.yellow_cards)                AS amarelos_total,
            ROUND(AVG(ms.yellow_cards)::numeric,3) AS media_amarelos,
            SUM(ms.red_cards)                   AS vermelhos_total,
            ROUND(AVG(ms.red_cards)::numeric,4) AS media_vermelhos,
            COUNT(DISTINCT m.referee_id)        AS n_arbitros,
            ROUND(
                (SUM(ms.yellow_cards)::float / NULLIF(SUM(ms.fouls), 0))::numeric, 4
            )                                   AS amarelos_por_falta
        FROM match_stats ms
        JOIN matches m USING (match_id)
        JOIN teams t ON t.team_id = ms.team_id
        JOIN seasons s USING (season_id)
        JOIN referees r ON r.referee_id = m.referee_id
        WHERE r.referee_id > 1
          AND s.label IN ('2017/18','2018/19','2019/20','2020/21',
                          '2021/22','2022/23','2023/24')
        GROUP BY t.name
        ORDER BY media_faltas DESC
    """)


@st.cache_data(ttl=300)
def load_teams_by_season():
    """Média de faltas e cartões por equipa por época."""
    return query("""
        SELECT
            t.name                              AS equipa,
            s.label                             AS epoca,
            COUNT(DISTINCT ms.match_id)         AS jogos,
            ROUND(AVG(ms.fouls)::numeric, 2)    AS media_faltas,
            ROUND(AVG(ms.yellow_cards)::numeric,3) AS media_amarelos,
            ROUND(AVG(ms.red_cards)::numeric,4) AS media_vermelhos
        FROM match_stats ms
        JOIN matches m USING (match_id)
        JOIN teams t ON t.team_id = ms.team_id
        JOIN seasons s USING (season_id)
        JOIN referees r ON r.referee_id = m.referee_id
        WHERE r.referee_id > 1
          AND s.label IN ('2017/18','2018/19','2019/20','2020/21',
                          '2021/22','2022/23','2023/24')
        GROUP BY t.name, s.label
        ORDER BY t.name, s.label
    """)




# ── Helpers de cor ────────────────────────────────────────────────────────────

def desvio_cor(val):
    if val > 0.8:   return "background-color: rgba(224,64,64,0.25)"
    if val > 0.4:   return "background-color: rgba(245,166,35,0.18)"
    if val < -0.4:  return "background-color: rgba(74,158,255,0.18)"
    return ""


def zscore_cor(val):
    try:
        v = float(val)
        if abs(v) >= 3:   return "color: #e04040; font-weight:600"
        if abs(v) >= 2:   return "color: #f5a623"
        if abs(v) >= 1.5: return "color: #d4d8de"
        return "color: #4e5663"
    except: return ""


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ CAL")
    st.markdown("**Criticar a Arbitragem Legalmente**")
    st.markdown("---")

    modo = st.radio(
        "Vista",
        ["🔍 Árbitro específico", "📊 Desvios extremos",
         "🗺️ Heatmap geral", "🏟️ Por equipa",
         "🔬 Perfil de equipa", "⚡ Alertas & Comparação", "❓ Ajuda"],
        index=0,
    )

    st.markdown("---")

    referees_df = load_referees()
    referee_names = referees_df["arbitro"].tolist()

    if modo == "🔍 Árbitro específico":
        selected_ref = st.selectbox("Árbitro", referee_names, index=0)
        ref_id = int(referees_df[referees_df["arbitro"] == selected_ref]["referee_id"].iloc[0])

    st.markdown("---")

    # ── Botão de actualização ────────────────────────────────────────────────
    st.markdown("#### 🔄 Actualizar dados")
    st.caption("Corre a pipeline completa para a época seleccionada.")
    epoca_update = st.text_input("Época a actualizar", value="2025/26")

    if st.button("▶ Actualizar agora", use_container_width=True, type="primary"):
        import subprocess
        steps = [
            ("📥 Ingestão football-data...",    ["python", "run_ingest.py",   epoca_update]),
            ("👤 Ingestão árbitros...",          ["python", "run_referees.py", epoca_update]),
            ("⚙️ Feature engineering...",        ["python", "run_features.py", epoca_update]),
            ("📊 Bias engine...",               ["python", "run_bias.py"]),
            ("🃏 Análise de cartões...",         ["python", "run_cards.py"]),
            ("📊 Histórico bias...",              ["python", "run_bias_history.py", epoca_update]),
            ("🏠 Home bias...",                    ["python", "run_home_bias.py",     epoca_update]),
        ]
        ok = True
        bar = st.progress(0)
        status = st.empty()
        for i, (label, cmd) in enumerate(steps):
            status.info(label)
            bar.progress((i + 1) / len(steps))
            res = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
            if res.returncode != 0:
                status.error(f"Erro: {label}\n{res.stderr[-400:]}")
                ok = False
                break
        bar.empty()
        if ok:
            status.success(f"✓ {epoca_update} actualizada com sucesso!")
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.caption("Dados: football-data.co.uk + Sofascore\nLiga Portugal · 2017/18–2023/24\n7 épocas completas")

    st.markdown("---")

    # ── Botão de relatório PDF ───────────────────────────────────────────────
    st.markdown("#### 📄 Relatório PDF")
    st.caption("Gera um PDF completo para o árbitro e época seleccionados.")

    epoca_pdf = st.text_input("Época (deixar vazio = todas)", value="", key="pdf_epoca",
                               placeholder="ex: 2023/24")
    if modo == "🔍 Árbitro específico" and st.button("📄 Gerar PDF", use_container_width=True):
        import subprocess, os
        args = ["python", "run_pdf_report.py", selected_ref]
        if epoca_pdf.strip():
            args.append(epoca_pdf.strip())
        with st.spinner("A gerar PDF..."):
            res = subprocess.run(args, capture_output=True, text=True, cwd="/app")
        if res.returncode == 0:
            fname = None
            for line in res.stdout.strip().split("\n"):
                if line.startswith("✓ PDF gerado:"):
                    fname = os.path.basename(line.replace("✓ PDF gerado:", "").strip())
                    break
            if fname:
                dest = "/downloads" if os.path.isdir("/downloads") else "/app/reports"
                st.success(f"✓ PDF guardado em {dest}/{fname}")
        else:
            st.error(f"Erro ao gerar PDF: {res.stderr[-300:]}")
    elif modo != "🔍 Árbitro específico":
        st.caption("Seleccionar a vista 🔍 Árbitro específico para activar.")


# ── MODO 1: ÁRBITRO ESPECÍFICO ────────────────────────────────────────────────

if modo == "🔍 Árbitro específico":

    total_df    = load_total(ref_id)
    season_df   = load_by_season(ref_id)
    bias_df     = load_bias_scores(ref_id)

    if total_df.empty:
        st.warning("Sem dados suficientes para este árbitro.")
        st.stop()

    # Métricas de cabeçalho
    n_jogos    = int(total_df["jogos_total"].sum())
    n_equipas  = len(total_df)
    media_y    = float(total_df["media_amarelos"].mean())
    total_df["media_faltas"] = pd.to_numeric(total_df.get("media_faltas", 0), errors="coerce").fillna(0)
    total_df["desvio_faltas_vs_media"] = pd.to_numeric(total_df.get("desvio_faltas_vs_media", 0), errors="coerce").fillna(0)
    total_df["percentil_faltas"] = pd.to_numeric(total_df.get("percentil_faltas", 0), errors="coerce").fillna(0)
    total_df["ranking_faltas"] = pd.to_numeric(total_df.get("ranking_faltas", 0), errors="coerce").fillna(0)
    media_f    = float(total_df["media_faltas"].mean())
    max_desv_y = float(total_df["desvio_vs_media_arbitro"].abs().max())
    max_desv_f = float(total_df["desvio_faltas_vs_media"].abs().max())
    n_ext_y    = int((total_df["desvio_vs_media_arbitro"].abs() > 0.8).sum())
    n_ext_f    = int((total_df["desvio_faltas_vs_media"].abs() > 2.0).sum())

    st.markdown(f"# {selected_ref}")
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Jogos arbitrados", n_jogos)
    c2.metric("Equipas", n_equipas)
    c3.metric("Média amarelos/jogo", f"{media_y:.2f}")
    c4.metric("Média faltas/jogo", f"{media_f:.1f}")
    c5.metric("Desvio amarelos máx", f"{max_desv_y:+.2f}", delta="⚠️" if n_ext_y > 0 else None, delta_color="inverse" if n_ext_y > 0 else "off")
    c6.metric("Desvio faltas máx", f"{max_desv_f:+.1f}", delta="⚠️" if n_ext_f > 0 else None, delta_color="inverse" if n_ext_f > 0 else "off")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    # ── Tabs: Amarelos | Faltas ──────────────────────────────────────────────
    with col_left:
        tab_y, tab_f = st.tabs(["🟡 Amarelos", "🦵 Faltas"])

        def make_display_y(df):
            df = df.copy()
            if "amarelos_por_falta" in df.columns:
                df["amarelos_por_falta"] = pd.to_numeric(df["amarelos_por_falta"], errors="coerce").fillna(0)
            else:
                df["amarelos_por_falta"] = 0.0
            d = df[["equipa","jogos_total","amarelos_total","media_amarelos",
                     "desvio_vs_media_arbitro","percentil_amarelos","ranking_amarelos",
                     "amarelos_por_falta","vermelhos_total","media_vermelhos"]].copy()
            d.columns = ["Equipa","Jogos","Amarelos","Média/jogo","Desvio","Pct","Rank","Amar/Falta","VM","VM/jogo"]
            d = d.sort_values("Desvio", ascending=False)
            for c in ["Média/jogo","Desvio","VM/jogo"]:
                d[c] = d[c].apply(lambda x: round(float(x),3))
            d["Amar/Falta"] = d["Amar/Falta"].apply(lambda x: f"{float(x):.4f}")
            d["Pct"] = d["Pct"].apply(lambda x: f"{float(x):.0f}%")
            return d

        def make_display_f(df):
            d = df[["equipa","jogos_total","faltas_total","media_faltas",
                     "desvio_faltas_vs_media","percentil_faltas","ranking_faltas"]].copy()
            d.columns = ["Equipa","Jogos","Faltas","Média/jogo","Desvio","Pct","Rank"]
            d = d.sort_values("Desvio", ascending=False)
            for c in ["Média/jogo","Desvio"]:
                d[c] = d[c].apply(lambda x: round(float(x),2))
            d["Pct"] = d["Pct"].apply(lambda x: f"{float(x):.0f}%")
            return d

        def style_row(row):
            try: d = float(row["Desvio"])
            except: return [""] * len(row)
            if d > 0.8 or d > 2:   bg = "background-color: rgba(214,96,77,0.18)"
            elif d > 0.4 or d > 1: bg = "background-color: rgba(245,166,77,0.12)"
            elif d < -0.4 or d < -1: bg = "background-color: rgba(67,147,195,0.1)"
            else: bg = ""
            return [bg] * len(row)

        with tab_y:
            dy = make_display_y(total_df)
            st.dataframe(dy.style.apply(style_row, axis=1),
                         use_container_width=True, height=430, hide_index=True)

        with tab_f:
            df_f = total_df.copy()
            for c in ["media_faltas","desvio_faltas_vs_media","percentil_faltas","ranking_faltas"]:
                if c in df_f.columns:
                    df_f[c] = pd.to_numeric(df_f[c], errors="coerce").fillna(0)
            dff = make_display_f(df_f)
            st.dataframe(dff.style.apply(style_row, axis=1),
                         use_container_width=True, height=430, hide_index=True)

    # ── Gráfico de barras horizontais ────────────────────────────────────────
    with col_right:
        metrica = st.radio("Métrica", ["🟡 Amarelos", "🦵 Faltas"], horizontal=True, key="metrica_bar")

        plot_df = total_df.copy()
        if metrica == "🟡 Amarelos":
            plot_df["_val"]  = plot_df["desvio_vs_media_arbitro"].astype(float)
            plot_df["_label"] = plot_df["_val"].apply(lambda x: f"{x:+.2f}")
            xtitle = "Desvio amarelos/jogo vs média árbitro"
        else:
            plot_df["_val"]  = plot_df["desvio_faltas_vs_media"].astype(float)
            plot_df["_label"] = plot_df["_val"].apply(lambda x: f"{x:+.1f}")
            xtitle = "Desvio faltas/jogo vs média árbitro"

        plot_df = plot_df.sort_values("_val")
        thr_hi = 0.8 if metrica == "🟡 Amarelos" else 2.0
        thr_lo = 0.4 if metrica == "🟡 Amarelos" else 1.0

        colors = ["#d6604d" if d > thr_hi else "#f5a623" if d > thr_lo
                  else "#4393c3" if d < -thr_lo else "#aab4c0"
                  for d in plot_df["_val"]]

        fig = go.Figure(go.Bar(
            x=plot_df["_val"],
            y=plot_df["equipa"],
            orientation="h",
            marker_color=colors,
            text=plot_df["_label"],
            textposition="outside",
            textfont=dict(size=10, color="#1a1d23"),
        ))
        fig.add_vline(x=0, line_color="#888", line_width=1)
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f5f7fa",
            font=dict(color="#1a1d23", size=11),
            margin=dict(l=10, r=50, t=10, b=10),
            height=430,
            xaxis=dict(title=xtitle, gridcolor="#d0d7e0",
                       zerolinecolor="#888", title_font_size=11),
            yaxis=dict(gridcolor="#d0d7e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Z-scores por época ───────────────────────────────────────────────────
    if not bias_df.empty:
        st.markdown("#### Z-scores de viés por época")
        for col in ["p_adj_yellow","p_adj_fouls","sig_yellow_fdr","sig_fouls_fdr"]:
            if col not in bias_df.columns:
                bias_df[col] = None

        # Construir tabela com colunas FDR
        bias_display = bias_df.copy()
        # Renomear colunas base (as primeiras 6 que vêm da query)
        base_cols = [c for c in bias_df.columns if c in [
            "season","n_games","yellow_diff_bias_z","red_diff_bias_z",
            "fouls_diff_bias_z","suspicion_score"]]
        bd = bias_df[base_cols].copy()
        bd.columns = ["Época","Jogos","Z Amarelos","Z Vermelhos","Z Faltas","SuspicionScore"]
        for col in ["Z Amarelos","Z Vermelhos","Z Faltas","SuspicionScore"]:
            bd[col] = bd[col].apply(lambda x: round(float(x), 3))

        # Adicionar badge FDR
        def fdr_badge(row):
            sig_y = bias_df.loc[row.name, "sig_yellow_fdr"] if "sig_yellow_fdr" in bias_df.columns else None
            sig_f = bias_df.loc[row.name, "sig_fouls_fdr"]  if "sig_fouls_fdr"  in bias_df.columns else None
            parts = []
            if sig_f  is True: parts.append("faltas ✓")
            if sig_y  is True: parts.append("amarelos ✓")
            return ", ".join(parts) if parts else "—"
        bd["FDR sig."] = bd.apply(fdr_badge, axis=1)

        def style_zscore(val):
            try:
                v = abs(float(val))
                if v >= 3:   return "color: #e04040; font-weight: 600"
                if v >= 2.5: return "color: #f5a623"
                if v >= 1.5: return "color: #d4d8de"
                return "color: #4e5663"
            except: return ""

        def style_fdr(val):
            if val and val != "—":
                return "color: #2e7d32; font-weight: 600"
            return "color: #999999"

        st.dataframe(
            bd.style
              .map(style_zscore, subset=["Z Amarelos","Z Vermelhos","Z Faltas","SuspicionScore"])
              .map(style_fdr, subset=["FDR sig."]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "**FDR sig.** = estatisticamente significativo após correcção de múltiplos testes "
            "(Benjamini-Hochberg, FDR 5%). Apenas estas métricas têm evidência robusta. "
            "Z-scores sem ✓ podem ser ruído estatístico."
        )

    # ── Evolução por época das 5 equipas mais extremas ───────────────────────
    if not season_df.empty:
        st.markdown("#### Evolução temporal — top 5 equipas com maior desvio absoluto")

        top5 = total_df.assign(
            desvio_vs_media_arbitro=total_df["desvio_vs_media_arbitro"].astype(float)
        ).nlargest(3, "desvio_vs_media_arbitro")["equipa"].tolist() + \
               total_df.assign(
            desvio_vs_media_arbitro=total_df["desvio_vs_media_arbitro"].astype(float)
        ).nsmallest(2, "desvio_vs_media_arbitro")["equipa"].tolist()

        evo_df = season_df[season_df["equipa"].isin(top5)].copy()
        evo_df["media_amarelos"] = evo_df["media_amarelos"].astype(float)

        if not evo_df.empty:
            fig2 = px.line(
                evo_df,
                x="epoca", y="media_amarelos",
                color="equipa",
                markers=True,
                labels={"media_amarelos": "Amarelos/jogo", "epoca": "Época"},
            )
            fig2.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23"),
                legend=dict(bgcolor="#f0f2f5", bordercolor="#d0d7e0"),
                xaxis=dict(gridcolor="#d0d7e0"),
                yaxis=dict(gridcolor="#d0d7e0"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── MODO 2: DESVIOS EXTREMOS ──────────────────────────────────────────────────

elif modo == "📊 Desvios extremos":

    st.markdown("# Desvios Extremos")
    st.markdown("---")

    ext_df  = load_extreme_desvios()
    ext_f   = load_extreme_faltas()

    tab_ey, tab_ef = st.tabs(["🟡 Amarelos — desvio > ±0.7/jogo", "🦵 Faltas — desvio > ±2/jogo"])

    with tab_ey:
        if ext_df.empty:
            st.info("Nenhum desvio extremo encontrado.")
        else:
            ext_df["media_amarelos"]          = ext_df["media_amarelos"].astype(float)
            ext_df["desvio_vs_media_arbitro"] = ext_df["desvio_vs_media_arbitro"].astype(float)
            ext_df["percentil_amarelos"]      = ext_df["percentil_amarelos"].astype(float)

            pos = ext_df[ext_df["desvio_vs_media_arbitro"] > 0]
            neg = ext_df[ext_df["desvio_vs_media_arbitro"] < 0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total pares extremos", len(ext_df[ext_df["desvio_vs_media_arbitro"].abs() > 0.7]))
            c2.metric("Árbitros que dão MAIS", len(pos))
            c3.metric("Árbitros que dão MENOS", len(neg))

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                sentido = st.selectbox("Sentido", ["Todos", "Dá mais (+)", "Dá menos (-)"], key="sy")
            with col_f2:
                arbs = ["Todos"] + sorted(ext_df["arbitro"].unique().tolist())
                filtro_arb = st.selectbox("Árbitro", arbs, key="ay")
            with col_f3:
                eqs = ["Todas"] + sorted(ext_df["equipa"].unique().tolist())
                filtro_eq = st.selectbox("Equipa", eqs, key="ey")

            filt = ext_df[ext_df["desvio_vs_media_arbitro"].abs() > 0.7].copy()
            if sentido == "Dá mais (+)": filt = filt[filt["desvio_vs_media_arbitro"] > 0]
            elif sentido == "Dá menos (-)": filt = filt[filt["desvio_vs_media_arbitro"] < 0]
            if filtro_arb != "Todos": filt = filt[filt["arbitro"] == filtro_arb]
            if filtro_eq  != "Todas": filt = filt[filt["equipa"]  == filtro_eq]

            if "amarelos_por_falta" not in filt.columns:
                filt["amarelos_por_falta"] = 0.0
            else:
                filt["amarelos_por_falta"] = pd.to_numeric(filt["amarelos_por_falta"], errors="coerce").fillna(0)
            disp = filt[["arbitro","equipa","jogos_total","epocas",
                          "media_amarelos","desvio_vs_media_arbitro","percentil_amarelos","amarelos_por_falta"]].copy()
            disp.columns = ["Árbitro","Equipa","Jogos","Épocas","Média/jogo","Desvio","Pct","Amar/Falta"]
            disp["Desvio"]    = disp["Desvio"].round(3)
            disp["Média/jogo"] = disp["Média/jogo"].round(3)
            disp["Pct"]       = disp["Pct"].apply(lambda x: f"{x:.0f}%")
            if "Amar/Falta" in disp.columns:
                disp["Amar/Falta"] = disp["Amar/Falta"].apply(lambda x: f"{float(x):.4f}")

            def sty_y(row):
                d = float(row["Desvio"])
                if d > 1.0: return ["background-color: rgba(214,96,77,0.22)"] * len(row)
                if d > 0.7: return ["background-color: rgba(245,166,35,0.15)"] * len(row)
                if d < -0.7: return ["background-color: rgba(67,147,195,0.15)"] * len(row)
                return [""] * len(row)

            st.dataframe(disp.style.apply(sty_y, axis=1),
                         use_container_width=True, height=500, hide_index=True)

    with tab_ef:
        if ext_f.empty:
            st.info("Nenhum desvio extremo de faltas encontrado.")
        else:
            ext_f["media_faltas"]         = ext_f["media_faltas"].astype(float)
            ext_f["desvio_faltas_vs_media"] = ext_f["desvio_faltas_vs_media"].astype(float)
            ext_f["percentil_faltas"]     = ext_f["percentil_faltas"].astype(float)

            pos_f = ext_f[ext_f["desvio_faltas_vs_media"] > 0]
            neg_f = ext_f[ext_f["desvio_faltas_vs_media"] < 0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total pares extremos", len(ext_f))
            c2.metric("Árbitros que assinalam MAIS", len(pos_f))
            c3.metric("Árbitros que assinalam MENOS", len(neg_f))

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                sentido_f = st.selectbox("Sentido", ["Todos", "Mais faltas (+)", "Menos faltas (-)"], key="sf")
            with col_f2:
                arbs_f = ["Todos"] + sorted(ext_f["arbitro"].unique().tolist())
                filtro_arb_f = st.selectbox("Árbitro", arbs_f, key="af")
            with col_f3:
                eqs_f = ["Todas"] + sorted(ext_f["equipa"].unique().tolist())
                filtro_eq_f = st.selectbox("Equipa", eqs_f, key="ef")

            filt_f = ext_f.copy()
            if sentido_f == "Mais faltas (+)": filt_f = filt_f[filt_f["desvio_faltas_vs_media"] > 0]
            elif sentido_f == "Menos faltas (-)": filt_f = filt_f[filt_f["desvio_faltas_vs_media"] < 0]
            if filtro_arb_f != "Todos": filt_f = filt_f[filt_f["arbitro"] == filtro_arb_f]
            if filtro_eq_f  != "Todas": filt_f = filt_f[filt_f["equipa"]  == filtro_eq_f]

            disp_f = filt_f[["arbitro","equipa","jogos_total","epocas",
                              "media_faltas","desvio_faltas_vs_media","percentil_faltas"]].copy()
            disp_f.columns = ["Árbitro","Equipa","Jogos","Épocas","Média faltas/jogo","Desvio","Pct"]
            disp_f["Desvio"]            = disp_f["Desvio"].round(2)
            disp_f["Média faltas/jogo"] = disp_f["Média faltas/jogo"].round(1)
            disp_f["Pct"]               = disp_f["Pct"].apply(lambda x: f"{x:.0f}%")

            def sty_f(row):
                d = float(row["Desvio"])
                if d > 3:  return ["background-color: rgba(214,96,77,0.22)"] * len(row)
                if d > 2:  return ["background-color: rgba(245,166,35,0.15)"] * len(row)
                if d < -2: return ["background-color: rgba(67,147,195,0.15)"] * len(row)
                return [""] * len(row)

            st.dataframe(disp_f.style.apply(sty_f, axis=1),
                         use_container_width=True, height=500, hide_index=True)

# ── MODO 3: HEATMAP GERAL ─────────────────────────────────────────────────────

elif modo == "🗺️ Heatmap geral":

    st.markdown("# Heatmap — Árbitro × Equipa")
    st.markdown("---")

    all_df = load_total()
    if all_df.empty:
        st.info("Sem dados.")
        st.stop()

    metrica_hm = st.radio("Métrica", ["🟡 Amarelos", "🦵 Faltas"], horizontal=True, key="hm_metrica")

    all_df["media_amarelos"] = all_df["media_amarelos"].astype(float)
    all_df["desvio_y"]       = all_df["desvio_vs_media_arbitro"].astype(float)
    all_df["desvio_f"]       = pd.to_numeric(all_df.get("desvio_faltas_vs_media", 0), errors="coerce").fillna(0)

    if metrica_hm == "🟡 Amarelos":
        all_df["desvio"] = all_df["desvio_y"]
        hm_title = "Desvio de amarelos/jogo vs média do árbitro"
        hm_note  = "Vermelho = dá mais amarelos a esta equipa. Azul = dá menos."
    else:
        all_df["desvio"] = all_df["desvio_f"]
        hm_title = "Desvio de faltas/jogo vs média do árbitro"
        hm_note  = "Vermelho = assinala mais faltas a esta equipa. Azul = assinala menos."

    # Filtrar árbitros com >= 5 equipas
    counts = all_df.groupby("arbitro")["equipa"].count()
    arbs_ok = counts[counts >= 5].index.tolist()
    plot_df = all_df[all_df["arbitro"].isin(arbs_ok)]

    pivot = plot_df.pivot_table(
        index="arbitro", columns="equipa",
        values="desvio", aggfunc="mean"
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#2166ac"],   # azul forte — desvio muito negativo
            [0.25, "#92c5de"],   # azul claro
            [0.5,  "#f7f7f7"],   # branco — sem desvio
            [0.75, "#f4a582"],   # laranja claro
            [1.0,  "#d6604d"],   # vermelho — desvio muito positivo
        ],
        zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else ""
               for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(
            title="Desvio",
            tickfont=dict(color="#1a1d23"),
            titlefont=dict(color="#1a1d23"),
        ),
        hoverongaps=False,
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f5f7fa",
        font=dict(color="#1a1d23", size=10),
        margin=dict(l=10, r=10, t=10, b=120),
        height=max(400, len(pivot) * 28),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(hm_note + " Células vazias = menos de 3 jogos em conjunto.")


# ── MODO 4: POR EQUIPA ────────────────────────────────────────────────────────

elif modo == "🏟️ Por equipa":

    st.markdown("# Ranking de Equipas")
    st.markdown("Faltas assinaladas e cartões dados **contra** cada equipa. Acumulado 7 épocas completas (2017/18–2023/24).")
    st.markdown("---")

    teams_df = load_teams_ranking()
    teams_by_s = load_teams_by_season()

    if teams_df.empty:
        st.info("Sem dados.")
        st.stop()

    for col in ["media_faltas","media_amarelos","media_vermelhos",
                "faltas_total","amarelos_total","vermelhos_total","jogos"]:
        teams_df[col] = pd.to_numeric(teams_df[col], errors="coerce").fillna(0)

    # ── Métricas globais ─────────────────────────────────────────────────────
    media_global_f = float(teams_df["media_faltas"].mean())
    media_global_y = float(teams_df["media_amarelos"].mean())
    equipa_mais_faltas = teams_df.iloc[0]["equipa"]
    equipa_mais_y      = teams_df.sort_values("media_amarelos", ascending=False).iloc[0]["equipa"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Média global de faltas/jogo", f"{media_global_f:.1f}")
    c2.metric("Equipa com mais faltas assinaladas", equipa_mais_faltas)
    c3.metric("Média global de amarelos/jogo", f"{media_global_y:.2f}")
    c4.metric("Equipa com mais amarelos", equipa_mais_y)

    st.markdown("---")

    tab_f, tab_y, tab_evo, tab_eur = st.tabs([
        "🦵 Ranking por Faltas", "🟡 Ranking por Amarelos",
        "📈 Evolução por época", "🏆 Top 5 — Lugares Europeus"
    ])

    with tab_f:
        st.markdown("##### Equipas ordenadas por média de faltas assinaladas contra si por jogo")

        df_f = teams_df.copy().sort_values("media_faltas", ascending=False).reset_index(drop=True)
        df_f.index += 1
        df_f["desvio_vs_media"] = (df_f["media_faltas"] - media_global_f).round(2)

        disp_f = df_f[["equipa","jogos","faltas_total","media_faltas",
                        "desvio_vs_media","n_arbitros"]].copy()
        disp_f.columns = ["Equipa","Jogos","Total faltas","Média/jogo","Desvio vs média","Árbitros"]
        disp_f["Média/jogo"] = disp_f["Média/jogo"].round(2)

        def sty_team_f(row):
            d = float(row["Desvio vs média"])
            if d > 2:   return ["background-color: rgba(214,96,77,0.18)"] * len(row)
            if d > 1:   return ["background-color: rgba(245,166,35,0.12)"] * len(row)
            if d < -1:  return ["background-color: rgba(67,147,195,0.12)"] * len(row)
            return [""] * len(row)

        col_t, col_g = st.columns([2, 3])
        with col_t:
            st.dataframe(disp_f.style.apply(sty_team_f, axis=1),
                         use_container_width=True, height=520)

        with col_g:
            fig_tf = go.Figure(go.Bar(
                x=df_f["media_faltas"],
                y=df_f["equipa"],
                orientation="h",
                marker_color=["#d6604d" if d > 2 else "#f5a623" if d > 1
                              else "#4393c3" if d < -1 else "#aab4c0"
                              for d in df_f["desvio_vs_media"]],
                text=df_f["media_faltas"].apply(lambda x: f"{x:.1f}"),
                textposition="outside",
                textfont=dict(size=9, color="#1a1d23"),
            ))
            fig_tf.add_vline(x=media_global_f, line_color="#888",
                             line_dash="dash", line_width=1,
                             annotation_text=f"Média: {media_global_f:.1f}",
                             annotation_font_size=10)
            fig_tf.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23", size=10),
                height=520, margin=dict(l=10, r=50, t=10, b=10),
                xaxis=dict(title="Faltas/jogo", gridcolor="#d0d7e0"),
                yaxis=dict(autorange="reversed", gridcolor="#d0d7e0"),
            )
            st.plotly_chart(fig_tf, use_container_width=True)

    with tab_y:
        st.markdown("##### Equipas ordenadas por média de cartões amarelos recebidos por jogo")

        df_y = teams_df.copy().sort_values("media_amarelos", ascending=False).reset_index(drop=True)
        df_y.index += 1
        df_y["desvio_vs_media_y"] = (df_y["media_amarelos"] - media_global_y).round(3)

        if "amarelos_por_falta" not in df_y.columns:
            df_y["amarelos_por_falta"] = 0.0
        else:
            df_y["amarelos_por_falta"] = pd.to_numeric(df_y["amarelos_por_falta"], errors="coerce").fillna(0)
        disp_y = df_y[["equipa","jogos","amarelos_total","media_amarelos",
                        "desvio_vs_media_y","amarelos_por_falta","vermelhos_total","media_vermelhos","n_arbitros"]].copy()
        disp_y.columns = ["Equipa","Jogos","Amarelos","Média/jogo","Desvio","Amar/Falta","Vermelhos","VM/jogo","Árbitros"]
        disp_y["Média/jogo"]   = disp_y["Média/jogo"].round(3)
        disp_y["VM/jogo"]      = disp_y["VM/jogo"].round(4)
        disp_y["Amar/Falta"]   = disp_y["Amar/Falta"].apply(lambda x: f"{float(x):.4f}")

        def sty_team_y(row):
            d = float(row["Desvio"])
            if d > 0.5:  return ["background-color: rgba(214,96,77,0.18)"] * len(row)
            if d > 0.2:  return ["background-color: rgba(245,166,35,0.12)"] * len(row)
            if d < -0.2: return ["background-color: rgba(67,147,195,0.12)"] * len(row)
            return [""] * len(row)

        col_t2, col_g2 = st.columns([2, 3])
        with col_t2:
            st.dataframe(disp_y.style.apply(sty_team_y, axis=1),
                         use_container_width=True, height=520)
        with col_g2:
            fig_ty = go.Figure(go.Bar(
                x=df_y["media_amarelos"],
                y=df_y["equipa"],
                orientation="h",
                marker_color=["#d6604d" if d > 0.5 else "#f5a623" if d > 0.2
                              else "#4393c3" if d < -0.2 else "#aab4c0"
                              for d in df_y["desvio_vs_media_y"]],
                text=df_y["media_amarelos"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                textfont=dict(size=9, color="#1a1d23"),
            ))
            fig_ty.add_vline(x=media_global_y, line_color="#888",
                             line_dash="dash", line_width=1,
                             annotation_text=f"Média: {media_global_y:.2f}",
                             annotation_font_size=10)
            fig_ty.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23", size=10),
                height=520, margin=dict(l=10, r=50, t=10, b=10),
                xaxis=dict(title="Amarelos/jogo", gridcolor="#d0d7e0"),
                yaxis=dict(autorange="reversed", gridcolor="#d0d7e0"),
            )
            st.plotly_chart(fig_ty, use_container_width=True)

    with tab_evo:
        st.markdown("##### Evolução por época — seleccionar equipa e métrica")

        teams_by_s_df = teams_by_s.copy()
        for col in ["media_faltas","media_amarelos","media_vermelhos","jogos"]:
            teams_by_s_df[col] = pd.to_numeric(teams_by_s_df[col], errors="coerce").fillna(0)

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            team_list = sorted(teams_by_s_df["equipa"].unique().tolist())
            selected_teams = st.multiselect("Equipas", team_list,
                                            default=team_list[:5] if len(team_list) >= 5 else team_list)
        with col_sel2:
            metrica_evo = st.radio("Métrica", ["🦵 Faltas", "🟡 Amarelos"], horizontal=True, key="evo_m")

        evo_col = "media_faltas" if metrica_evo == "🦵 Faltas" else "media_amarelos"
        y_label = "Faltas/jogo" if metrica_evo == "🦵 Faltas" else "Amarelos/jogo"

        evo_filtered = teams_by_s_df[teams_by_s_df["equipa"].isin(selected_teams)]
        if not evo_filtered.empty:
            fig_evo = px.line(evo_filtered, x="epoca", y=evo_col, color="equipa",
                              markers=True, labels={evo_col: y_label, "epoca": "Época"})
            fig_evo.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23"),
                legend=dict(bgcolor="#f0f2f5", bordercolor="#d0d7e0"),
                xaxis=dict(gridcolor="#d0d7e0"),
                yaxis=dict(gridcolor="#d0d7e0"),
                height=400, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_evo, use_container_width=True)

    with tab_eur:
        st.markdown("##### Top 5 por época — equipas com acesso a competições europeias")
        st.caption("Classificação calculada por pontos (V=3, E=1, D=0). Faltas e cartões referem-se aos jogos da Liga.")

        eur_df = load_top5_europe()
        if eur_df.empty:
            st.info("Sem dados.")
        else:
            for col in ["media_faltas","media_amarelos","faltas_total",
                        "amarelos_total","vermelhos_total","pontos","gm","gs","dg"]:
                eur_df[col] = pd.to_numeric(eur_df[col], errors="coerce").fillna(0)

            # ── Tabela pivot por época ────────────────────────────────────────
            epocas = sorted(eur_df["epoca"].unique())
            sel_epoca = st.selectbox("Época", ["Todas as épocas"] + epocas, key="eur_epoca")

            if sel_epoca != "Todas as épocas":
                df_show = eur_df[eur_df["epoca"] == sel_epoca].copy()
            else:
                df_show = eur_df.copy()

            disp_eur = df_show[[
                "epoca","pos","equipa","pontos","jogos","gm","gs","dg",
                "media_faltas","faltas_total","media_amarelos","amarelos_total","vermelhos_total"
            ]].copy()
            disp_eur.columns = [
                "Época","Pos","Equipa","Pontos","Jogos","GM","GS","DG",
                "Faltas/jogo","Total faltas","Amarelos/jogo","Total amarelos","Vermelhos"
            ]
            disp_eur["Faltas/jogo"]    = disp_eur["Faltas/jogo"].round(1)
            disp_eur["Amarelos/jogo"]  = disp_eur["Amarelos/jogo"].round(3)

            # Médias da liga para comparação
            global_f = float(teams_df["media_faltas"].mean())
            global_y = float(teams_df["media_amarelos"].mean())

            def sty_eur(row):
                f = float(row["Faltas/jogo"])
                y = float(row["Amarelos/jogo"])
                if f > global_f + 1.5 or y > global_y + 0.4:
                    return ["background-color: rgba(214,96,77,0.15)"] * len(row)
                if f < global_f - 1.5 or y < global_y - 0.4:
                    return ["background-color: rgba(67,147,195,0.12)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                disp_eur.style.apply(sty_eur, axis=1),
                use_container_width=True,
                height=min(600, len(disp_eur) * 38 + 50),
                hide_index=True,
            )

            st.caption(f"Médias da liga: {global_f:.1f} faltas/jogo | {global_y:.2f} amarelos/jogo. "
                       f"Vermelho = acima da média. Azul = abaixo.")

            # ── Gráfico comparativo por época ─────────────────────────────────
            if sel_epoca == "Todas as épocas":
                st.markdown("##### Evolução de faltas por época — top 5")
                metrica_eur = st.radio("Métrica", ["🦵 Faltas/jogo", "🟡 Amarelos/jogo"],
                                       horizontal=True, key="eur_m")
                col_eur = "media_faltas" if "Faltas" in metrica_eur else "media_amarelos"

                fig_eur = px.line(
                    eur_df, x="epoca", y=col_eur, color="equipa",
                    markers=True,
                    labels={col_eur: metrica_eur.split(" ")[1], "epoca": "Época"},
                    title="Apenas equipas que terminaram no top 5"
                )
                fig_eur.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                    font=dict(color="#1a1d23"),
                    legend=dict(bgcolor="#f0f2f5", bordercolor="#d0d7e0"),
                    xaxis=dict(gridcolor="#d0d7e0"),
                    yaxis=dict(gridcolor="#d0d7e0"),
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_eur, use_container_width=True)


# ── MODO 5: PERFIL DE EQUIPA ─────────────────────────────────────────────────

elif modo == "🔬 Perfil de equipa":

    st.markdown("# 🔬 Perfil de Equipa")
    st.markdown("Agrega as três perspectivas de viés para uma equipa: como cada árbitro se comporta com ela, evolução histórica e posição no ranking da liga.")
    st.markdown("---")

    teams_df_all = load_teams()
    team_names   = teams_df_all["equipa"].tolist()

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        sel_team = st.selectbox("Seleccionar equipa", team_names)
        team_id  = int(teams_df_all[teams_df_all["equipa"] == sel_team]["team_id"].iloc[0])

    profile_df  = load_team_profile(team_id)
    season_df_t = load_team_season_stats(team_id)
    ranking_df  = load_teams_ranking()

    for col in ["media_faltas","media_amarelos","desvio_faltas","desvio_amarelos",
                "media_geral_faltas_arbitro","media_geral_amarelos_arbitro","jogos"]:
        if col in profile_df.columns:
            profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce").fillna(0)
    for col in ["media_faltas","media_amarelos","faltas_total","amarelos_total","jogos"]:
        if col in season_df_t.columns:
            season_df_t[col] = pd.to_numeric(season_df_t[col], errors="coerce").fillna(0)
    for col in ["media_faltas","media_amarelos"]:
        if col in ranking_df.columns:
            ranking_df[col] = pd.to_numeric(ranking_df[col], errors="coerce").fillna(0)

    # ── Posição no ranking geral ─────────────────────────────────────────────
    rank_row = ranking_df[ranking_df["equipa"] == sel_team]
    if not rank_row.empty:
        r = rank_row.iloc[0]
        total_equipas = len(ranking_df)
        rank_f = int(ranking_df.sort_values("media_faltas", ascending=False).reset_index(drop=True).index[ranking_df["equipa"] == sel_team][0]) + 1
        rank_y = int(ranking_df.sort_values("media_amarelos", ascending=False).reset_index(drop=True).index[ranking_df["equipa"] == sel_team][0]) + 1
        global_f = float(ranking_df["media_faltas"].mean())
        global_y = float(ranking_df["media_amarelos"].mean())
        desvio_f_liga = float(r["media_faltas"]) - global_f
        desvio_y_liga = float(r["media_amarelos"]) - global_y

        with col_info:
            st.markdown(f"**{sel_team}** — posição no ranking da liga (7 épocas completas)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faltas/jogo", f"{float(r['media_faltas']):.1f}",
                      delta=f"{desvio_f_liga:+.1f} vs liga", delta_color="inverse")
            c2.metric(f"Rank faltas ({total_equipas} eq.)", f"#{rank_f}",
                      delta="⚠️ mais faltas" if rank_f <= 5 else ("✓ menos faltas" if rank_f >= total_equipas-4 else None),
                      delta_color="inverse" if rank_f <= 5 else "normal")
            c3.metric("Amarelos/jogo", f"{float(r['media_amarelos']):.3f}",
                      delta=f"{desvio_y_liga:+.3f} vs liga", delta_color="inverse")
            c4.metric(f"Rank amarelos ({total_equipas} eq.)", f"#{rank_y}",
                      delta="⚠️ mais amarelos" if rank_y <= 5 else ("✓ menos amarelos" if rank_y >= total_equipas-4 else None),
                      delta_color="inverse" if rank_y <= 5 else "normal")

    st.markdown("---")

    tab_arb, tab_evo_t, tab_veredito = st.tabs([
        "⚖️ Por árbitro", "📈 Evolução por época", "🏁 Veredito"
    ])

    # ── Tab 1: Por árbitro ───────────────────────────────────────────────────
    with tab_arb:
        st.markdown(f"##### Como cada árbitro se comporta com o **{sel_team}** vs a sua própria média")
        st.caption("Desvio positivo (vermelho) = árbitro dá mais a esta equipa do que a sua média com outras equipas.")

        col_tl, col_tr = st.columns([3, 2])

        with col_tl:
            disp_p = profile_df[[
                "arbitro","jogos","media_faltas","desvio_faltas",
                "media_geral_faltas_arbitro","media_amarelos","desvio_amarelos",
                "media_geral_amarelos_arbitro"
            ]].copy()
            disp_p.columns = [
                "Árbitro","Jogos","Faltas/jogo","Desv.Faltas",
                "Média faltas árbitro","Amarelos/jogo","Desv.Amarelos",
                "Média amar. árbitro"
            ]
            for c in ["Faltas/jogo","Desv.Faltas","Média faltas árbitro",
                      "Amarelos/jogo","Desv.Amarelos","Média amar. árbitro"]:
                disp_p[c] = disp_p[c].apply(lambda x: round(float(x),2))

            def sty_profile(row):
                df_val = float(row["Desv.Faltas"])
                dy_val = float(row["Desv.Amarelos"])
                score  = abs(df_val)/2 + abs(dy_val)
                if df_val > 2 or dy_val > 0.5:
                    return ["background-color: rgba(214,96,77,0.18)"] * len(row)
                if df_val < -2 or dy_val < -0.5:
                    return ["background-color: rgba(67,147,195,0.15)"] * len(row)
                return [""] * len(row)

            st.dataframe(disp_p.style.apply(sty_profile, axis=1),
                         use_container_width=True, height=460, hide_index=True)

        with col_tr:
            metrica_p = st.radio("Métrica", ["🟡 Desvio amarelos", "🦵 Desvio faltas"],
                                 horizontal=True, key="prof_m")
            col_p = "desvio_amarelos" if "amarelos" in metrica_p else "desvio_faltas"
            thr   = 0.5 if "amarelos" in metrica_p else 2.0
            plot_p = profile_df.sort_values(col_p)

            colors_p = ["#d6604d" if d > thr else "#f5a623" if d > thr/2
                        else "#4393c3" if d < -thr/2 else "#aab4c0"
                        for d in plot_p[col_p]]

            fig_p = go.Figure(go.Bar(
                x=plot_p[col_p], y=plot_p["arbitro"],
                orientation="h", marker_color=colors_p,
                text=plot_p[col_p].apply(lambda x: f"{x:+.2f}"),
                textposition="outside",
                textfont=dict(size=9, color="#1a1d23"),
            ))
            fig_p.add_vline(x=0, line_color="#888", line_width=1)
            fig_p.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23", size=10),
                height=460, margin=dict(l=10, r=50, t=10, b=10),
                xaxis=dict(title="Desvio vs média do árbitro", gridcolor="#d0d7e0", zerolinecolor="#888"),
                yaxis=dict(gridcolor="#d0d7e0"),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Tab 2: Evolução por época ─────────────────────────────────────────────
    with tab_evo_t:
        st.markdown(f"##### Evolução de faltas e cartões do **{sel_team}** por época")

        if not season_df_t.empty:
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(
                x=season_df_t["epoca"], y=season_df_t["media_faltas"].astype(float),
                name="Faltas/jogo", marker_color="#4393c3", yaxis="y"
            ))
            fig_ev.add_trace(go.Scatter(
                x=season_df_t["epoca"], y=season_df_t["media_amarelos"].astype(float),
                name="Amarelos/jogo", line=dict(color="#f5a623", width=2),
                mode="lines+markers", yaxis="y2"
            ))
            fig_ev.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1a1d23"),
                height=340, margin=dict(l=10, r=60, t=20, b=10),
                xaxis=dict(gridcolor="#d0d7e0"),
                yaxis=dict(title="Faltas/jogo", gridcolor="#d0d7e0"),
                yaxis2=dict(title="Amarelos/jogo", overlaying="y",
                            side="right", gridcolor="#d0d7e0"),
                legend=dict(bgcolor="#f0f2f5"),
                barmode="group",
            )
            st.plotly_chart(fig_ev, use_container_width=True)

            disp_s = season_df_t[["epoca","jogos","media_faltas","faltas_total",
                                   "media_amarelos","amarelos_total"]].copy()
            disp_s.columns = ["Época","Jogos","Faltas/jogo","Total faltas",
                               "Amarelos/jogo","Total amarelos"]
            st.dataframe(disp_s, use_container_width=True, hide_index=True)

    # ── Tab 3: Veredito ────────────────────────────────────────────────────────
    with tab_veredito:
        st.markdown(f"##### Síntese — **{sel_team}** é beneficiada ou prejudicada?")

        if profile_df.empty or rank_row.empty:
            st.info("Dados insuficientes para esta equipa.")
        else:
            # Contar árbitros com desvio extremo positivo e negativo
            n_arb_mais_f  = int((profile_df["desvio_faltas"]  > 2.0).sum())
            n_arb_menos_f = int((profile_df["desvio_faltas"]  < -2.0).sum())
            n_arb_mais_y  = int((profile_df["desvio_amarelos"] > 0.5).sum())
            n_arb_menos_y = int((profile_df["desvio_amarelos"] < -0.5).sum())
            n_arb_total   = len(profile_df)

            st.markdown("**Síntese estatística:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Árbitros com mais faltas (+2)", n_arb_mais_f,
                      delta=f"de {n_arb_total}", delta_color="inverse" if n_arb_mais_f > n_arb_menos_f else "off")
            c2.metric("Árbitros com menos faltas (-2)", n_arb_menos_f,
                      delta=f"de {n_arb_total}", delta_color="normal" if n_arb_menos_f > n_arb_mais_f else "off")
            c3.metric("Árbitros com mais amarelos (+0.5)", n_arb_mais_y,
                      delta=f"de {n_arb_total}", delta_color="inverse" if n_arb_mais_y > n_arb_menos_y else "off")
            c4.metric("Árbitros com menos amarelos (-0.5)", n_arb_menos_y,
                      delta=f"de {n_arb_total}", delta_color="normal" if n_arb_menos_y > n_arb_mais_y else "off")

            # Veredito automático
            score_prej = n_arb_mais_f + n_arb_mais_y * 2
            score_benef = n_arb_menos_f + n_arb_menos_y * 2

            st.markdown("---")
            if score_prej > score_benef * 1.5:
                st.error(f"⚠️ **Indício de prejuízo** — mais árbitros desviam positivamente (mais faltas/cartões) do que negativamente. Score: prejudicada={score_prej} vs beneficiada={score_benef}.")
            elif score_benef > score_prej * 1.5:
                st.success(f"✓ **Indício de benefício** — mais árbitros desviam negativamente (menos faltas/cartões) do que positivamente. Score: beneficiada={score_benef} vs prejudicada={score_prej}.")
            else:
                st.info(f"⚖️ **Sem padrão claro** — os desvios positivos e negativos são semelhantes. Score: prejudicada={score_prej} vs beneficiada={score_benef}.")

            st.markdown("---")
            st.caption("""
            ⚠️ **Nota metodológica:** este veredito é indicativo, não conclusivo.
            Um score elevado de "prejudicada" significa que mais árbitros lhe dão
            mais faltas/cartões do que a sua própria média — mas não prova intenção.
            Para evidência mais robusta, verificar se os mesmos árbitros aparecem
            em múltiplas épocas com o mesmo padrão (vista Árbitro específico → Z-scores).
            """)


# ── MODO 6: ALERTAS & COMPARAÇÃO ─────────────────────────────────────────────

elif modo == "⚡ Alertas & Comparação":

    st.markdown("# ⚡ Alertas & Comparação")
    st.markdown("---")

    tab_alerts, tab_hist, tab_comp, tab_hb = st.tabs([
        "🚨 Alertas da época", "📈 Histórico por jornada",
        "⚖️ Comparar árbitros", "🏠 Home Bias"
    ])

    # ── Tab 1: Alertas ───────────────────────────────────────────────────────
    with tab_alerts:
        from cal.analysis.alerts import (
            load_current_bias, detect_threshold_alerts,
            detect_top3, detect_multi_season, load_multi_season_bias,
            THRESHOLDS
        )

        all_seasons = ["2024/25","2023/24","2022/23","2021/22",
                       "2020/21","2019/20","2018/19","2017/18"]
        sel_season_a = st.selectbox("Época", all_seasons, index=0, key="alert_season")

        bias_df_a = load_bias_scores_all()
        bias_df_a = bias_df_a[bias_df_a["season"] == sel_season_a].copy()
        for col in ["yellow_diff_bias_z","red_diff_bias_z",
                    "fouls_diff_bias_z","suspicion_score","n_games"]:
            bias_df_a[col] = pd.to_numeric(bias_df_a[col], errors="coerce").fillna(0)

        if bias_df_a.empty:
            st.info(f"Sem dados de bias scores para {sel_season_a}.")
        else:
            # Top 3
            top3 = bias_df_a.nlargest(3, "suspicion_score")
            st.markdown("#### 🏆 Top 3 — maior SuspicionScore na época")
            c1, c2, c3 = st.columns(3)
            for col_w, (_, row) in zip([c1, c2, c3], top3.iterrows()):
                col_w.metric(
                    row["referee"],
                    f"Score {float(row['suspicion_score']):.3f}",
                    delta=f"{int(row['n_games'])} jogos"
                )

            st.markdown("---")
            st.markdown("#### Árbitros por nível de alerta")

            level_cfg = {
                "anomalia":    ("🚨 ANOMALIA",    3.0, 6.0, "#d6604d"),
                "suspeito":    ("⚠️ SUSPEITO",    2.5, 4.0, "#f5a623"),
                "monitorizar": ("⚡ MONITORIZAR", 1.0, 2.0, "#4393c3"),
            }

            for level_key, (label, z_thr, s_thr, color) in level_cfg.items():
                mask = (
                    (bias_df_a[["yellow_diff_bias_z","red_diff_bias_z","fouls_diff_bias_z"]].abs().max(axis=1) >= z_thr) |
                    (bias_df_a["suspicion_score"] >= s_thr)
                )
                grp = bias_df_a[mask].copy()
                if grp.empty:
                    continue

                st.markdown(f"**{label}** — |Z| ≥ {z_thr} ou SuspicionScore ≥ {s_thr}")
                for col in ["sig_yellow_fdr","sig_fouls_fdr","p_adj_yellow","p_adj_fouls"]:
                    if col not in grp.columns:
                        grp[col] = None

                disp_a = grp[["referee","n_games","suspicion_score",
                               "yellow_diff_bias_z","red_diff_bias_z","fouls_diff_bias_z",
                               "sig_yellow_fdr","sig_fouls_fdr"]].copy()

                # Badge FDR por linha
                def make_fdr_badge(row):
                    parts = []
                    if row.get("sig_fouls_fdr")  is True: parts.append("faltas ✓")
                    if row.get("sig_yellow_fdr") is True: parts.append("amarelos ✓")
                    return ", ".join(parts) if parts else "⚠ sem FDR"
                disp_a["FDR sig."] = disp_a.apply(make_fdr_badge, axis=1)

                disp_a = disp_a.drop(columns=["sig_yellow_fdr","sig_fouls_fdr"])
                disp_a.columns = ["Árbitro","Jogos","SuspicionScore",
                                   "Z Amarelos","Z Vermelhos","Z Faltas","FDR sig."]
                for c in ["SuspicionScore","Z Amarelos","Z Vermelhos","Z Faltas"]:
                    disp_a[c] = disp_a[c].round(3)
                disp_a = disp_a.sort_values("SuspicionScore", ascending=False)

                def sty_alert(row):
                    try:
                        max_z = max(abs(float(row["Z Amarelos"])),
                                    abs(float(row["Z Vermelhos"])),
                                    abs(float(row["Z Faltas"])))
                        if max_z >= 3.0: return [f"background-color: rgba(214,96,77,0.2)"] * len(row)
                        if max_z >= 2.5: return [f"background-color: rgba(245,166,35,0.15)"] * len(row)
                        return ["background-color: rgba(67,147,195,0.1)"] * len(row)
                    except: return [""] * len(row)

                def sty_z(val):
                    try:
                        v = abs(float(val))
                        if v >= 3.0: return "color: #d6604d; font-weight: 600"
                        if v >= 2.5: return "color: #e07000; font-weight: 500"
                        if v >= 1.5: return "color: #1a1d23"
                        return "color: #888"
                    except: return ""

                def sty_fdr(val):
                    if val and "✓" in str(val):
                        return "color: #2e7d32; font-weight: 600"
                    if val == "⚠ sem FDR":
                        return "color: #999999; font-style: italic"
                    return ""

                st.dataframe(
                    disp_a.style
                          .apply(sty_alert, axis=1)
                          .map(sty_z, subset=["Z Amarelos","Z Vermelhos","Z Faltas"])
                          .map(sty_fdr, subset=["FDR sig."]),
                    use_container_width=True, hide_index=True,
                    height=min(300, len(disp_a) * 38 + 50)
                )
                st.caption(
                    "**FDR sig.**: ✓ = evidência robusta após correcção Benjamini-Hochberg (FDR 5%). "
                    "Linhas com '⚠ sem FDR' devem ser interpretadas com cautela — o Z-score pode ser ruído estatístico."
                )

            # Padrão multi-época
            multi_df_a = load_bias_scores_all()
            for col in ["yellow_diff_bias_z","red_diff_bias_z","fouls_diff_bias_z","suspicion_score"]:
                multi_df_a[col] = pd.to_numeric(multi_df_a[col], errors="coerce").fillna(0)

            st.markdown("---")
            st.markdown("#### 🔄 Padrão multi-época — mesmo sentido em 2+ épocas")
            multi_alerts = []
            for ref_id, grp in multi_df_a[multi_df_a["season"].isin(
                    ["2022/23","2023/24","2024/25"])].groupby("referee_id"):
                grp = grp.sort_values("season")
                z_y = grp["yellow_diff_bias_z"].values
                biased = np.abs(z_y) > 1.5
                if biased.sum() >= 2 and len(set(np.sign(z_y[biased]))) == 1:
                    multi_alerts.append({
                        "Árbitro": grp["referee"].iloc[0],
                        "Épocas":  ", ".join(grp["season"].tolist()),
                        "Z por época": " | ".join(f"{s}: {z:+.2f}" for s, z in zip(grp["season"], z_y)),
                        "Sentido": "↑ mais amarelos" if z_y[biased][0] > 0 else "↓ menos amarelos",
                    })
            if multi_alerts:
                st.dataframe(pd.DataFrame(multi_alerts), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum padrão multi-época detectado nas últimas 3 épocas.")

    # ── Tab 2: Histórico por jornada ─────────────────────────────────────────
    with tab_hist:
        st.markdown("#### Evolução do SuspicionScore ao longo da época — por jornada")

        referees_df2 = load_referees()
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            sel_refs_h = st.multiselect(
                "Árbitros", referees_df2["arbitro"].tolist(),
                default=referees_df2["arbitro"].tolist()[:3],
                key="hist_refs"
            )
        with col_r2:
            all_seasons_h = ["2024/25","2023/24","2022/23","2021/22","2020/21","2019/20","2018/19"]
            sel_season_h = st.selectbox("Época", all_seasons_h, index=0, key="hist_season")

        metrica_h = st.radio("Métrica", ["SuspicionScore","Z Amarelos","Z Faltas"],
                             horizontal=True, key="hist_metrica")
        col_map = {"SuspicionScore": "suspicion_score",
                   "Z Amarelos":     "yellow_diff_bias_z",
                   "Z Faltas":       "fouls_diff_bias_z"}
        col_h = col_map[metrica_h]

        if sel_refs_h:
            hist_frames = []
            for ref_name in sel_refs_h:
                ref_id_h = int(referees_df2[referees_df2["arbitro"] == ref_name]["referee_id"].iloc[0])
                h = load_bias_history(ref_id_h, season=sel_season_h)
                if not h.empty:
                    h["arbitro"] = ref_name
                    hist_frames.append(h)

            if hist_frames:
                hist_all = pd.concat(hist_frames)
                hist_all[col_h] = pd.to_numeric(hist_all[col_h], errors="coerce")

                fig_h = px.line(
                    hist_all, x="matchday", y=col_h, color="arbitro",
                    markers=True,
                    labels={col_h: metrica_h, "matchday": "Jornada"},
                )
                # Linhas de threshold
                for thr, color, dash in [(1.0,"#4393c3","dot"),
                                          (2.5,"#f5a623","dash"),
                                          (3.0,"#d6604d","solid")]:
                    if col_h == "suspicion_score":
                        thr_val = thr * 2
                    else:
                        thr_val = thr
                    fig_h.add_hline(y=thr_val, line_color=color, line_dash=dash,
                                    line_width=1, opacity=0.6)
                    if col_h != "suspicion_score":
                        fig_h.add_hline(y=-thr_val, line_color=color, line_dash=dash,
                                        line_width=1, opacity=0.6)

                fig_h.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                    font=dict(color="#1a1d23"),
                    legend=dict(bgcolor="#f0f2f5", bordercolor="#d0d7e0"),
                    xaxis=dict(gridcolor="#d0d7e0", title="Jornada"),
                    yaxis=dict(gridcolor="#d0d7e0"),
                    height=400, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption("Linhas: azul=monitorizar, laranja=suspeito, vermelho=anomalia")
            else:
                st.info("Sem histórico disponível. Correr run_bias_history.py primeiro.")

    # ── Tab 3: Comparar árbitros ──────────────────────────────────────────────
    with tab_comp:
        st.markdown("#### Comparação directa entre árbitros — Z-scores por época")

        referees_df3 = load_referees()
        sel_refs_c = st.multiselect(
            "Seleccionar 2 a 4 árbitros para comparar",
            referees_df3["arbitro"].tolist(),
            default=referees_df3["arbitro"].tolist()[:3],
            key="comp_refs",
            max_selections=4,
        )

        metrica_c = st.radio(
            "Métrica", ["SuspicionScore","Z Amarelos","Z Vermelhos","Z Faltas"],
            horizontal=True, key="comp_metrica"
        )
        col_c_map = {
            "SuspicionScore": "suspicion_score",
            "Z Amarelos":     "yellow_diff_bias_z",
            "Z Vermelhos":    "red_diff_bias_z",
            "Z Faltas":       "fouls_diff_bias_z",
        }
        col_c = col_c_map[metrica_c]

        if sel_refs_c:
            all_scores = load_bias_scores_all()
            ref_ids_c = referees_df3[referees_df3["arbitro"].isin(sel_refs_c)]["referee_id"].tolist()
            comp_df = all_scores[all_scores["referee_id"].isin(ref_ids_c)].copy()
            comp_df[col_c] = pd.to_numeric(comp_df[col_c], errors="coerce")

            col_left_c, col_right_c = st.columns([3, 2])

            with col_left_c:
                fig_c = px.line(
                    comp_df, x="season", y=col_c, color="referee",
                    markers=True,
                    labels={col_c: metrica_c, "season": "Época", "referee": "Árbitro"},
                )
                fig_c.add_hline(y=0, line_color="#888", line_width=1)
                if col_c != "suspicion_score":
                    for thr, color, dash in [(2.5,"#f5a623","dash"),(3.0,"#d6604d","solid")]:
                        fig_c.add_hline(y=thr,  line_color=color, line_dash=dash, line_width=1, opacity=0.5)
                        fig_c.add_hline(y=-thr, line_color=color, line_dash=dash, line_width=1, opacity=0.5)
                fig_c.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                    font=dict(color="#1a1d23"),
                    legend=dict(bgcolor="#f0f2f5", bordercolor="#d0d7e0"),
                    xaxis=dict(gridcolor="#d0d7e0", tickangle=30),
                    yaxis=dict(gridcolor="#d0d7e0"),
                    height=380, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_c, use_container_width=True)

            with col_right_c:
                # Tabela pivot: épocas em linha, árbitros em coluna
                pivot_c = comp_df.pivot_table(
                    index="season", columns="referee",
                    values=col_c, aggfunc="mean"
                ).round(3)
                st.dataframe(pivot_c, use_container_width=True, height=380)

    # ── Tab Home Bias ────────────────────────────────────────────────────────
    with tab_hb:
        st.markdown("#### 🏠 Home Bias — decisões por contexto de marcador")
        st.markdown("""
        Compara as faltas e amarelos dados pelo árbitro à equipa da casa vs. visitante,
        segmentados pelo marcador ao intervalo. O **PressureBiasIndex** mede a diferença
        entre o tratamento da casa quando está a perder vs. quando está a ganhar —
        valor positivo indica que o árbitro é mais favorável à equipa da casa sob pressão.
        """)

        try:
            hb_all = load_home_bias_summary()
        except Exception:
            hb_all = pd.DataFrame()

        if hb_all.empty:
            st.info("Sem dados. Correr: docker compose exec dev python run_home_bias.py")
        else:
            for col in ["media_fouls_home","media_fouls_away","media_fouls_diff",
                        "home_bias_fouls_z","home_bias_yellow_z","pressure_bias_index",
                        "n_games"]:
                if col in hb_all.columns:
                    hb_all[col] = pd.to_numeric(hb_all[col], errors="coerce").fillna(0)

            # Filtros
            col_hb1, col_hb2, col_hb3 = st.columns(3)
            with col_hb1:
                seasons_hb = ["Todas"] + sorted(hb_all["season"].unique().tolist())
                sel_s_hb = st.selectbox("Época", seasons_hb, key="hb_season")
            with col_hb2:
                ctx_opts = ["Todos", "Casa a perder", "Empate", "Casa a ganhar"]
                sel_ctx = st.selectbox("Contexto", ctx_opts, key="hb_ctx")
            with col_hb3:
                refs_hb = ["Todos"] + sorted(hb_all["referee"].unique().tolist())
                sel_ref_hb = st.selectbox("Árbitro", refs_hb, key="hb_ref")

            ctx_map_rev = {
                "Casa a perder": "HOME_LOSING",
                "Empate":        "DRAW",
                "Casa a ganhar": "HOME_WINNING"
            }

            filt_hb = hb_all.copy()
            if sel_s_hb   != "Todas":  filt_hb = filt_hb[filt_hb["season"]        == sel_s_hb]
            if sel_ctx     != "Todos":  filt_hb = filt_hb[filt_hb["score_context"] == ctx_map_rev.get(sel_ctx, sel_ctx)]
            if sel_ref_hb  != "Todos":  filt_hb = filt_hb[filt_hb["referee"]       == sel_ref_hb]

            context_labels = {"HOME_WINNING": "Casa a ganhar",
                              "DRAW": "Empate", "HOME_LOSING": "Casa a perder"}
            filt_hb["Contexto"] = filt_hb["score_context"].map(context_labels)

            # Ranking por PressureBiasIndex
            st.markdown("##### Ranking por PressureBiasIndex (árbitros com maior home bias sob pressão)")
            pbi_summary = (
                filt_hb.groupby(["referee","season"])["pressure_bias_index"]
                .first().reset_index()
                .sort_values("pressure_bias_index", ascending=False)
                .rename(columns={"referee":"Árbitro","season":"Época",
                                  "pressure_bias_index":"PressureBiasIndex"})
            )
            pbi_summary["PressureBiasIndex"] = pbi_summary["PressureBiasIndex"].round(3)

            def sty_pbi(row):
                v = abs(float(row["PressureBiasIndex"]))
                if v >= 2: return ["background-color: rgba(214,96,77,0.18)"] * len(row)
                if v >= 1: return ["background-color: rgba(245,166,35,0.12)"] * len(row)
                return [""] * len(row)

            st.dataframe(pbi_summary.style.apply(sty_pbi, axis=1),
                         use_container_width=True, height=320, hide_index=True)

            # Tabela detalhada
            st.markdown("##### Detalhe por árbitro × contexto")
            disp_hb2 = filt_hb[[
                "referee","season","Contexto","n_games",
                "media_fouls_home","media_fouls_away","media_fouls_diff","home_bias_fouls_z",
                "media_yc_home","media_yc_away","home_bias_yellow_z"
            ]].copy()
            disp_hb2.columns = [
                "Árbitro","Época","Contexto","Jogos",
                "F.Casa","F.Visit.","Dif.Faltas","Z Faltas",
                "A.Casa","A.Visit.","Z Amarelos"
            ]
            for c in ["F.Casa","F.Visit.","Dif.Faltas","Z Faltas","A.Casa","A.Visit.","Z Amarelos"]:
                disp_hb2[c] = disp_hb2[c].apply(lambda x: round(float(x), 2))
            disp_hb2 = disp_hb2.sort_values("Z Faltas", ascending=False)

            st.dataframe(disp_hb2, use_container_width=True, height=400, hide_index=True)
            st.caption("Z Faltas > 0 = árbitro marca mais faltas à casa do que ao visitante neste contexto. "
                       "PressureBiasIndex positivo = árbitro protege mais a casa quando esta perde.")


# ── MODO 7: AJUDA / FAQ ───────────────────────────────────────────────────────

elif modo == "❓ Ajuda":  # MODO 7

    st.markdown("# ❓ Guia de utilização — CAL")
    st.markdown("---")

    with st.expander("📌 O que é o CAL e o que mede?", expanded=True):
        st.markdown("""
**CAL — Criticar a Arbitragem Legalmente** é um sistema de análise estatística
que quantifica desvios entre as decisões de arbitragem *esperadas* e as *observadas*
na Primeira Liga Portuguesa (2017/18 → presente).

O sistema não prova intenção — identifica **padrões estatisticamente anómalos**
que merecem atenção. Um Z-score elevado significa que o padrão tem baixa
probabilidade de ocorrer por acaso.
        """)

    with st.expander("🗃️ Origem dos dados e frequência de actualização"):
        st.markdown("""
**Fontes de dados:**

| Fonte | O que fornece | Disponibilidade |
|-------|--------------|-----------------|
| [football-data.co.uk](https://www.football-data.co.uk) | Golos, cartões, faltas, remates, cantos por jogo | Gratuito |
| [Sofascore API](https://www.sofascore.com) | Árbitro de cada jogo | Gratuito (API não-oficial) |

**Nota importante:** o football-data.co.uk não inclui o nome do árbitro para a
Primeira Liga — essa informação é obtida exclusivamente via Sofascore.

**Cobertura actual:** 2017/18 → 2025/26 (épocas completas: 2017/18 a 2023/24).
98.4% dos jogos têm árbitro identificado nas 7 épocas completas.

---

**Frequência de actualização recomendada:**

Durante a época, o botão **▶ Actualizar agora** na sidebar deve ser usado:
- **Domingo à noite** — após os jogos do fim de semana
- **Segunda-feira** — após o último jogo da jornada (jogos em atraso)

O botão corre automaticamente 5 passos sequenciais:
1. Ingestão de resultados (football-data.co.uk)
2. Ingestão de árbitros (Sofascore)
3. Recálculo de features
4. Recálculo de bias scores e análise de cartões
5. Dashboard recarrega com os dados actualizados

O processo demora ~2-3 minutos. O botão é **idempotente** — pode ser carregado
várias vezes sem duplicar dados.

**Fim de época (após o último jogo):**
Para além do botão, re-treinar os modelos estatísticos via terminal:
```
docker compose exec dev python run_models.py
```
Isto melhora a precisão dos bias scores para as épocas seguintes.
        """)

    with st.expander("🔍 Vista — Árbitro específico"):
        st.markdown("""
**Como usar:**
1. Seleccionar o árbitro na sidebar
2. Ver as 6 métricas de cabeçalho — desvios máximos de amarelos e faltas
3. **Tab Amarelos** — tabela de equipas ordenada por desvio de cartões amarelos
4. **Tab Faltas** — tabela de equipas ordenada por desvio de faltas assinaladas
5. Selector **🟡 Amarelos / 🦵 Faltas** no gráfico de barras alterna a métrica

**Como interpretar o Desvio:**
- Desvio positivo (vermelho) → árbitro dá mais cartões/faltas a esta equipa do que a sua própria média
- Desvio negativo (azul) → dá menos do que a sua média
- O desvio é calculado face à média do próprio árbitro, não à média geral da liga

**Z-scores por época:**
- |Z| < 1.5 → normal
- 1.5 ≤ |Z| < 2.5 → ligeiramente atípico
- |Z| ≥ 2.5 → suspeito (p < 0.01)
- |Z| ≥ 3 → anomalia forte (p < 0.003)
        """)

    with st.expander("📊 Vista — Desvios extremos"):
        st.markdown("""
**Tab Amarelos** — pares árbitro×equipa com desvio > ±0.7 amarelos/jogo face à média do árbitro.
Threshold de 0.7 equivale a ~25-30% acima/abaixo da média típica.

**Tab Faltas** — pares árbitro×equipa com desvio > ±2 faltas/jogo face à média do árbitro.
Threshold de 2 faltas é equivalente em termos relativos ao de 0.7 amarelos.

**Filtros disponíveis:** sentido (mais/menos), árbitro específico, equipa específica.

**Para encontrar os casos mais graves:** ordenar pela coluna Desvio e filtrar
por árbitros com mais de 5 jogos na combinação — base estatística mais robusta.
        """)

    with st.expander("🏟️ Vista — Por equipa (ranking geral e lugares europeus)"):
        st.markdown("""
Esta vista responde directamente às perguntas:
- **Qual a equipa que sofre mais faltas?** → Tab Faltas
- **Qual a equipa que recebe mais cartões?** → Tab Amarelos
- **Como evoluiu isso ao longo das épocas?** → Tab Evolução
- **As equipas que terminaram no top 5 sofrem mais ou menos faltas/cartões?** → Tab 🏆 Top 5 — Lugares Europeus

**Tab Top 5 — Lugares Europeus:**
Mostra as 5 equipas que terminaram cada época nos lugares de acesso às competições
europeias (UEFA Champions League, Europa League, Conference League), com as
respectivas estatísticas de faltas e cartões nessa época.

O selector de época permite ver uma época específica ou todas em simultâneo.
Com "Todas as épocas" aparece também um gráfico de evolução.

Vermelho = equipa significativamente acima da média da liga; azul = abaixo.

⚠️ Nota: faltas e cartões elevados não significam necessariamente prejuízo —
equipas mais agressivas ou que dominam o jogo podem acumular mais faltas
ao tentar recuperar a bola.
        """)

    with st.expander("🗺️ Vista — Heatmap geral"):
        st.markdown("""
Matriz árbitro × equipa com o desvio como valor.

**Selector 🟡 Amarelos / 🦵 Faltas** alterna entre as duas métricas.

**Como ler:** cada célula mostra quanto acima ou abaixo da sua própria média
o árbitro decide relativamente a essa equipa. Cores relativas:
- Vermelho escuro → desvio muito positivo (árbitro dá muito mais a esta equipa)
- Branco → sem desvio
- Azul escuro → desvio muito negativo

**Células vazias:** menos de 3 jogos em conjunto — amostra insuficiente.
        """)

    with st.expander("🔄 Actualizar dados — botão na sidebar"):
        st.markdown("""
O botão **▶ Actualizar agora** corre a pipeline completa para a época seleccionada:

1. Ingestão de jogos (football-data.co.uk)
2. Ingestão de árbitros (Sofascore API)
3. Feature engineering
4. Bias engine (Z-scores + análise de cartões)
5. Dashboard recarrega automaticamente

**Frequência recomendada:**
- Domingo à noite — após os jogos do fim de semana
- Segunda-feira — após o último jogo da jornada

O botão é **idempotente** — pode ser carregado quantas vezes quiser sem duplicar dados.

**Fim de época:** após o último jogo, re-treinar os modelos via CLI:
```
docker compose exec dev python run_models.py
```
        """)

    with st.expander("🔬 Como identificar a equipa mais beneficiada ou mais prejudicada?"):
        st.markdown("""
A resposta não é simples porque "beneficiada" e "prejudicada" têm dimensões diferentes.

**Equipa mais beneficiada** — procurar convergência em três análises:

1. **Vista Desvios extremos → Tab Faltas e Tab Amarelos**
Filtrar pela equipa e verificar se aparece consistentemente no lado *negativo* (azul) —
ou seja, árbitros que lhe assinalam sistematicamente *menos* faltas e *menos* cartões
do que a sua própria média com outras equipas.

2. **Vista Por equipa → Tab Faltas e Tab Amarelos**
Verificar se a equipa está no fundo do ranking — sofre poucas faltas e recebe
poucos cartões face à média da liga. Combinado com o ponto anterior, sugere que
não só joga "limpa" em termos absolutos, como os árbitros tendem a ser mais permissivos.

3. **Vista 🔬 Perfil de equipa → Tab Veredito**
Síntese automática: conta quantos árbitros desviam positivamente (mais) e negativamente
(menos) para esta equipa. Score de benefício > score de prejuízo sugere equipa beneficiada.

---

**Equipa mais prejudicada** — simetria inversa, com uma nuance importante:

1. **Vista Desvios extremos → Tab Faltas e Tab Amarelos**
Filtrar pela equipa e verificar se aparece consistentemente no lado *positivo* (vermelho) —
árbitros que lhe assinalam sistematicamente *mais* faltas e *mais* cartões do que a sua média.

2. **Vista Por equipa → Tab Faltas e Tab Amarelos**
Verificar se a equipa está no topo do ranking em faltas e cartões.
⚠️ **Nuance:** uma equipa pode ter muitas faltas porque joga de forma mais agressiva,
não porque é prejudicada. Este ponto sozinho não é suficiente.

3. **Vista Top 5 — Lugares Europeus**
Este é o ponto mais revelador. Se uma equipa que termina consistentemente nos
primeiros lugares aparece também com faltas e cartões acima da média, há duas hipóteses:
ou joga agressivamente para ganhar, ou é de facto prejudicada. Para distinguir,
cruzar com os desvios extremos — se os árbitros lhe dão *mais* do que a sua própria
média para outras equipas, o argumento da agressividade fica enfraquecido.

---

**O que o sistema não consegue distinguir sozinho:**

A limitação fundamental é que não controlamos o comportamento da equipa — uma equipa
que pressiona alto, faz muitos duelos e é fisicamente intensa vai naturalmente acumular
mais faltas. O sistema mede o desvio face à média do árbitro, o que atenua este problema,
mas não o elimina completamente.

A **evidência mais robusta** seria: equipa X aparece nos desvios extremos positivos
(mais faltas/cartões) para muitos árbitros diferentes, ao longo de várias épocas,
mesmo quando está em desvantagem no marcador — contexto em que normalmente se
esperaria mais faltas ofensivas.
        """)


    with st.expander("🤖 Declaração de Utilização de Inteligência Artificial"):
        st.markdown("""
Este sistema foi desenvolvido com assistência de ferramentas de inteligência artificial
(**Claude**, Anthropic, 2025–2026), em conformidade com o EU AI Act (Regulamento UE 2024/1689).

**Âmbito de utilização de IA:**
- Geração de código Python, SQL, JavaScript e HTML
- Documentação técnica e estruturação do paper académico
- Revisão de análises estatísticas

**Responsabilidade humana:**
A concepção metodológica, validação dos modelos estatísticos, interpretação dos resultados
e todas as afirmações analíticas são da exclusiva responsabilidade do autor (Pedro M. Pestana).
Todo o código gerado com assistência de IA foi revisto e testado pelo autor.

**Dados:** nenhum dado foi gerado ou modificado por IA.
Todas as fontes são públicas e verificáveis (football-data.co.uk, Sofascore).

**Nota:** este sistema não constitui acusação — identifica padrões estatísticos
que merecem escrutínio público.
        """)

    with st.expander("📐 Metodologia — como são calculados os Z-scores?"):
        st.markdown("""
**xYC / xRC (cartões):** regressão logística que estima P(cartão | contexto do jogo).
Features: is_home, matchday, força da equipa (rolling 10j), tendência do árbitro (rolling 20j).

**xF (faltas):** modelo Poisson que estima E(faltas | contexto do jogo).

**Validação:** leave-one-season-out (LOSO) — os modelos são treinados em N-1 épocas
e validados na época excluída, garantindo avaliação genuinamente out-of-sample.
*(Ref: Bergmeir & Benítez, 2012)*

---

**Z-score de viés (cartões) — fórmula corrigida (v0.8.0):**
`Z = (jogos_com_cartão_obs - Σ Pᵢ) / √(Σ Pᵢ × (1 - Pᵢ))`

A variância correcta para Bernoullis **não identicamente distribuídas** é **Σ Pᵢ(1−Pᵢ)**.
A fórmula anterior `exp*(1-exp/n)` era matematicamente incorrecta — subestimava a variância.
*(Ref: Agresti, 2002, Categorical Data Analysis)*

**Z-score de viés (faltas):**
`Z = (média_faltas_obs - média_faltas_esperada) / (std_obs / √n)`
Teste t-student; para n < 30 a aproximação Normal é imprecisa — interpretar com precaução.

**SuspicionScore:** `|Z_amarelos| + |Z_vermelhos| + |Z_faltas|`
Medida ordinal de intensidade agregada — não tem distribuição estatística conhecida.
Não deve ser usado como teste de hipótese.

---

**Correcção de múltiplos testes — FDR global (v0.9.1):**
Com 510 testes em simultâneo (170 árbitros×épocas × 3 métricas), a probabilidade de
falsos positivos sem correcção é substancial. O sistema aplica **Benjamini-Hochberg**
sobre o conjunto **total** de 510 p-values numa **única família global**.

⚠️ Versão anterior (v0.9.0) aplicava FDR separadamente por métrica — 3 famílias de
170 testes cada. Isso era incorrecto: poderia resultar em até 14.3% de falsos positivos
no total, em vez dos 5% pretendidos.
*(Ref: Benjamini & Hochberg, 1995, JRSS-B)*

**Thresholds de Z-score** (|Z| ≥ 1.0 / 2.5 / 3.0) são **orientativos**.
A evidência estatística robusta é definida **exclusivamente pela coluna FDR sig.**
(p_adj < 0.05). Um Z-score elevado sem confirmação FDR deve ser lido como sinal
para análise mais detalhada, não como evidência de viés.

---

**Overdispersion (modelo Poisson) — teste formal Cameron & Trivedi (1990):**
Regressão auxiliar: (Y−μ)² − Y = α·μ² (OLS sem constante). H₀: α = 0.

Resultado empírico: **alpha = 0.0139, t = 8.183, p ≈ 0**

Overdispersion estatisticamente significativa mas **magnitude negligenciável**:
a variância observada é apenas 1.39% acima da esperada pelo Poisson.
Com n = 4816, efeitos minúsculos atingem elevada significância estatística —
distinção crítica entre significância estatística e significância prática.
O modelo Poisson é adequado; a overdispersion é uma limitação menor documentada.

---

**Resultado empírico principal (FDR global, 510 testes):**
- 🔴 Faltas: **11 padrões significativos** em **8 árbitros distintos**
- ⚪ Amarelos: **0 padrões significativos** (melhor p_adj = 0.548)
- ⚪ Vermelhos: **0 padrões significativos**

Não existe evidência estatisticamente robusta de desvios em cartões amarelos
no período 2017/18–2025/26 após correcção adequada para múltiplos testes.
        """)
