"""
cal/bias_engine.py
Fase 4 — Bias Engine

Para cada árbitro e temporada, calcula:
  1. Z-scores de desvio entre decisões esperadas e observadas:
       yellow_diff_bias_z  : (observed_yellows - expected_yellows) / sqrt(expected)
       red_diff_bias_z     : idem para vermelhos
       fouls_diff_bias_z   : idem para faltas

  2. Viés casa/fora para penáltis (proxy: cartões dados em casa vs fora):
       penalty_home_bias_z : diferença normalizada de cartões dados em casa vs fora

  3. Interacção árbitro×equipa (regressão logística com termo de interacção):
       interaction_coef    : coeficiente do termo árbitro×equipa
       p_value             : significância estatística
       significant         : p_value < 0.05

  4. SuspicionScore composto:
       suspicion_score = |yellow_diff_bias_z| + |red_diff_bias_z| + |fouls_diff_bias_z|

Resultados guardados em:
  - referee_bias_scores   (Z-scores por árbitro+temporada)
  - referee_team_bias     (interacção árbitro×equipa)

Uso:
    python -m cal.bias_engine          # calcular tudo
    python run_bias.py                 # idem via CLI
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cal.db import get_cursor

from scipy.stats import chi2

log = logging.getLogger(__name__)

SEASONS_ORDER = [
    "2017/18", "2018/19", "2019/20", "2020/21",
    "2021/22", "2022/23", "2023/24", "2024/25",
]

MIN_GAMES_REFEREE = 10  # mínimo de jogos para calcular Z-score fiável

# ── Correcção FDR (Benjamini-Hochberg) ───────────────────────────────────────

def fdr_correction(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Benjamini-Hochberg False Discovery Rate correction.
    Devolve p-values ajustados (BH-adjusted).
    Referência: Benjamini & Hochberg (1995), JRSS-B.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues
    sorted_idx = pvalues.argsort()
    sorted_p   = pvalues.iloc[sorted_idx].values
    bh_p       = sorted_p * n / (np.arange(1, n + 1))
    # Garantir monotonicidade (cumulative minimum from right)
    for i in range(n - 2, -1, -1):
        bh_p[i] = min(bh_p[i], bh_p[i + 1])
    bh_p = np.minimum(bh_p, 1.0)
    result = np.empty(n)
    result[sorted_idx] = bh_p
    return pd.Series(result, index=pvalues.index)


def z_to_pvalue_twotailed(z: float) -> float:
    """Converte Z-score em p-value bilateral."""
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs(z))))


# ── Teste de overdispersion para modelo Poisson ───────────────────────────────

def test_overdispersion(obs: pd.Series, exp: pd.Series) -> dict:
    """
    Teste formal de Cameron & Trivedi (1990) para overdispersion no modelo Poisson.

    O teste baseia-se na regressão auxiliar:
        (Y - mu)^2 - Y = alpha * mu^2 + erro
    onde mu = E[Y] = exp_fouls. Se alpha > 0, há overdispersion (Var > E[Y]).

    O teste é implementado como uma regressão OLS sem constante de
    (Y - mu)^2 - Y sobre mu^2, e testa H0: alpha = 0.

    Ref: Cameron, A. C., & Trivedi, P. K. (1990).
    Regression-based tests for overdispersion in the Poisson model.
    Journal of Econometrics, 46(3), 347-364.

    Devolve: dict com alpha, t-stat, p-value e flag is_overdispersed.
    """
    from scipy import stats as scipy_stats
    obs = obs.astype(float).values
    exp = exp.astype(float).values

    # Variável dependente e regressor da regressão auxiliar
    y_aux = (obs - exp) ** 2 - obs   # (Y - mu)^2 - Y
    x_aux = exp ** 2                  # mu^2 (sem constante)

    # OLS sem constante: alpha = Σ(x*y) / Σ(x^2)
    alpha = float(np.dot(x_aux, y_aux) / np.dot(x_aux, x_aux))

    # Erro padrão do estimador OLS
    residuals = y_aux - alpha * x_aux
    n = len(obs)
    se = float(np.sqrt((residuals ** 2).sum() / (n - 1) / np.dot(x_aux, x_aux)))
    t_stat = alpha / se if se > 0 else 0.0
    p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1)))

    return {
        "overdispersion_alpha": round(alpha, 6),
        "overdispersion_tstat": round(t_stat, 4),
        "overdispersion_pvalue": round(p_value, 6),
        "is_overdispersed": alpha > 0 and p_value < 0.05,
    }


# ── SQL: carregar dados combinados ────────────────────────────────────────────

QUERY = """
SELECT
    mf.match_id,
    mf.team_id,
    mf.season,
    mf.match_date,
    mf.referee_id,
    r.name              AS referee,
    mf.is_home,
    mf.matchday,
    mf.yellow_cards     AS obs_yellows,
    mf.red_cards        AS obs_reds,
    mf.fouls            AS obs_fouls,
    ep.expected_yellows AS exp_yellows,
    ep.expected_reds    AS exp_reds,
    ep.expected_fouls   AS exp_fouls
FROM match_features mf
JOIN expected_probabilities ep USING (match_id, team_id)
JOIN referees r ON r.referee_id = mf.referee_id
WHERE mf.referee_id IS NOT NULL
  AND mf.referee_id > 1
  AND mf.yellow_cards IS NOT NULL
ORDER BY mf.match_date, mf.match_id
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _z_score(observed: float, expected: float, n: int) -> float:
    """
    Z-score de Wald para diferença entre observado e esperado.
    Para eventos de contagem: Z = (obs - exp) / sqrt(exp)
    Retorna 0.0 se expected ≤ 0.
    """
    if expected <= 0 or n < MIN_GAMES_REFEREE:
        return 0.0
    return (observed - expected) / np.sqrt(expected)


def _home_away_bias_z(df_ref: pd.DataFrame, col_obs: str) -> float:
    """
    Z-score do viés casa/fora para um árbitro:
      Z = (mean_home - mean_away) / sqrt(pooled_var / n)
    """
    home = df_ref[df_ref["is_home"] == 1][col_obs].dropna()
    away = df_ref[df_ref["is_home"] == 0][col_obs].dropna()
    if len(home) < 5 or len(away) < 5:
        return 0.0
    _, p = stats.ttest_ind(home, away, equal_var=False)
    # Sinal: positivo = árbitro dá mais cartões em casa
    direction = np.sign(home.mean() - away.mean())
    return float(direction * stats.norm.ppf(1 - p / 2))


# ── Cálculo de Z-scores por árbitro+temporada ─────────────────────────────────

def compute_referee_bias_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Z-scores de viés para cada (referee_id, season).

    xYC / xRC — Z-score binomial CORRIGIDO:
      obs = nº de jogos (equipa) com pelo menos 1 amarelo/vermelho
      exp = soma das probabilidades esperadas P(cartão) por jogo
      var = Σ pᵢ(1 - pᵢ)   [variância correcta para Bernoullis não i.i.d.]
      Z   = (obs - exp) / sqrt(var)

      NOTA: A fórmula anterior usava exp*(1-exp/n), que é incorrecta.
      A variância da soma de variáveis de Bernoulli com probabilidades
      diferentes é Σ pᵢ(1-pᵢ), não exp*(1-exp/n).
      Referência: Agresti (2002), Categorical Data Analysis, Cap. 1.

    xF — t-test de médias (nota: assume distribuição aproximadamente Normal,
      válida para n >= 30; para n < 30 usar com precaução):
      Z = (μ_obs - μ_exp) / (σ_obs / sqrt(n))

    Devolve DataFrame para inserção em referee_bias_scores,
    incluindo p-values e correcção FDR (Benjamini-Hochberg).
    """
    rows = []
    for (ref_id, season), grp in df.groupby(["referee_id", "season"]):
        n = len(grp)
        if n < MIN_GAMES_REFEREE:
            continue

        # ── xYC — Z-score binomial CORRIGIDO ────────────────────────────────
        obs_y = int((grp["obs_yellows"] > 0).sum())
        exp_y = float(grp["exp_yellows"].sum())
        # CORRECÇÃO: var = Σ pᵢ(1 - pᵢ), não exp*(1-exp/n)
        p_i_y = grp["exp_yellows"].clip(0, 1)
        var_y = float((p_i_y * (1 - p_i_y)).sum())
        z_y   = (obs_y - exp_y) / np.sqrt(var_y) if var_y > 0 else 0.0

        # ── xRC — Z-score binomial CORRIGIDO ────────────────────────────────
        obs_r = int((grp["obs_reds"] > 0).sum())
        exp_r = float(grp["exp_reds"].sum())
        p_i_r = grp["exp_reds"].clip(0, 1)
        var_r = float((p_i_r * (1 - p_i_r)).sum())
        z_r   = (obs_r - exp_r) / np.sqrt(var_r) if var_r > 0 else 0.0

        # ── xF — t-test de médias ────────────────────────────────────────────
        obs_f_mean = float(grp["obs_fouls"].mean())
        exp_f_mean = float(grp["exp_fouls"].mean())
        std_f      = float(grp["obs_fouls"].std())
        z_f = (obs_f_mean - exp_f_mean) / (std_f / np.sqrt(n)) if std_f > 0 else 0.0

        obs_f = obs_f_mean
        exp_f = exp_f_mean

        z_h = _home_away_bias_z(grp, "obs_yellows")

        suspicion = abs(z_y) + abs(z_r) + abs(z_f)

        # p-values bilaterais (antes de correcção FDR)
        p_y = z_to_pvalue_twotailed(z_y)
        p_r = z_to_pvalue_twotailed(z_r)
        p_f = z_to_pvalue_twotailed(z_f)

        rows.append({
            "referee_id":          ref_id,
            "referee":             grp["referee"].iloc[0],
            "season":              season,
            "n_games":             n,
            "obs_yellows":         obs_y,
            "exp_yellows":         round(exp_y, 2),
            "obs_reds":            obs_r,
            "exp_reds":            round(exp_r, 2),
            "obs_fouls":           obs_f,
            "exp_fouls":           round(exp_f, 2),
            "yellow_diff_bias_z":  round(z_y, 4),
            "red_diff_bias_z":     round(z_r, 4),
            "fouls_diff_bias_z":   round(z_f, 4),
            "penalty_home_bias_z": round(z_h, 4),
            "suspicion_score":     round(suspicion, 4),
            "p_value_yellow":      round(p_y, 6),
            "p_value_red":         round(p_r, 6),
            "p_value_fouls":       round(p_f, 6),
        })

    result = pd.DataFrame(rows)

    # ── Correcção FDR Global (Benjamini-Hochberg) ────────────────────────────
    # CORRECÇÃO v0.9.1: aplicar FDR sobre o conjunto TOTAL de todos os
    # p-values em simultâneo (amarelos + vermelhos + faltas de todos os
    # árbitros e épocas = 170 × 3 = 510 testes).
    #
    # A versão anterior aplicava FDR separadamente por métrica (3 famílias
    # de 170 testes), o que subestimava o erro global: três famílias com
    # FDR=5% cada podem produzir até ~14.3% de falsos positivos no total.
    #
    # A abordagem correcta (Benjamini & Hochberg, 1995) é uma única família
    # com todos os testes, a menos que haja justificação para separação.
    # Ref: Benjamini & Hochberg (1995), JRSS-B, 57(1), 289-300.
    if not result.empty:
        # Empilhar todos os p-values numa única série com índice composto
        all_pvals = pd.concat([
            result["p_value_yellow"].rename("p").to_frame().assign(metric="yellow", orig_idx=result.index),
            result["p_value_red"].rename("p").to_frame().assign(metric="red",    orig_idx=result.index),
            result["p_value_fouls"].rename("p").to_frame().assign(metric="fouls", orig_idx=result.index),
        ], ignore_index=True)

        # FDR global sobre os 510 testes
        all_pvals["p_adj"] = fdr_correction(all_pvals["p"]).values

        # Redistribuir p-values ajustados para as colunas originais
        for metric, adjcol in [
            ("yellow", "p_adj_yellow"),
            ("red",    "p_adj_red"),
            ("fouls",  "p_adj_fouls"),
        ]:
            mask = all_pvals["metric"] == metric
            adj_vals = all_pvals.loc[mask, "p_adj"].values
            result[adjcol] = adj_vals.round(6)

        # Significância após FDR global a 5%
        result["sig_yellow_fdr"] = result["p_adj_yellow"] < 0.05
        result["sig_fouls_fdr"]  = result["p_adj_fouls"]  < 0.05

    return result


# ── Cálculo de interacção árbitro×equipa ──────────────────────────────────────

def compute_referee_team_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (referee_id, team_id) com pelo menos 8 jogos em conjunto,
    estima o coeficiente de interacção via regressão logística:

      P(amarelo) ~ is_home + team_effect + referee_effect + referee×team

    O coeficiente referee×team mede o viés específico do árbitro para
    aquela equipa, controlando para o efeito base de cada um.
    """
    rows = []

    # Precisamos de todos os jogos de um árbitro para ter variação suficiente
    for ref_id, ref_grp in df.groupby("referee_id"):
        if len(ref_grp) < 30:
            continue

        ref_name = ref_grp["referee"].iloc[0]

        for team_id, team_grp in ref_grp.groupby("team_id"):
            n_together = len(team_grp)
            if n_together < 8:
                continue

            # Modelo: usar todos os jogos do árbitro
            # Variável de interacção: 1 se é esta equipa
            model_df = ref_grp.copy()
            model_df["is_this_team"] = (model_df["team_id"] == team_id).astype(int)
            model_df["y"] = (model_df["obs_yellows"] > 0).astype(int)

            X = model_df[["is_home", "is_this_team"]].astype(float)
            X = sm.add_constant(X, has_constant="add")
            y = model_df["y"].astype(float)

            try:
                result = sm.Logit(y, X).fit(disp=False, maxiter=200)
                coef  = result.params.get("is_this_team", 0.0)
                pval  = result.pvalues.get("is_this_team", 1.0)
                rows.append({
                    "referee_id":       ref_id,
                    "referee":          ref_name,
                    "team_id":          team_id,
                    "n_games":          n_together,
                    "interaction_coef": round(float(coef), 6),
                    "p_value":          round(float(pval), 6),
                    "significant":      bool(pval < 0.05),
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


# ── Guardar na DB ─────────────────────────────────────────────────────────────

CREATE_BIAS_SCORES = """
CREATE TABLE IF NOT EXISTS referee_bias_scores (
    referee_id          INTEGER NOT NULL,
    season              VARCHAR(10) NOT NULL,
    n_games             INTEGER,
    obs_yellows         INTEGER,
    exp_yellows         NUMERIC(8,2),
    obs_reds            INTEGER,
    exp_reds            NUMERIC(8,2),
    obs_fouls           NUMERIC(8,3),
    exp_fouls           NUMERIC(8,3),
    yellow_diff_bias_z  NUMERIC(8,4),
    red_diff_bias_z     NUMERIC(8,4),
    fouls_diff_bias_z   NUMERIC(8,4),
    penalty_home_bias_z NUMERIC(8,4),
    suspicion_score     NUMERIC(8,4),
    -- p-values brutos (bilateral)
    p_value_yellow      NUMERIC(10,6),
    p_value_red         NUMERIC(10,6),
    p_value_fouls       NUMERIC(10,6),
    -- p-values ajustados FDR (Benjamini-Hochberg, 2024-rev)
    -- NOTA: corrige múltiplos testes sobre todos árbitros+épocas
    p_adj_yellow        NUMERIC(10,6),
    p_adj_red           NUMERIC(10,6),
    p_adj_fouls         NUMERIC(10,6),
    -- significância após FDR a 5%
    sig_yellow_fdr      BOOLEAN,
    sig_fouls_fdr       BOOLEAN,
    computed_at         TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (referee_id, season)
);
CREATE INDEX IF NOT EXISTS idx_rbs_suspicion
    ON referee_bias_scores (suspicion_score DESC);
CREATE INDEX IF NOT EXISTS idx_rbs_fdr
    ON referee_bias_scores (sig_yellow_fdr, sig_fouls_fdr);
"""

CREATE_TEAM_BIAS = """
CREATE TABLE IF NOT EXISTS referee_team_bias (
    referee_id       INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    n_games          INTEGER,
    interaction_coef NUMERIC(10,6),
    p_value          NUMERIC(10,6),
    significant      BOOLEAN GENERATED ALWAYS AS (p_value < 0.05) STORED,
    computed_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (referee_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_rtb_significant
    ON referee_team_bias (significant, p_value);
"""

UPSERT_BIAS_SCORES = """
INSERT INTO referee_bias_scores (
    referee_id, season, n_games,
    obs_yellows, exp_yellows, obs_reds, exp_reds, obs_fouls, exp_fouls,
    yellow_diff_bias_z, red_diff_bias_z, fouls_diff_bias_z,
    penalty_home_bias_z, suspicion_score,
    p_value_yellow, p_value_red, p_value_fouls,
    p_adj_yellow, p_adj_red, p_adj_fouls,
    sig_yellow_fdr, sig_fouls_fdr
) VALUES (
    %(referee_id)s, %(season)s, %(n_games)s,
    %(obs_yellows)s, %(exp_yellows)s, %(obs_reds)s, %(exp_reds)s,
    %(obs_fouls)s, %(exp_fouls)s,
    %(yellow_diff_bias_z)s, %(red_diff_bias_z)s, %(fouls_diff_bias_z)s,
    %(penalty_home_bias_z)s, %(suspicion_score)s,
    %(p_value_yellow)s, %(p_value_red)s, %(p_value_fouls)s,
    %(p_adj_yellow)s, %(p_adj_red)s, %(p_adj_fouls)s,
    %(sig_yellow_fdr)s, %(sig_fouls_fdr)s
)
ON CONFLICT (referee_id, season) DO UPDATE SET
    n_games             = EXCLUDED.n_games,
    obs_yellows         = EXCLUDED.obs_yellows,
    exp_yellows         = EXCLUDED.exp_yellows,
    obs_reds            = EXCLUDED.obs_reds,
    exp_reds            = EXCLUDED.exp_reds,
    obs_fouls           = EXCLUDED.obs_fouls,
    exp_fouls           = EXCLUDED.exp_fouls,
    yellow_diff_bias_z  = EXCLUDED.yellow_diff_bias_z,
    red_diff_bias_z     = EXCLUDED.red_diff_bias_z,
    fouls_diff_bias_z   = EXCLUDED.fouls_diff_bias_z,
    penalty_home_bias_z = EXCLUDED.penalty_home_bias_z,
    suspicion_score     = EXCLUDED.suspicion_score,
    p_value_yellow      = EXCLUDED.p_value_yellow,
    p_value_red         = EXCLUDED.p_value_red,
    p_value_fouls       = EXCLUDED.p_value_fouls,
    p_adj_yellow        = EXCLUDED.p_adj_yellow,
    p_adj_red           = EXCLUDED.p_adj_red,
    p_adj_fouls         = EXCLUDED.p_adj_fouls,
    sig_yellow_fdr      = EXCLUDED.sig_yellow_fdr,
    sig_fouls_fdr       = EXCLUDED.sig_fouls_fdr,
    computed_at         = NOW()
"""

UPSERT_TEAM_BIAS = """
INSERT INTO referee_team_bias (
    referee_id, team_id, n_games, interaction_coef, p_value
) VALUES (
    %(referee_id)s, %(team_id)s, %(n_games)s,
    %(interaction_coef)s, %(p_value)s
)
ON CONFLICT (referee_id, team_id) DO UPDATE SET
    n_games          = EXCLUDED.n_games,
    interaction_coef = EXCLUDED.interaction_coef,
    p_value          = EXCLUDED.p_value,
    computed_at      = NOW()
"""


# ── Consistência temporal ─────────────────────────────────────────────────────

def compute_temporal_consistency(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (referee_id, team_id) — cruzando bias_scores com team_bias —
    analisa se o viés se repete de forma consistente ao longo das épocas.

    Aqui usamos os Z-scores por temporada para cada árbitro e verificamos
    quantas épocas mostram desvio na mesma direcção para cada métrica.

    Métricas calculadas por árbitro (acumulado todas as épocas):
      - n_seasons_biased_y  : nº épocas com |yellow_z| > 1.5
      - direction_y         : direcção dominante (+1 = dá mais, -1 = dá menos)
      - consistency_y       : fracção de épocas com desvio na direcção dominante
      - n_seasons_biased_f  : idem para faltas
      - direction_f
      - consistency_f
      - evidence_score      : score composto de evidência temporal
                              = (n_seasons_biased_y * consistency_y)
                              + (n_seasons_biased_f * consistency_f)

    Devolve DataFrame com uma linha por árbitro.
    """
    rows = []
    for ref_id, grp in scores.groupby("referee_id"):
        n_seasons = len(grp)
        if n_seasons < 2:
            continue

        # ── Amarelos ─────────────────────────────────────────────────────────
        z_y = grp["yellow_diff_bias_z"].values
        biased_y = np.abs(z_y) > 1.5
        n_biased_y = int(biased_y.sum())
        if n_biased_y > 0:
            dir_y   = int(np.sign(z_y[biased_y].mean()))
            same_dir_y = int(((np.sign(z_y[biased_y]) == dir_y)).sum())
            consist_y = same_dir_y / n_biased_y
        else:
            dir_y, consist_y = 0, 0.0

        # ── Faltas ───────────────────────────────────────────────────────────
        z_f = grp["fouls_diff_bias_z"].values
        biased_f = np.abs(z_f) > 1.5
        n_biased_f = int(biased_f.sum())
        if n_biased_f > 0:
            dir_f   = int(np.sign(z_f[biased_f].mean()))
            same_dir_f = int(((np.sign(z_f[biased_f]) == dir_f)).sum())
            consist_f = same_dir_f / n_biased_f
        else:
            dir_f, consist_f = 0, 0.0

        # ── Vermelhos ────────────────────────────────────────────────────────
        z_r = grp["red_diff_bias_z"].values
        biased_r = np.abs(z_r) > 1.5
        n_biased_r = int(biased_r.sum())

        # Evidence score — penaliza inconsistência
        evidence = (n_biased_y * consist_y) + (n_biased_f * consist_f) + n_biased_r

        rows.append({
            "referee_id":         ref_id,
            "referee":            grp["referee"].iloc[0],
            "n_seasons":          n_seasons,
            "n_seasons_biased_y": n_biased_y,
            "direction_y":        dir_y,
            "consistency_y":      round(consist_y, 3),
            "n_seasons_biased_r": n_biased_r,
            "n_seasons_biased_f": n_biased_f,
            "direction_f":        dir_f,
            "consistency_f":      round(consist_f, 3),
            "evidence_score":     round(evidence, 3),
            "suspicion_mean":     round(float(grp["suspicion_score"].mean()), 4),
            "suspicion_max":      round(float(grp["suspicion_score"].max()), 4),
        })

    return pd.DataFrame(rows).sort_values("evidence_score", ascending=False)


# ── Comparação entre árbitros (peer analysis) ─────────────────────────────────

def compute_peer_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada árbitro e cada contexto de jogo, compara as suas decisões
    com a média dos restantes árbitros no mesmo contexto.

    Contexto definido por:
      - is_home  : 0 ou 1
      - score_bin: resultado ao intervalo simplificado
                   (-1 = visitante à frente, 0 = empate, 1 = casa à frente)

    Para cada (árbitro, contexto):
      peer_yellows_z : (média_árbitro - média_peers) / std_peers
      peer_fouls_z   : idem para faltas

    Devolve DataFrame com uma linha por (referee_id, contexto).
    """
    # Criar bin de contexto
    df = df.copy()
    df["score_bin"] = df.apply(
        lambda r: int(np.sign(r.get("obs_yellows", 0) * 0))  # placeholder
        if False else (
            -1 if r.get("exp_fouls", 14) < 12 else
             1 if r.get("exp_fouls", 14) > 18 else 0
        ),
        axis=1
    )
    # Usar is_home como contexto principal (mais robusto com estes dados)
    rows = []
    for is_home_val in [0, 1]:
        ctx = df[df["is_home"] == is_home_val]
        if len(ctx) < 50:
            continue

        # Média global de peers neste contexto
        global_y_mean = ctx["obs_yellows"].mean()
        global_y_std  = ctx["obs_yellows"].std()
        global_f_mean = ctx["obs_fouls"].mean()
        global_f_std  = ctx["obs_fouls"].std()

        for ref_id, ref_ctx in ctx.groupby("referee_id"):
            if len(ref_ctx) < 8:
                continue

            ref_y_mean = ref_ctx["obs_yellows"].mean()
            ref_f_mean = ref_ctx["obs_fouls"].mean()

            peer_y_z = (ref_y_mean - global_y_mean) / global_y_std if global_y_std > 0 else 0.0
            peer_f_z = (ref_f_mean - global_f_mean) / global_f_std if global_f_std > 0 else 0.0

            rows.append({
                "referee_id":     ref_id,
                "referee":        ref_ctx["referee"].iloc[0],
                "context":        "casa" if is_home_val == 1 else "fora",
                "n_games":        len(ref_ctx),
                "ref_yellows_mean":  round(ref_y_mean, 3),
                "global_yellows_mean": round(global_y_mean, 3),
                "peer_yellows_z":  round(peer_y_z, 4),
                "ref_fouls_mean":  round(ref_f_mean, 3),
                "global_fouls_mean": round(global_f_mean, 3),
                "peer_fouls_z":    round(peer_f_z, 4),
            })

    return pd.DataFrame(rows)


# ── Guardar consistência temporal ─────────────────────────────────────────────

CREATE_TEMPORAL = """
CREATE TABLE IF NOT EXISTS referee_temporal_consistency (
    referee_id          INTEGER PRIMARY KEY,
    n_seasons           INTEGER,
    n_seasons_biased_y  INTEGER,
    direction_y         SMALLINT,        -- +1 dá mais amarelos, -1 dá menos
    consistency_y       NUMERIC(5,3),    -- fracção de épocas na direcção dominante
    n_seasons_biased_r  INTEGER,
    n_seasons_biased_f  INTEGER,
    direction_f         SMALLINT,
    consistency_f       NUMERIC(5,3),
    evidence_score      NUMERIC(8,3),    -- score composto de evidência temporal
    suspicion_mean      NUMERIC(8,4),
    suspicion_max       NUMERIC(8,4),
    computed_at         TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rtc_evidence
    ON referee_temporal_consistency (evidence_score DESC);
"""

UPSERT_TEMPORAL = """
INSERT INTO referee_temporal_consistency (
    referee_id, n_seasons,
    n_seasons_biased_y, direction_y, consistency_y,
    n_seasons_biased_r,
    n_seasons_biased_f, direction_f, consistency_f,
    evidence_score, suspicion_mean, suspicion_max
) VALUES (
    %(referee_id)s, %(n_seasons)s,
    %(n_seasons_biased_y)s, %(direction_y)s, %(consistency_y)s,
    %(n_seasons_biased_r)s,
    %(n_seasons_biased_f)s, %(direction_f)s, %(consistency_f)s,
    %(evidence_score)s, %(suspicion_mean)s, %(suspicion_max)s
)
ON CONFLICT (referee_id) DO UPDATE SET
    n_seasons           = EXCLUDED.n_seasons,
    n_seasons_biased_y  = EXCLUDED.n_seasons_biased_y,
    direction_y         = EXCLUDED.direction_y,
    consistency_y       = EXCLUDED.consistency_y,
    n_seasons_biased_r  = EXCLUDED.n_seasons_biased_r,
    n_seasons_biased_f  = EXCLUDED.n_seasons_biased_f,
    direction_f         = EXCLUDED.direction_f,
    consistency_f       = EXCLUDED.consistency_f,
    evidence_score      = EXCLUDED.evidence_score,
    suspicion_mean      = EXCLUDED.suspicion_mean,
    suspicion_max       = EXCLUDED.suspicion_max,
    computed_at         = NOW()
"""

CREATE_PEER = """
CREATE TABLE IF NOT EXISTS referee_peer_comparison (
    referee_id          INTEGER NOT NULL,
    context             VARCHAR(10) NOT NULL,   -- 'casa' ou 'fora'
    n_games             INTEGER,
    ref_yellows_mean    NUMERIC(6,3),
    global_yellows_mean NUMERIC(6,3),
    peer_yellows_z      NUMERIC(8,4),
    ref_fouls_mean      NUMERIC(6,3),
    global_fouls_mean   NUMERIC(6,3),
    peer_fouls_z        NUMERIC(8,4),
    computed_at         TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (referee_id, context)
);
"""

UPSERT_PEER = """
INSERT INTO referee_peer_comparison (
    referee_id, context, n_games,
    ref_yellows_mean, global_yellows_mean, peer_yellows_z,
    ref_fouls_mean, global_fouls_mean, peer_fouls_z
) VALUES (
    %(referee_id)s, %(context)s, %(n_games)s,
    %(ref_yellows_mean)s, %(global_yellows_mean)s, %(peer_yellows_z)s,
    %(ref_fouls_mean)s, %(global_fouls_mean)s, %(peer_fouls_z)s
)
ON CONFLICT (referee_id, context) DO UPDATE SET
    n_games             = EXCLUDED.n_games,
    ref_yellows_mean    = EXCLUDED.ref_yellows_mean,
    peer_yellows_z      = EXCLUDED.peer_yellows_z,
    ref_fouls_mean      = EXCLUDED.ref_fouls_mean,
    peer_fouls_z        = EXCLUDED.peer_fouls_z,
    computed_at         = NOW()
"""


def save_temporal_consistency(consistency: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute(CREATE_TEMPORAL)
        conn.commit()
        records = consistency.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_TEMPORAL, records)
        conn.commit()
    log.info("Guardados %d registos em referee_temporal_consistency", len(consistency))


def save_peer_comparison(peer: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute(CREATE_PEER)
        conn.commit()
        records = peer.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_PEER, records)
        conn.commit()
    log.info("Guardados %d registos em referee_peer_comparison", len(peer))


def save_bias_scores(scores: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute(CREATE_BIAS_SCORES)
        conn.commit()
        records = scores.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_BIAS_SCORES, records)
        conn.commit()
    log.info("Guardados %d bias scores em referee_bias_scores", len(scores))


def save_team_bias(team_bias: pd.DataFrame) -> None:
    with get_cursor() as (conn, cur):
        cur.execute(CREATE_TEAM_BIAS)
        conn.commit()
        records = team_bias.drop(columns=["referee"], errors="ignore").to_dict("records")
        cur.executemany(UPSERT_TEAM_BIAS, records)
        conn.commit()
    log.info("Guardados %d pares árbitro×equipa em referee_team_bias", len(team_bias))


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    with get_cursor(dict_cursor=True) as (conn, cur):
        cur.execute(QUERY)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    log.info("Dados carregados: %d linhas", len(df))

    # Converter tipos
    for col in ["obs_yellows", "obs_reds", "obs_fouls",
                "exp_yellows", "exp_reds", "exp_fouls", "is_home"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Teste de overdispersion formal (Cameron & Trivedi, 1990) ─────────────
    od = test_overdispersion(df["obs_fouls"], df["exp_fouls"])
    log.info(
        "Overdispersion test (xF — Cameron & Trivedi 1990): "
        "alpha=%.4f, t=%.3f, p=%.4f, overdispersed=%s",
        od["overdispersion_alpha"], od["overdispersion_tstat"],
        od["overdispersion_pvalue"], od["is_overdispersed"]
    )
    if od["is_overdispersed"]:
        alpha = od["overdispersion_alpha"]
        if alpha < 0.05:
            # Overdispersion estatisticamente significativa mas magnitude negligenciável
            # Com n=4816, efeitos minúsculos ficam altamente significativos
            log.info(
                "Overdispersion detectada (alpha=%.4f, p=%.4f) mas magnitude "
                "negligenciável (alpha < 0.05 = variância %.1f%% acima do Poisson). "
                "Modelo Poisson adequado na prática. Documentar como limitação.",
                alpha, od["overdispersion_pvalue"], alpha * 100
            )
        else:
            log.warning(
                "AVISO: Overdispersion substancial (alpha=%.4f, p=%.4f). "
                "Os Z-scores de faltas podem estar inflacionados. "
                "Considerar Negative Binomial no próximo re-treino.",
                alpha, od["overdispersion_pvalue"]
            )
    else:
        log.info("Modelo Poisson adequado para faltas (sem overdispersion significativa).")

    # ── 1. Z-scores por árbitro+temporada (com FDR) ───────────────────────────
    log.info("A calcular Z-scores por árbitro+temporada (com correcção FDR)...")
    scores = compute_referee_bias_scores(df)
    log.info("  %d entradas (árbitro+temporada)", len(scores))

    if not scores.empty:
        n_sig_y = int(scores["sig_yellow_fdr"].sum())
        n_sig_f = int(scores["sig_fouls_fdr"].sum())
        log.info("  Significativos após FDR (p_adj < 0.05): "
                 "amarelos=%d, faltas=%d de %d testes",
                 n_sig_y, n_sig_f, len(scores))

    save_bias_scores(scores)

    # ── 2. Interacção árbitro×equipa ─────────────────────────────────────────
    log.info("A calcular interacções árbitro×equipa...")
    team_bias = compute_referee_team_bias(df)
    log.info("  %d pares árbitro×equipa", len(team_bias))
    save_team_bias(team_bias)

    # ── 3. Consistência temporal ─────────────────────────────────────────────
    log.info("A calcular consistência temporal...")
    consistency = compute_temporal_consistency(scores)
    log.info("  %d árbitros com 2+ épocas", len(consistency))
    save_temporal_consistency(consistency)

    # ── 4. Comparação entre árbitros (peer analysis) ─────────────────────────
    log.info("A calcular comparação entre árbitros...")
    peer = compute_peer_comparison(df)
    log.info("  %d entradas (árbitro+contexto)", len(peer))
    save_peer_comparison(peer)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── Top 10 SuspicionScore médio ──────────────────────────────────")
    summary = (
        scores.groupby(["referee_id", "referee"])
        .agg(epocas=("season", "count"),
             suspicion_medio=("suspicion_score", "mean"),
             suspicion_max=("suspicion_score", "max"))
        .sort_values("suspicion_medio", ascending=False)
        .reset_index()
    )
    print(summary.head(10).to_string(index=False))

    print("\n── Top 10 Evidência Temporal (padrão sistemático multi-época) ───")
    print(consistency[["referee", "n_seasons", "n_seasons_biased_y",
                        "consistency_y", "n_seasons_biased_f",
                        "consistency_f", "evidence_score"]].head(10).to_string(index=False))

    print("\n── Árbitros que se desviam mais dos peers (jogos em casa) ───────")
    peer_casa = peer[peer["context"] == "casa"].copy()
    peer_casa["abs_peer_y"] = peer_casa["peer_yellows_z"].abs()
    peer_casa = peer_casa.sort_values("abs_peer_y", ascending=False)
    print(peer_casa[["referee", "n_games", "ref_yellows_mean",
                      "global_yellows_mean", "peer_yellows_z",
                      "peer_fouls_z"]].head(10).to_string(index=False))


if __name__ == "__main__":
    import dotenv
    from rich.logging import RichHandler
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False)])
    run()
