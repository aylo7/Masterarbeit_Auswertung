# ============================================================
# 09_demographics_usage_associations.py
# Zweck (Explorativ):
# (1) Randomisierung-Check:
#     Prüfen, ob Demografie/Nutzung in Version A vs B ähnlich verteilt ist
#     -> Chi-Quadrat oder Fisher Exact (bei 2x2)
#
# (2) Demografie/Nutzung -> Skalenbewertungen (gesamt):
#     - 2 Gruppen: Mann–Whitney U + Cliff's delta
#     - >2 Gruppen: Kruskal–Wallis H + Epsilon-Squared (ε²)
#     -> Holm-Korrektur über alle Tests
#
# (3) Subgruppenanalyse:
#     A vs B innerhalb jeder Kategorie (z.B. Männer/Frauen; Altersgruppen; DHP use ja/nein)
#     -> Mann–Whitney U + Cliff's delta je Subgruppe
#     -> nur wenn nA und nB jeweils >= min_cell
#
# Output:
#   outputs/demographics_usage_associations.xlsx
#
# Hinweis:
# - Explorativer Block: Ergebnisse vorsichtig interpretieren und Multiple Testing beachten.
# - PLZ/Region wird hier bewusst NICHT genutzt.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact

# ------------------------------------------------------------
# 1) Pfade definieren
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"
OUT_FILE = BASE_DIR / "outputs" / "demographics_usage_associations.xlsx"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Nicht gefunden: {DATA_FILE}\n"
        "Bitte zuerst Schritt 3 ausführen (survey_scales.xlsx erzeugen)."
    )

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(DATA_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Skalen festlegen (Median-Skalen = ordinal-freundlich)
# ------------------------------------------------------------

scale_cols = [
    "cx_median",
    "personalization_median",
    "trust_median",
    "transparency_median",
    "satisfaction_median",
    "loyalty_median",
]
scale_cols = [c for c in scale_cols if c in df.columns]

# Für Subgruppen-Analysen fokussieren wir primär auf Hypothesen-Konstrukte
primary_scales = ["cx_median", "trust_median", "satisfaction_median", "loyalty_median"]
primary_scales = [c for c in primary_scales if c in df.columns]

# ------------------------------------------------------------
# 4) Demografie/Nutzung Variablen festlegen (ohne PLZ)
# ------------------------------------------------------------

predictors = [
    "gender",
    "age_group",
    "education",
    "employment",
    "dhp_use_12m",
    "dhp_freq",
    "ai_tools_use_12m",
]
predictors = [c for c in predictors if c in df.columns]

print("Predictors used:", predictors)
print("Scales used:", scale_cols)
print()

# ------------------------------------------------------------
# 5) Hilfsfunktionen: Holm / Effektgrößen
# ------------------------------------------------------------

def holm_correction(p_values):
    """Holm-Korrektur: p_adj in Original-Reihenfolge"""
    p = np.array(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    p_adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        factor = m - i
        val = factor * p_sorted[i]
        running_max = max(running_max, val)
        p_adj_sorted[i] = running_max
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = np.minimum(p_adj_sorted, 1.0)
    return p_adj.tolist()

def cliffs_delta(a, b):
    """
    Cliff's delta (ordinal Effektgröße) über Mann–Whitney-U:
      delta = (2U)/(n1*n2) - 1
    delta > 0: Gruppe a tendenziell höhere Werte als b
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * U) / (n1 * n2) - 1)

def epsilon_squared_kruskal(H, k, n):
    """Kruskal–Wallis Effektgröße: ε² = (H - k + 1) / (n - k)"""
    if n <= k:
        return np.nan
    return float((H - k + 1) / (n - k))

# ------------------------------------------------------------
# 6) (1) Randomisierung-Check: Predictor vs Version (A/B)
# ------------------------------------------------------------

randomization_rows = []

for pred in predictors:
    ct = pd.crosstab(df["version"], df[pred], dropna=False)  # A/B x Kategorien

    is_2x2 = (ct.shape[0] == 2) and (ct.shape[1] == 2)

    if is_2x2:
        # Fisher Exact: besser bei kleinen erwarteten Häufigkeiten
        oddsratio, p = fisher_exact(ct.to_numpy())
        test_name = "Fisher exact (2x2)"
        stat = float(oddsratio)
    else:
        chi2, p, dof, exp = chi2_contingency(ct.to_numpy())
        test_name = "Chi-square"
        stat = float(chi2)

    randomization_rows.append({
        "predictor": pred,
        "test": test_name,
        "statistic": stat,
        "p_value": float(p),
        "table_shape": str(ct.shape),
        "note": "Randomization check: Verteilung Predictor in A vs B"
    })

randomization_df = pd.DataFrame(randomization_rows)
randomization_df["p_holm"] = holm_correction(randomization_df["p_value"].tolist())

print("===== RANDOMISIERUNG-CHECK (Predictor vs Version) =====")
print(randomization_df[["predictor", "test", "p_value", "p_holm"]])
print()

# ------------------------------------------------------------
# 7) (2) Predictor -> Skalen (gesamt)
# ------------------------------------------------------------

rows_total = []

for pred in predictors:
    for sc in scale_cols:
        tmp = df[[pred, sc]].dropna().copy()

        # Wenn weniger als 2 Gruppen vorhanden: skip
        if tmp[pred].nunique() < 2:
            continue

        # 2 Gruppen -> Mann–Whitney
        if tmp[pred].nunique() == 2:
            g1, g2 = tmp[pred].unique().tolist()
            x = tmp[tmp[pred] == g1][sc]
            y = tmp[tmp[pred] == g2][sc]

            U, p = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)

            rows_total.append({
                "set": "total",
                "predictor": pred,
                "scale": sc,
                "test": "Mann-Whitney U",
                "n": int(tmp.shape[0]),
                "groups": f"{g1} vs {g2}",
                "median_g1": float(x.median()),
                "median_g2": float(y.median()),
                "statistic": float(U),
                "p_value": float(p),
                "effect": float(delta),
                "effect_name": "Cliff's delta (g1 vs g2)"
            })

        # >2 Gruppen -> Kruskal–Wallis
        else:
            groups = [g[sc].to_numpy() for _, g in tmp.groupby(pred)]
            H, p = kruskal(*groups)

            k = tmp[pred].nunique()
            n = tmp.shape[0]
            eps2 = epsilon_squared_kruskal(H, k, n)

            medians = tmp.groupby(pred)[sc].median().to_dict()

            rows_total.append({
                "set": "total",
                "predictor": pred,
                "scale": sc,
                "test": "Kruskal-Wallis H",
                "n": int(n),
                "groups": f"{k} groups",
                "group_medians": str(medians),
                "statistic": float(H),
                "p_value": float(p),
                "effect": float(eps2),
                "effect_name": "Epsilon-squared (ε²)"
            })

total_df = pd.DataFrame(rows_total)
if len(total_df) > 0:
    total_df["p_holm"] = holm_correction(total_df["p_value"].tolist())

print("===== PREDICTOR -> SKALEN (TOTAL) Preview =====")
print(total_df.head(10))
print()

# ------------------------------------------------------------
# 8) (3) Subgruppen: A vs B innerhalb jeder Kategorie (min_cell)
# ------------------------------------------------------------

min_cell = 10
sub_rows = []

for pred in predictors:
    # nur echte Kategorien, Missing ignorieren
    levels = df[pred].dropna().unique().tolist()

    for level in levels:
        sub = df[df[pred] == level].copy()

        nA = int((sub["version"] == "A").sum())
        nB = int((sub["version"] == "B").sum())

        # kleine Zellen überspringen (sonst instabil)
        if (nA < min_cell) or (nB < min_cell):
            continue

        for sc in primary_scales:
            x = sub[sub["version"] == "A"][sc].dropna()
            y = sub[sub["version"] == "B"][sc].dropna()

            U, p = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)

            sub_rows.append({
                "predictor": pred,
                "level": str(level),
                "scale": sc,
                "nA": int(len(x)),
                "median_A": float(x.median()),
                "nB": int(len(y)),
                "median_B": float(y.median()),
                "U": float(U),
                "p_value": float(p),
                "cliffs_delta(A_vs_B)": float(delta),
                "note": f"Only levels with nA,nB >= {min_cell}"
            })

sub_df = pd.DataFrame(sub_rows)
if len(sub_df) > 0:
    sub_df["p_holm"] = holm_correction(sub_df["p_value"].tolist())

print("===== SUBGROUP (A vs B innerhalb Predictor-Level) Preview =====")
print(sub_df.head(10))
print()

# ------------------------------------------------------------
# 9) Export
# ------------------------------------------------------------

with pd.ExcelWriter(OUT_FILE) as writer:
    randomization_df.to_excel(writer, sheet_name="randomization_check", index=False)
    total_df.to_excel(writer, sheet_name="predictor_to_scales_total", index=False)
    sub_df.to_excel(writer, sheet_name="A_vs_B_within_levels", index=False)

print("Export:", OUT_FILE)
print("DONE.")