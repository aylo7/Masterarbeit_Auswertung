# ============================================================
# 06_correlations_spearman.py
# Ziel:
# - survey_scales.xlsx laden
# - Spearman-Korrelationen zwischen Skalen berechnen
# - Gesamt + getrennt nach Version (A/B)
# - Optional: Holm-Korrektur über alle Paar-Korrelationen
# - Export nach outputs/spearman_correlations.xlsx
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# ------------------------------------------------------------
# 1) Pfade definieren
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
SCALES_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"
OUT_FILE = BASE_DIR / "outputs" / "spearman_correlations.xlsx"

if not SCALES_FILE.exists():
    raise FileNotFoundError(
        f"Nicht gefunden: {SCALES_FILE}\n"
        "Bitte zuerst Schritt 3 ausführen."
    )

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(SCALES_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Welche Variablen korrelieren wir?
#    (Median-Skalen = ordinal-freundlicher)
# ------------------------------------------------------------

scales = [
    "cx_median",
    "personalization_median",
    "trust_median",
    "transparency_median",
    "satisfaction_median",
    "loyalty_median",
]

# Sicherheitscheck: nur Spalten nehmen, die existieren
scales = [c for c in scales if c in df.columns]

# ------------------------------------------------------------
# 4) Holm-Korrektur (für multiple Korrelationstests)
# ------------------------------------------------------------

def holm_correction(p_values):
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

# ------------------------------------------------------------
# 5) Funktion: Spearman-Matrix + Long-Format Tabelle
# ------------------------------------------------------------

def spearman_tables(data, label="total"):
    """
    Erzeugt:
    - Korrelationsmatrix (rho)
    - p-Wert-Matrix
    - Long-Format Tabelle (paarweise)
    """
    # Spearmanr kann direkt Matrix liefern
    rho, p = spearmanr(data[scales], nan_policy="omit")

    # In Matrixform bringen
    rho_mat = pd.DataFrame(rho, index=scales, columns=scales)
    p_mat = pd.DataFrame(p, index=scales, columns=scales)

    # Long-Format: nur obere Dreiecksmatrix ohne Diagonale
    rows = []
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            v1, v2 = scales[i], scales[j]
            rows.append({
                "set": label,
                "var1": v1,
                "var2": v2,
                "spearman_rho": float(rho_mat.loc[v1, v2]),
                "p_value": float(p_mat.loc[v1, v2]),
                "n_pairwise": int(data[[v1, v2]].dropna().shape[0])
            })

    long_df = pd.DataFrame(rows)

    # Holm-Korrektur über alle Paar-Tests in diesem Set
    long_df["p_holm"] = holm_correction(long_df["p_value"].tolist())

    return rho_mat, p_mat, long_df

# ------------------------------------------------------------
# 6) Gesamt + nach Version berechnen
# ------------------------------------------------------------

rho_total, p_total, long_total = spearman_tables(df, label="total")

rho_A, p_A, long_A = spearman_tables(df[df["version"] == "A"].copy(), label="A")
rho_B, p_B, long_B = spearman_tables(df[df["version"] == "B"].copy(), label="B")

long_all = pd.concat([long_total, long_A, long_B], ignore_index=True)

print("===== SPEARMAN (Long) Preview =====")
print(long_all.head(10))
print()

# ------------------------------------------------------------
# 7) Export nach Excel (mehrere Sheets)
# ------------------------------------------------------------

with pd.ExcelWriter(OUT_FILE) as writer:
    rho_total.to_excel(writer, sheet_name="rho_total")
    p_total.to_excel(writer, sheet_name="p_total")
    rho_A.to_excel(writer, sheet_name="rho_A")
    p_A.to_excel(writer, sheet_name="p_A")
    rho_B.to_excel(writer, sheet_name="rho_B")
    p_B.to_excel(writer, sheet_name="p_B")
    long_all.to_excel(writer, sheet_name="long_all", index=False)

print("Export:", OUT_FILE)
print("DONE.")