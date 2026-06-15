# ============================================================
# 08_moderation_transparency_trust.py
# Ziel:
# - survey_scales.xlsx laden
# - Transparenz (transparency_median) in LOW vs HIGH splitten (Median-Split)
# - Innerhalb jeder Transparenz-Gruppe:
#     Teste KI-Effekt auf Trust (trust_median): Mann–Whitney U + Cliff's delta
# - Moderationssignal:
#     Vergleiche Effektgrößen: Delta_diff = delta_high - delta_low
#     Bootstrap-CI für Delta_diff (nonparametrisch)
# - Export nach outputs/moderation_transparency_trust.xlsx
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1) Pfade
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"
OUT_FILE = BASE_DIR / "outputs" / "moderation_transparency_trust.xlsx"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Nicht gefunden: {DATA_FILE}\n"
        "Bitte zuerst Schritt 3 ausführen (survey_scales.xlsx erzeugen)."
    )

# ------------------------------------------------------------
# 2) Daten laden + benötigte Spalten prüfen
# ------------------------------------------------------------

df = pd.read_excel(DATA_FILE)

needed = ["version", "condition_ai", "trust_median", "transparency_median"]
missing = [c for c in needed if c not in df.columns]
if len(missing) > 0:
    raise ValueError("Fehlende Spalten: " + ", ".join(missing))

# Nur relevante Spalten auswählen
d = df[needed].dropna().copy()

print("===== DATEN GELADEN =====")
print("N:", d.shape[0])
print("Version counts:\n", d["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Transparenz-Gruppen bilden (Median-Split)
# ------------------------------------------------------------

# transparency_median ist bei dir ein Single-Item (1-7), daher integer/ordinal
t_median = float(d["transparency_median"].median())

# Regel:
# - LOW: <= Median
# - HIGH: > Median
d["transparency_group"] = np.where(d["transparency_median"] <= t_median, "low", "high")

print("===== TRANSPARENCY SPLIT =====")
print("Median (Cut):", t_median)
print(d["transparency_group"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 4) Hilfsfunktionen: Cliff's delta + Mann–Whitney U
# ------------------------------------------------------------

def cliffs_delta(a, b):
    """
    Cliff's delta Effektgröße:
      delta = (2U)/(n1*n2) - 1, wobei U für sample a berechnet wird.
    Interpretation:
      delta > 0  -> a tendenziell höhere Werte als b
      |delta| nahe 1 -> sehr großer Effekt
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan

    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * U) / (n1 * n2) - 1)

def mannwhitney_trust_effect(sub_df, groupA="A", groupB="B"):
    """
    Testet innerhalb eines Subsets (z.B. transparency_group=low):
      Trust in Version A vs Version B
    """
    x = sub_df[sub_df["version"] == groupA]["trust_median"].dropna()
    y = sub_df[sub_df["version"] == groupB]["trust_median"].dropna()

    U, p = mannwhitneyu(x, y, alternative="two-sided")
    delta = cliffs_delta(x, y)

    return {
        "nA": int(len(x)),
        "median_A": float(x.median()) if len(x) else np.nan,
        "nB": int(len(y)),
        "median_B": float(y.median()) if len(y) else np.nan,
        "U": float(U),
        "p_two_sided": float(p),
        "cliffs_delta(A_vs_B)": float(delta),
    }

# ------------------------------------------------------------
# 5) Ergebnisse pro Transparenz-Gruppe
# ------------------------------------------------------------

rows = []

for g in ["low", "high"]:
    sub = d[d["transparency_group"] == g].copy()
    res = mannwhitney_trust_effect(sub, groupA="A", groupB="B")
    res["transparency_group"] = g
    rows.append(res)

result_by_group = pd.DataFrame(rows)

print("===== TRUST: A vs B innerhalb Transparenz-Gruppen =====")
print(result_by_group)
print()

# ------------------------------------------------------------
# 6) Moderationssignal: Unterschied der Effektgrößen (Delta_diff)
# ------------------------------------------------------------

# delta_high - delta_low
delta_low = float(result_by_group.loc[result_by_group["transparency_group"] == "low", "cliffs_delta(A_vs_B)"].iloc[0])
delta_high = float(result_by_group.loc[result_by_group["transparency_group"] == "high", "cliffs_delta(A_vs_B)"].iloc[0])
delta_diff = delta_high - delta_low

print("===== MODERATION SIGNAL (Effect-size difference) =====")
print("delta_low:", delta_low)
print("delta_high:", delta_high)
print("delta_diff (high-low):", delta_diff)
print()

# ------------------------------------------------------------
# 7) Bootstrap-CI für delta_diff (nonparametrisch)
# ------------------------------------------------------------

n_boot = 5000
rng = np.random.default_rng(42)

# Für Bootstrap ziehen wir innerhalb jeder transparency_group separat mit Zurücklegen
boot_diffs = np.zeros(n_boot)

def bootstrap_delta(sub_df):
    """
    Bootstrap-Schätzung der Cliff's delta innerhalb eines Subsets (A vs B).
    """
    # getrennt nach Version resamplen (damit Gruppengrößen stabil bleiben)
    a = sub_df[sub_df["version"] == "A"]["trust_median"].to_numpy()
    b = sub_df[sub_df["version"] == "B"]["trust_median"].to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    a_bs = rng.choice(a, size=len(a), replace=True)
    b_bs = rng.choice(b, size=len(b), replace=True)

    # Cliff's delta aus resampled arrays
    U, _ = mannwhitneyu(a_bs, b_bs, alternative="two-sided")
    return float((2 * U) / (len(a_bs) * len(b_bs)) - 1)

low_df = d[d["transparency_group"] == "low"].copy()
high_df = d[d["transparency_group"] == "high"].copy()

for i in range(n_boot):
    dl = bootstrap_delta(low_df)
    dh = bootstrap_delta(high_df)
    boot_diffs[i] = dh - dl

ci_low = float(np.quantile(boot_diffs, 0.025))
ci_high = float(np.quantile(boot_diffs, 0.975))

# p-like: Anteil <=0 bzw >=0 (zweiseitig grob)
p_boot = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
p_boot = float(min(p_boot, 1.0))

print("===== BOOTSTRAP delta_diff =====")
print("delta_diff:", float(delta_diff))
print("95% CI:", (ci_low, ci_high))
print("p_boot (approx):", p_boot)
print()

# ------------------------------------------------------------
# 8) Export
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "transparency_split_cut_median": t_median,
    "delta_low": delta_low,
    "delta_high": delta_high,
    "delta_diff_high_minus_low": float(delta_diff),
    "boot_ci_low_2.5%": ci_low,
    "boot_ci_high_97.5%": ci_high,
    "boot_p_approx": p_boot,
    "note": "Moderations-Sensitivität: Median-Split Transparenz, A vs B auf Trust je Gruppe + Bootstrap auf Δδ"
}])

with pd.ExcelWriter(OUT_FILE) as writer:
    result_by_group.to_excel(writer, sheet_name="trust_by_transparency", index=False)
    summary.to_excel(writer, sheet_name="moderation_summary", index=False)
    pd.DataFrame({"boot_delta_diff": boot_diffs}).to_excel(writer, sheet_name="bootstrap_dist", index=False)

print("Export:", OUT_FILE)
print("DONE.")