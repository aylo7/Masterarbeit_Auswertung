# ============================================================
# 12_bootstrap_effectsize_ci.py
# Zweck:
# - Bootstrap-Konfidenzintervalle (95%) für Cliff's delta (A vs. B)
# - Für Skalen (aus survey_scales.xlsx) UND für Top-Items (aus item_level_tests.xlsx)
#
# Warum?
# - p-Werte + Effektgröße sind gut, aber CI zeigt Unsicherheit und macht die Interpretation stärker.
#
# Methodik:
# - Stratified Bootstrap: innerhalb Gruppe A und Gruppe B getrennt resamplen
# - pro Bootstrap: Cliff's delta berechnen
# - 95%-KI über Quantile (2.5% / 97.5%)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1) Pfade definieren
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

# Skalen-Datensatz (enthält *_median Skalen und version A/B)
SCALES_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"

# Item-Datensatz (enthält alle Einzelitems)
CLEAN_FILE = BASE_DIR / "data" / "processed" / "survey_clean.xlsx"

# Ergebnis aus Schritt 11 (enthält Top-Items)
ITEM_TESTS_FILE = BASE_DIR / "outputs" / "item_level_tests.xlsx"

# Output
OUT_FILE = BASE_DIR / "outputs" / "effectsize_bootstrap_ci.xlsx"

# Sicherheitschecks
if not SCALES_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {SCALES_FILE} (Schritt 3 ausführen)")
if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CLEAN_FILE} (Schritt 2 ausführen)")
if not ITEM_TESTS_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {ITEM_TESTS_FILE} (Schritt 11 ausführen)")

# ------------------------------------------------------------
# 2) Parameter
# ------------------------------------------------------------

# Bootstrap-Anzahl (du hast 5000 schon mehrfach erfolgreich genutzt)
N_BOOT = 5000

# Seed für Reproduzierbarkeit
RNG = np.random.default_rng(42)

# Wie viele Top-Items (aus Schritt 11) sollen wir bootstrappen?
TOP_N_ITEMS = 10

# ------------------------------------------------------------
# 3) Daten laden
# ------------------------------------------------------------

df_scales = pd.read_excel(SCALES_FILE)
df_items = pd.read_excel(CLEAN_FILE)

# Item-Level Testergebnisse laden (Sheet "item_tests_sorted")
item_tests_sorted = pd.read_excel(ITEM_TESTS_FILE, sheet_name="item_tests_sorted")

print("===== DATEN GELADEN =====")
print("Scales rows:", df_scales.shape[0], "| Items rows:", df_items.shape[0])
print("Version counts (scales):\n", df_scales["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 4) Hilfsfunktionen
# ------------------------------------------------------------

def cliffs_delta_from_arrays(a, b):
    """
    Cliff's delta über Mann–Whitney-U:
      delta = (2U)/(n1*n2) - 1
    a = Werte aus Gruppe A
    b = Werte aus Gruppe B
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)

    # Sicherheitscheck
    if n1 == 0 or n2 == 0:
        return np.nan

    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * U) / (n1 * n2) - 1)

def bootstrap_delta_ci(df, value_col, group_col="version", groupA="A", groupB="B"):
    """
    Berechnet:
    - delta (Punkt-Schätzer)
    - 95% Bootstrap-KI
    - p_boot (approx.) zweiseitig
    """

    # Originaldaten je Gruppe ziehen
    a = df[df[group_col] == groupA][value_col].dropna().to_numpy()
    b = df[df[group_col] == groupB][value_col].dropna().to_numpy()

    # Punkt-Schätzer
    delta_hat = cliffs_delta_from_arrays(a, b)

    # Bootstrap-Verteilung
    boot = np.zeros(N_BOOT)

    for i in range(N_BOOT):
        # innerhalb jeder Gruppe resamplen (stratifiziert)
        a_bs = RNG.choice(a, size=len(a), replace=True)
        b_bs = RNG.choice(b, size=len(b), replace=True)

        boot[i] = cliffs_delta_from_arrays(a_bs, b_bs)

        # kleine Progress-Anzeige alle 1000 Iterationen
        if (i + 1) % 1000 == 0:
            pass  # absichtlich kein Print (sonst zu spammy)

    # 95%-KI
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))

    # p_boot (approx.): Anteil <=0 bzw >=0
    p_boot = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p_boot = float(min(p_boot, 1.0))

    return {
        "nA": int(len(a)),
        "nB": int(len(b)),
        "delta_hat": float(delta_hat),
        "ci_low_2.5%": ci_low,
        "ci_high_97.5%": ci_high,
        "p_boot_approx": p_boot
    }

# ------------------------------------------------------------
# 5) Variablen definieren: Skalen + Top-Items
# ------------------------------------------------------------

# Skalen: wir nehmen die Median-Skalen (ordinal-freundlicher)
scale_vars = [
    "cx_median",
    "personalization_median",
    "trust_median",
    "transparency_median",
    "satisfaction_median",
    "loyalty_median",
]
scale_vars = [c for c in scale_vars if c in df_scales.columns]

# Top-Items aus Schritt 11 (inkl. Manipulationschecks, wie in deiner Top-10 Liste)
top_items = item_tests_sorted["item"].head(TOP_N_ITEMS).tolist()
top_items = [c for c in top_items if c in df_items.columns]

print("Skalen für Bootstrap:", scale_vars)
print("Top-Items für Bootstrap:", top_items)
print()

# ------------------------------------------------------------
# 6) Bootstrap für Skalen
# ------------------------------------------------------------

scale_rows = []
for var in scale_vars:
    res = bootstrap_delta_ci(df_scales, var, group_col="version", groupA="A", groupB="B")
    res.update({"variable": var, "type": "scale_median"})
    scale_rows.append(res)

scale_ci = pd.DataFrame(scale_rows).sort_values("delta_hat", ascending=False)

print("===== BOOTSTRAP CI (Skalen) Preview =====")
print(scale_ci)
print()

# ------------------------------------------------------------
# 7) Bootstrap für Top-Items
# ------------------------------------------------------------

item_rows = []
for var in top_items:
    res = bootstrap_delta_ci(df_items, var, group_col="version", groupA="A", groupB="B")
    res.update({"variable": var, "type": "item"})
    item_rows.append(res)

item_ci = pd.DataFrame(item_rows).sort_values("delta_hat", ascending=False)

print("===== BOOTSTRAP CI (Top-Items) Preview =====")
print(item_ci)
print()

# ------------------------------------------------------------
# 8) Export
# ------------------------------------------------------------

settings = pd.DataFrame([{
    "N_BOOT": N_BOOT,
    "TOP_N_ITEMS": TOP_N_ITEMS,
    "note": "Bootstrap CI für Cliff's delta (A vs B), stratifiziertes Resampling."
}])

with pd.ExcelWriter(OUT_FILE) as writer:
    scale_ci.to_excel(writer, sheet_name="scales_ci", index=False)
    item_ci.to_excel(writer, sheet_name="top_items_ci", index=False)
    settings.to_excel(writer, sheet_name="settings", index=False)

print("Export:", OUT_FILE)
print("DONE.")