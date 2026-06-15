# ============================================================
# 11_item_level_tests.py
# Ziel:
# - survey_clean.xlsx laden (enthält alle Einzelitems + version A/B)
# - Alle Likert-Items (1–7) itemweise zwischen A vs B vergleichen
# - Test: Mann–Whitney U (zweiseitig)
# - Effektgröße: Cliff's delta
# - Multiple Testing: Holm-Korrektur über alle Items
# - Zusätzlich: Median_A / Median_B, nA / nB
# - Export nach outputs/item_level_tests.xlsx
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import json

# ------------------------------------------------------------
# 1) Pfade definieren
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

# Input: bereinigter Datensatz mit kurzen Spaltennamen (aus Schritt 2)
CLEAN_FILE = BASE_DIR / "data" / "processed" / "survey_clean.xlsx"

# Input: Constructs (Item-Gruppierung) aus Schritt 2
CONSTRUCTS_FILE = BASE_DIR / "outputs" / "constructs.json"

# Output
OUT_FILE = BASE_DIR / "outputs" / "item_level_tests.xlsx"

# Sicherheitschecks
if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CLEAN_FILE} (Bitte Schritt 2 ausführen)")
if not CONSTRUCTS_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CONSTRUCTS_FILE} (Bitte Schritt 2 ausführen)")

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(CLEAN_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Constructs laden (enthält alle Likert Items inkl. Manipulationscheck)
# ------------------------------------------------------------

with open(CONSTRUCTS_FILE, "r", encoding="utf-8") as f:
    constructs_raw = json.load(f)

def normalize_items(v):
    # constructs.json kann Liste oder dict sein
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        try:
            keys_sorted = sorted(v.keys(), key=lambda x: int(x))
            return [v[k] for k in keys_sorted]
        except Exception:
            return list(v.values())
    return []

constructs = {k: normalize_items(v) for k, v in constructs_raw.items()}

# Alle Items einsammeln (ohne Skalen, nur Itemspalten)
all_items = []
for _, items in constructs.items():
    all_items.extend(items)

# Duplikate entfernen
all_items = list(dict.fromkeys(all_items))

# Sicherstellen, dass Items im df existieren
all_items = [c for c in all_items if c in df.columns]

print("Anzahl Items für Item-Tests:", len(all_items))
print()

# ------------------------------------------------------------
# 4) Hilfsfunktionen: Cliff's delta + Holm
# ------------------------------------------------------------

def cliffs_delta(a, b):
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * U) / (n1 * n2) - 1)

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
# 5) Item-Tests A vs B
# ------------------------------------------------------------

rows = []

for item in all_items:
    x = df[df["version"] == "A"][item].dropna()
    y = df[df["version"] == "B"][item].dropna()

    # Mann–Whitney U
    U, p = mannwhitneyu(x, y, alternative="two-sided")

    # Effektgröße
    delta = cliffs_delta(x, y)

    rows.append({
        "item": item,
        "nA": int(len(x)),
        "median_A": float(x.median()) if len(x) else np.nan,
        "nB": int(len(y)),
        "median_B": float(y.median()) if len(y) else np.nan,
        "U": float(U),
        "p_two_sided": float(p),
        "cliffs_delta(A_vs_B)": float(delta),
    })

res = pd.DataFrame(rows)

# Holm-Korrektur über alle Items
res["p_holm"] = holm_correction(res["p_two_sided"].tolist())

# Sortieren: erst nach p_holm, dann nach Effektgröße
res_sorted = res.sort_values(["p_holm", "cliffs_delta(A_vs_B)"], ascending=[True, False]).reset_index(drop=True)

print("===== ITEM TESTS (Top 10 nach p_holm) =====")
print(res_sorted.head(10))
print()

# ------------------------------------------------------------
# 6) Export nach Excel
# ------------------------------------------------------------

with pd.ExcelWriter(OUT_FILE) as writer:
    res_sorted.to_excel(writer, sheet_name="item_tests_sorted", index=False)
    res.to_excel(writer, sheet_name="item_tests_raw", index=False)

print("Export:", OUT_FILE)
print("DONE.")