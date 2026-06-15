# ============================================================
# 03_scales_reliability.py
# Ziel:
# - survey_clean.xlsx laden
# - constructs.json laden (Item-Gruppierung)
# - Skalenwerte berechnen (Median + Mean)
# - Cronbach's Alpha für Multi-Item-Skalen berechnen
# - Ergebnisse exportieren
# ============================================================

import pandas as pd              # Tabellen/Excel verarbeiten
import numpy as np               # numerische Funktionen
import json                      # JSON-Dateien lesen
from pathlib import Path         # saubere Dateipfade

# ------------------------------------------------------------
# 1) Pfade definieren (Repo-Root automatisch)
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # Hauptordner deines Projekts

CLEAN_FILE = BASE_DIR / "data" / "processed" / "survey_clean.xlsx"   # Input: bereinigte Daten
CONSTRUCTS_FILE = BASE_DIR / "outputs" / "constructs.json"           # Input: Konstrukte/Items
OUT_SCALES_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"  # Output: Daten + Skalen
OUT_ALPHA_FILE = BASE_DIR / "outputs" / "reliability_alpha.xlsx"     # Output: α-Tabelle

# Sicherheitscheck: existieren die Dateien?
if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CLEAN_FILE}")
if not CONSTRUCTS_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CONSTRUCTS_FILE}")

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(CLEAN_FILE)  # df = DataFrame (Tabelle)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Constructs (Item-Listen) laden
#    (constructs.json kann als Liste oder als dict gespeichert sein)
# ------------------------------------------------------------

with open(CONSTRUCTS_FILE, "r", encoding="utf-8") as f:
    constructs_raw = json.load(f)

def normalize_items(v):
    """
    Sicherstellen, dass wir immer eine echte Python-Liste von Items bekommen.
    - Wenn v Liste ist -> direkt zurückgeben
    - Wenn v dict ist (z.B. {'0':'item1','1':'item2'}) -> sortieren und als Liste zurückgeben
    """
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

print("===== KONSTRUKTE (geladen) =====")
for con, items in constructs.items():
    print(con, "->", items)
print()

# ------------------------------------------------------------
# 4) Cronbach's Alpha Funktion (ohne Zusatzpakete)
# ------------------------------------------------------------

def cronbach_alpha(df_items):
    """
    Berechnet Cronbach's α.
    df_items = DataFrame mit den Item-Spalten (numerisch).
    """
    df_items = df_items.dropna()       # Zeilen mit Missing entfernen
    k = df_items.shape[1]              # Anzahl Items

    if k < 2:                          # Alpha nur bei >=2 Items sinnvoll
        return np.nan

    item_vars = df_items.var(axis=0, ddof=1)      # Varianz je Item
    total_score = df_items.sum(axis=1)            # Summenscore pro Person
    total_var = total_score.var(ddof=1)           # Varianz des Summenscores

    if total_var == 0:                            # Schutz gegen Division durch 0
        return np.nan

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)

# ------------------------------------------------------------
# 5) Skalen definieren (deine Masterarbeit-Konstrukte)
# ------------------------------------------------------------

scale_defs = {
    "cx": constructs["cx_usability_efficiency"],
    "personalization": constructs["personalization_support"],
    "trust": constructs["trust"],
    "transparency": constructs["transparency"],        # Single-Item
    "satisfaction": constructs["satisfaction"],
    "loyalty": constructs["loyalty"],
}

# ------------------------------------------------------------
# 6) Skalenwerte berechnen (Median + Mean)
#    Median = ordinal-freundlich
# ------------------------------------------------------------

for scale_name, items in scale_defs.items():
    # Median über Items pro Person
    df[f"{scale_name}_median"] = df[items].median(axis=1)

    # Mean über Items pro Person (optional)
    df[f"{scale_name}_mean"] = df[items].mean(axis=1)

print("===== SKALEN BERECHNET (Preview) =====")
print(df[[c for c in df.columns if c.endswith('_median')]].head())
print()

# ------------------------------------------------------------
# 7) Reliabilität je Skala (Cronbach's α)
# ------------------------------------------------------------

alpha_rows = []

for scale_name, items in scale_defs.items():
    k = len(items)

    if k < 2:
        alpha_rows.append({
            "scale": scale_name,
            "n_items": k,
            "cronbach_alpha": np.nan,
            "note": "Single-Item (kein α)"
        })
        continue

    items_df = df[items].apply(pd.to_numeric, errors="coerce")
    alpha_val = cronbach_alpha(items_df)

    alpha_rows.append({
        "scale": scale_name,
        "n_items": k,
        "cronbach_alpha": alpha_val,
        "note": ""
    })

alpha_table = pd.DataFrame(alpha_rows)

print("===== RELIABILITÄT (Cronbach's α) =====")
print(alpha_table)
print()

# ------------------------------------------------------------
# 8) Export
# ------------------------------------------------------------

df.to_excel(OUT_SCALES_FILE, index=False)
alpha_table.to_excel(OUT_ALPHA_FILE, index=False)

print("===== EXPORT FERTIG =====")
print("Scales dataset:", OUT_SCALES_FILE)
print("Alpha table:", OUT_ALPHA_FILE)
print("\nDONE.")