# ============================================================
# 13_scale_diagnostics.py
# Ziel:
# - survey_clean.xlsx + constructs.json laden
# - Skalen-Diagnostik (ordinal-freundlich):
#   * Cronbach's alpha pro Skala
#   * korrigierte Item-Total-Korrelation (CITC) via Spearman
#   * alpha-if-deleted pro Item
#   * Inter-Item-Korrelationen (Spearman) als Long-Format
# - Speziell prüfen:
#   * Korrelation zwischen cx_time_save_1 und cx_time_save_2
#   * mögliche Redundanz bei Loyalty (sehr hohe alpha)
# - Export: outputs/scale_diagnostics.xlsx
# ============================================================

import pandas as pd                    # Tabellen/Excel
import numpy as np                     # Numerik
import json                            # JSON lesen
from pathlib import Path               # Dateipfade
from scipy.stats import spearmanr      # Spearman-Korrelation (ordinal-passend)

# ------------------------------------------------------------
# 1) Pfade definieren
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # Projekt-Root

CLEAN_FILE = BASE_DIR / "data" / "processed" / "survey_clean.xlsx"  # bereinigte Daten
CONSTRUCTS_FILE = BASE_DIR / "outputs" / "constructs.json"          # Item-Gruppierung
OUT_FILE = BASE_DIR / "outputs" / "scale_diagnostics.xlsx"          # Output

# Sicherheitschecks: existieren die Dateien?
if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CLEAN_FILE} (Schritt 2 ausführen)")
if not CONSTRUCTS_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {CONSTRUCTS_FILE} (Schritt 2 ausführen)")

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(CLEAN_FILE)  # df = DataFrame mit kurzen Spaltennamen

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Constructs laden (Items pro Skala)
# ------------------------------------------------------------

with open(CONSTRUCTS_FILE, "r", encoding="utf-8") as f:
    constructs_raw = json.load(f)

def normalize_items(v):
    """
    constructs.json kann eine Liste oder ein dict enthalten.
    -> Wir geben immer eine Python-Liste zurück.
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

# Wir definieren die Skalen (ohne manipulation_check als Skalenkonstrukt)
scale_defs = {
    "cx": constructs["cx_usability_efficiency"],
    "personalization": constructs["personalization_support"],
    "trust": constructs["trust"],
    "transparency": constructs["transparency"],          # Single-Item
    "satisfaction": constructs["satisfaction"],
    "loyalty": constructs["loyalty"],
}

# ------------------------------------------------------------
# 4) Hilfsfunktionen: Cronbach's alpha + Diagnostik
# ------------------------------------------------------------

def cronbach_alpha(df_items):
    """
    Cronbach's alpha für ein Item-Set.
    df_items: DataFrame nur mit Item-Spalten (numerisch).
    """
    df_items = df_items.dropna()          # Missing entfernen
    k = df_items.shape[1]                 # Anzahl Items
    if k < 2:
        return np.nan

    item_vars = df_items.var(axis=0, ddof=1)       # Varianz je Item
    total_score = df_items.sum(axis=1)             # Summenscore
    total_var = total_score.var(ddof=1)            # Varianz Summenscore

    if total_var == 0:
        return np.nan

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)

def corrected_item_total_corr(df_items, item):
    """
    Korrigierte Item-Total-Korrelation:
    - Korrelieren item mit (Summe aller anderen Items)
    - Spearman (ordinal-freundlich)
    """
    x = df_items[item]
    total_excl = df_items.drop(columns=[item]).sum(axis=1)
    # Spearmanr liefert rho und p; wir brauchen rho
    rho, _ = spearmanr(x, total_excl, nan_policy="omit")
    return float(rho)

def mean_inter_item_corr(df_items):
    """
    Durchschnittliche Inter-Item-Korrelation (Spearman) über obere Dreiecksmatrix.
    Robust berechnet über numpy-Indizes.
    """
    corr = df_items.corr(method="spearman").to_numpy()
    k = corr.shape[0]
    if k < 2:
        return np.nan

    iu = np.triu_indices(k, k=1)  # obere Dreiecksmatrix ohne Diagonale
    vals = corr[iu]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))

# ------------------------------------------------------------
# 5) Haupt-Diagnostik pro Skala
# ------------------------------------------------------------

summary_rows = []       # pro Skala eine Zeile
item_rows = []          # Item-level Diagnostik (long)
interitem_rows = []     # Inter-item correlations (long)

for scale_name, items in scale_defs.items():
    # Nur Items verwenden, die im df existieren
    items = [c for c in items if c in df.columns]

    # Items in numerisch umwandeln (falls als Text importiert)
    df_items = df[items].apply(pd.to_numeric, errors="coerce")

    k = len(items)

    # Alpha (nur bei k>=2)
    alpha = cronbach_alpha(df_items) if k >= 2 else np.nan

    # Mittlere Inter-Item-Korrelation (nur bei k>=2)
    miic = mean_inter_item_corr(df_items) if k >= 2 else np.nan

    # Summary-Zeile
    summary_rows.append({
        "scale": scale_name,
        "n_items": k,
        "cronbach_alpha": alpha,
        "mean_inter_item_corr_spearman": miic,
        "note": "Single-Item" if k < 2 else ""
    })

    # Item-level Diagnostik
    for item in items:
        # CITC nur bei Skalen mit >=2 Items sinnvoll
        citc = corrected_item_total_corr(df_items, item) if k >= 2 else np.nan

        # Alpha if deleted: Alpha neu berechnen, wenn item entfernt wird
        if k >= 3:
            alpha_del = cronbach_alpha(df_items.drop(columns=[item]))
        else:
            # bei 2 Items wäre alpha_if_deleted nicht sinnvoll als Diagnose
            alpha_del = np.nan

        item_rows.append({
            "scale": scale_name,
            "item": item,
            "item_mean": float(df_items[item].mean()),
            "item_sd": float(df_items[item].std(ddof=1)),
            "citc_spearman": citc,
            "alpha_if_deleted": alpha_del
        })

    # Inter-Item-Korrelationen (long)
    if k >= 2:
        corr = df_items.corr(method="spearman")
        for i in range(k):
            for j in range(i + 1, k):
                interitem_rows.append({
                    "scale": scale_name,
                    "item1": items[i],
                    "item2": items[j],
                    "spearman_rho": float(corr.loc[items[i], items[j]])
                })

# DataFrames bauen
summary_df = pd.DataFrame(summary_rows)
item_diag_df = pd.DataFrame(item_rows)
interitem_df = pd.DataFrame(interitem_rows)

print("===== SKALEN-SUMMARY =====")
print(summary_df)
print()

# ------------------------------------------------------------
# 6) Spezielle Checks: doppelte Zeit-Items + Loyalty-Redundanz
# ------------------------------------------------------------

# Zeit-Items Korrelation (falls vorhanden)
if ("cx_time_save_1" in df.columns) and ("cx_time_save_2" in df.columns):
    t1 = pd.to_numeric(df["cx_time_save_1"], errors="coerce")
    t2 = pd.to_numeric(df["cx_time_save_2"], errors="coerce")
    rho_time, p_time = spearmanr(t1, t2, nan_policy="omit")
    time_check = pd.DataFrame([{
        "pair": "cx_time_save_1 vs cx_time_save_2",
        "spearman_rho": float(rho_time),
        "p_value": float(p_time)
    }])
else:
    time_check = pd.DataFrame([{
        "pair": "cx_time_save_1 vs cx_time_save_2",
        "spearman_rho": np.nan,
        "p_value": np.nan
    }])

print("===== TIME-ITEM CHECK =====")
print(time_check)
print()

# ------------------------------------------------------------
# 7) Export nach Excel
# ------------------------------------------------------------

with pd.ExcelWriter(OUT_FILE) as writer:
    summary_df.to_excel(writer, sheet_name="scale_summary", index=False)
    item_diag_df.to_excel(writer, sheet_name="item_diagnostics", index=False)
    interitem_df.to_excel(writer, sheet_name="interitem_corr_long", index=False)
    time_check.to_excel(writer, sheet_name="time_item_check", index=False)

print("Export:", OUT_FILE)
print("DONE.")