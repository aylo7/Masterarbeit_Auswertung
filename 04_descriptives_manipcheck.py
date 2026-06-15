# ============================================================
# 04_descriptives_manipcheck.py
# Zweck:
# - survey_scales.xlsx laden (enthält bereits Skalenwerte)
# - Deskriptive Tabellen erstellen:
#   (a) Skalen: gesamt + nach Version A/B
#   (b) Demografie/Usage: Häufigkeiten + Prozent (gesamt + nach Version)
# - Manipulationscheck:
#   Mann-Whitney-U + Effektgröße Cliff's delta
# - Ergebnisse als Excel in outputs/ speichern
# ============================================================

# pandas = Tabellen/Excel verarbeiten
import pandas as pd

# numpy = numerische Hilfsfunktionen
import numpy as np

# pathlib = saubere Dateipfade
from pathlib import Path

# scipy = enthält Mann-Whitney-U Test (nichtparametrisch)
from scipy.stats import mannwhitneyu


# ------------------------------------------------------------
# 1) Pfade definieren (Projekt-Root automatisch finden)
# ------------------------------------------------------------

# BASE_DIR = Repo-Hauptordner (Masterarbeit_Auswertung)
BASE_DIR = Path(__file__).resolve().parents[1]

# Input: Datensatz mit Skalen (aus Schritt 3)
SCALES_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"

# Output-Ordner
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # falls nicht vorhanden -> anlegen

# Output-Dateien
OUT_SCALES_DESC = OUT_DIR / "descriptives_scales.xlsx"
OUT_DEMO_DESC = OUT_DIR / "descriptives_demographics.xlsx"
OUT_MANIP_TESTS = OUT_DIR / "manipulation_check_tests.xlsx"

# Sicherheitscheck: existiert survey_scales.xlsx?
if not SCALES_FILE.exists():
    raise FileNotFoundError(
        f"Nicht gefunden: {SCALES_FILE}\n"
        "Bitte zuerst Schritt 3 ausführen: python/03_scales_reliability.py"
    )


# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

# df = Haupt-DataFrame mit allen Variablen + Skalen
df = pd.read_excel(SCALES_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()


# ------------------------------------------------------------
# 3) Hilfsfunktionen
# ------------------------------------------------------------

def iqr(series):
    """
    IQR = Interquartilsabstand = Q3 - Q1
    robustes Streuungsmaß (ordinal-freundlich)
    """
    return series.quantile(0.75) - series.quantile(0.25)

def descriptives_for_cols(data, cols, group_col=None):
    """
    Erstellt Deskriptivstatistik für mehrere Spalten.
    Kennzahlen:
      n, median, iqr, mean, sd, min, max

    Wenn group_col angegeben:
      Ausgabe zusätzlich je Gruppe (z.B. Version A/B)
    """
    def summarize(df_part):
        # Wir erzeugen eine Tabelle, Zeilen = Variablen
        return pd.DataFrame({
            "n": df_part[cols].count(),
            "median": df_part[cols].median(),
            "iqr": df_part[cols].apply(iqr),
            "mean": df_part[cols].mean(),
            "sd": df_part[cols].std(ddof=1),
            "min": df_part[cols].min(),
            "max": df_part[cols].max(),
        })

    if group_col is None:
        out = summarize(data)
        out.index.name = "variable"
        return out.reset_index()

    frames = []
    for g, gdf in data.groupby(group_col):
        tmp = summarize(gdf)
        tmp["group"] = g
        tmp.index.name = "variable"
        frames.append(tmp.reset_index())
    return pd.concat(frames, ignore_index=True)

def freq_pct(data, col, group_col=None):
    """
    Häufigkeiten + Prozent für kategoriale Variablen.
    Wenn group_col gesetzt:
      Prozent je Gruppe (A/B)
    """
    if group_col is None:
        counts = data[col].value_counts(dropna=False)
        pct = (counts / len(data) * 100).round(1)
        out = pd.DataFrame({"value": counts.index.astype(str), "n": counts.values, "pct": pct.values})
        out.insert(0, "variable", col)
        return out

    frames = []
    for g, gdf in data.groupby(group_col):
        counts = gdf[col].value_counts(dropna=False)
        pct = (counts / len(gdf) * 100).round(1)
        tmp = pd.DataFrame({"value": counts.index.astype(str), "n": counts.values, "pct": pct.values})
        tmp.insert(0, "variable", col)
        tmp.insert(1, "group", g)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)

def cliffs_delta(a, b):
    """
    Cliff's delta Effektgröße (ordinal/nichtparametrisch geeignet).
    delta = (2U)/(n1*n2) - 1   (U bezieht sich auf sample a)
    Wertebereich: -1 bis +1
      >0  => a tendenziell höhere Werte als b
      <0  => a tendenziell niedrigere Werte als b
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan

    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    delta = (2 * U) / (n1 * n2) - 1
    return float(delta)

def mannwhitney_with_effect(df, col, group_col="version", groupA="A", groupB="B"):
    """
    Mann-Whitney-U für 2 Gruppen + Cliff's delta.
    """
    x = df[df[group_col] == groupA][col].dropna()
    y = df[df[group_col] == groupB][col].dropna()

    U, p = mannwhitneyu(x, y, alternative="two-sided")
    delta = cliffs_delta(x, y)

    return {
        "variable": col,
        "groupA": groupA,
        "nA": len(x),
        "groupB": groupB,
        "nB": len(y),
        "U": float(U),
        "p_two_sided": float(p),
        "cliffs_delta(A_vs_B)": float(delta),
        # zur Interpretation hilfreich:
        "median_A": float(x.median()) if len(x) else np.nan,
        "median_B": float(y.median()) if len(y) else np.nan,
    }


# ------------------------------------------------------------
# 4) Skalen-Deskriptiven (gesamt + nach Version)
# ------------------------------------------------------------

# Primär: Median-Skalen (ordinal-freundlich)
scale_cols_median = [
    "cx_median",
    "personalization_median",
    "trust_median",
    "transparency_median",
    "satisfaction_median",
    "loyalty_median",
]

# Sekundär: Mean-Skalen (optional)
scale_cols_mean = [
    "cx_mean",
    "personalization_mean",
    "trust_mean",
    "transparency_mean",
    "satisfaction_mean",
    "loyalty_mean",
]

desc_scales_total_median = descriptives_for_cols(df, scale_cols_median)
desc_scales_by_version_median = descriptives_for_cols(df, scale_cols_median, group_col="version")

desc_scales_total_mean = descriptives_for_cols(df, scale_cols_mean)
desc_scales_by_version_mean = descriptives_for_cols(df, scale_cols_mean, group_col="version")

print("===== SKALEN DESKRIPTIV (Median) - gesamt =====")
print(desc_scales_total_median)
print()


# ------------------------------------------------------------
# 5) Demografie / Usage: Häufigkeiten + Prozent
# ------------------------------------------------------------

demo_cols = [
    "gender",
    "age_group",
    "education",
    "employment",
    "dhp_use_12m",
    "dhp_freq",
    "ai_tools_use_12m",
]

demo_total_frames = []
demo_by_version_frames = []

for c in demo_cols:
    # Nur auswerten, wenn Spalte existiert
    if c in df.columns:
        demo_total_frames.append(freq_pct(df, c))
        demo_by_version_frames.append(freq_pct(df, c, group_col="version"))

demo_total = pd.concat(demo_total_frames, ignore_index=True) if len(demo_total_frames) else pd.DataFrame()
demo_by_version = pd.concat(demo_by_version_frames, ignore_index=True) if len(demo_by_version_frames) else pd.DataFrame()


# ------------------------------------------------------------
# 6) Manipulationscheck (A vs B)
# ------------------------------------------------------------

# Diese beiden Items sollen prüfen, ob A wirklich als "KI/Assistenz" wahrgenommen wurde
manip_cols = ["manip_assist_functions", "manip_ai"]

manip_results_list = []
for c in manip_cols:
    if c in df.columns:
        manip_results_list.append(mannwhitney_with_effect(df, c, group_col="version", groupA="A", groupB="B"))

manip_results = pd.DataFrame(manip_results_list)

print("===== MANIPULATION CHECK (Mann–Whitney + Cliff’s delta) =====")
print(manip_results)
print()


# ------------------------------------------------------------
# 7) Export in Excel (mehrere Sheets)
# ------------------------------------------------------------

# Skalen-Deskriptiven als Excel mit mehreren Tabellenblättern
with pd.ExcelWriter(OUT_SCALES_DESC) as writer:
    desc_scales_total_median.to_excel(writer, sheet_name="total_median", index=False)
    desc_scales_by_version_median.to_excel(writer, sheet_name="by_version_median", index=False)
    desc_scales_total_mean.to_excel(writer, sheet_name="total_mean", index=False)
    desc_scales_by_version_mean.to_excel(writer, sheet_name="by_version_mean", index=False)

# Demografie-Deskriptiven als Excel
with pd.ExcelWriter(OUT_DEMO_DESC) as writer:
    demo_total.to_excel(writer, sheet_name="total", index=False)
    demo_by_version.to_excel(writer, sheet_name="by_version", index=False)

# Manipulationscheck-Resultate separat
manip_results.to_excel(OUT_MANIP_TESTS, index=False)

print("===== EXPORT FERTIG =====")
print("Skalen-Deskriptiven:", OUT_SCALES_DESC)
print("Demografie-Deskriptiven:", OUT_DEMO_DESC)
print("Manipulationscheck-Tests:", OUT_MANIP_TESTS)
print("\nDONE.")