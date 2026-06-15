# ============================================================
# 10_moderation_demographics_bootstrap.py
# Ziel:
# - Moderation (explorativ): Unterscheidet sich der KI-Effekt (A vs B)
#   je Demografie-/Nutzungsgruppe?
#
# Vorgehen (ordinal-freundlich):
# - Innerhalb jeder Gruppe: Cliff's delta für A vs B auf Outcome (z.B. loyalty_median)
# - Für binäre Moderatoren: delta_diff = delta(level1) - delta(level2)
# - Bootstrap-CI für delta_diff (Resampling innerhalb jeder Level×Version-Zelle)
# - Holm-Korrektur für multiple delta_diff-Tests pro Moderator/Outcome
#
# WICHTIG:
# - Explorativ, daher Ergebnisse defensiv interpretieren.
# - Altersgruppen werden zu 45+ zusammengefasst.
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
OUT_FILE = BASE_DIR / "outputs" / "moderation_demographics_bootstrap.xlsx"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {DATA_FILE}\nBitte zuerst Schritt 3 ausführen.")

# ------------------------------------------------------------
# 2) Einstellungen
# ------------------------------------------------------------

# Outcomes, die wir moderationsmäßig prüfen (hypothesennah)
OUTCOMES = ["cx_median", "trust_median", "satisfaction_median", "loyalty_median"]

# Mindestgröße pro Zelle (Level×Version), sonst wird der Level übersprungen
MIN_CELL = 10

# Bootstrap-Iterationen (2000 schneller; 5000 stabiler)
N_BOOT = 5000

# Zufallsseed für Reproduzierbarkeit
RNG = np.random.default_rng(42)

# ------------------------------------------------------------
# 3) Daten laden
# ------------------------------------------------------------

df = pd.read_excel(DATA_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 4) Re-Coding: Altersgruppen zu 45+ zusammenfassen
# ------------------------------------------------------------

if "age_group" in df.columns:
    def recode_age(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x in ["45-54", "55-64", "65+"]:
            return "45+"
        # die anderen lassen wir wie sie sind
        if x in ["18-24", "25-34", "35-44"]:
            return x
        return x  # falls unerwartete Kategorie
    df["age_group_collapsed"] = df["age_group"].apply(recode_age)
else:
    df["age_group_collapsed"] = np.nan

# ------------------------------------------------------------
# 5) Re-Coding: binäre Nutzungsvariablen (unsure wird entfernt)
# ------------------------------------------------------------

def make_binary_yes_no(series):
    """
    Mappt auf 'Ja'/'Nein', alles andere (z.B. unsure) -> NaN
    """
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan})
    s = s.where(s.isin(["Ja", "Nein"]), np.nan)
    return s

if "dhp_use_12m" in df.columns:
    df["dhp_use_12m_bin"] = make_binary_yes_no(df["dhp_use_12m"])
else:
    df["dhp_use_12m_bin"] = np.nan

if "ai_tools_use_12m" in df.columns:
    df["ai_tools_use_12m_bin"] = make_binary_yes_no(df["ai_tools_use_12m"])
else:
    df["ai_tools_use_12m_bin"] = np.nan

# ------------------------------------------------------------
# 6) Erwerbsstatus optional zusammenfassen (damit Zellen nicht zu klein sind)
# ------------------------------------------------------------

if "employment" in df.columns:
    def recode_employment(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x == "Angestellt":
            return "Angestellt"
        if x == "Selbstständig":
            return "Selbstständig"
        if x == "Student:in / in Ausbildung":
            return "Student/Ausbildung"
        # alles andere bündeln
        return "Other"
    df["employment_collapsed"] = df["employment"].apply(recode_employment)
else:
    df["employment_collapsed"] = np.nan

# ------------------------------------------------------------
# 7) Moderatoren definieren
#    (binäre Moderatoren sind am “saubersten” für Δδ)
# ------------------------------------------------------------

MODERATORS = {
    "gender": "gender",
    "age_group_45plus": "age_group_collapsed",
    "employment_collapsed": "employment_collapsed",
    "dhp_use_12m_bin": "dhp_use_12m_bin",
    "ai_tools_use_12m_bin": "ai_tools_use_12m_bin",
}

# ------------------------------------------------------------
# 8) Hilfsfunktionen
# ------------------------------------------------------------

def cliffs_delta(a, b):
    """
    Cliff's delta über U:
      delta = (2U)/(n1*n2) - 1
    """
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

def delta_A_vs_B_in_subset(sub_df, outcome):
    """
    Cliff's delta für A vs B innerhalb eines Subsets (z.B. nur Frauen)
    """
    a = sub_df[sub_df["version"] == "A"][outcome]
    b = sub_df[sub_df["version"] == "B"][outcome]
    return cliffs_delta(a, b)

def bootstrap_delta_diff(sub_df, moderator_col, level1, level2, outcome):
    """
    Bootstrap für delta_diff = delta(level1) - delta(level2)
    Resampling getrennt innerhalb jeder Level×Version-Zelle.
    """
    boot = np.zeros(N_BOOT)

    # Daten je Level
    d1 = sub_df[sub_df[moderator_col] == level1].copy()
    d2 = sub_df[sub_df[moderator_col] == level2].copy()

    # Arrays je Zelle vorbereiten (damit schnell)
    def cell_arrays(d, ver):
        return d[d["version"] == ver][outcome].dropna().to_numpy()

    a1 = cell_arrays(d1, "A"); b1 = cell_arrays(d1, "B")
    a2 = cell_arrays(d2, "A"); b2 = cell_arrays(d2, "B")

    # Sicherheitscheck
    if min(len(a1), len(b1), len(a2), len(b2)) < MIN_CELL:
        return None  # nicht bootstrappen

    for i in range(N_BOOT):
        a1_bs = RNG.choice(a1, size=len(a1), replace=True)
        b1_bs = RNG.choice(b1, size=len(b1), replace=True)
        a2_bs = RNG.choice(a2, size=len(a2), replace=True)
        b2_bs = RNG.choice(b2, size=len(b2), replace=True)

        # delta je Level
        U1, _ = mannwhitneyu(a1_bs, b1_bs, alternative="two-sided")
        d1_delta = (2 * U1) / (len(a1_bs) * len(b1_bs)) - 1

        U2, _ = mannwhitneyu(a2_bs, b2_bs, alternative="two-sided")
        d2_delta = (2 * U2) / (len(a2_bs) * len(b2_bs)) - 1

        boot[i] = d1_delta - d2_delta

    return boot

# ------------------------------------------------------------
# 9) Hauptanalyse: deltas pro Level + Bootstrap-Δδ (pairwise)
# ------------------------------------------------------------

delta_rows = []
diff_rows = []

# Wir arbeiten nur mit Zeilen, die Version & Outcome nicht fehlen
base = df.copy()

for mod_name, mod_col in MODERATORS.items():
    if mod_col not in base.columns:
        continue

    # Für jeden Outcome separat
    for outcome in OUTCOMES:
        if outcome not in base.columns:
            continue

        # Daten ohne Missing im Moderator/Outcome
        d = base[[mod_col, "version", outcome]].dropna().copy()

        # Levels prüfen
        levels = d[mod_col].unique().tolist()

        # Deltas je Level berechnen (nur wenn nA und nB >= MIN_CELL)
        valid_levels = []
        for lev in levels:
            sub = d[d[mod_col] == lev]
            nA = int((sub["version"] == "A").sum())
            nB = int((sub["version"] == "B").sum())
            if nA >= MIN_CELL and nB >= MIN_CELL:
                valid_levels.append(lev)
                delta = delta_A_vs_B_in_subset(sub, outcome)
                delta_rows.append({
                    "moderator": mod_name,
                    "moderator_col": mod_col,
                    "outcome": outcome,
                    "level": str(lev),
                    "nA": nA,
                    "nB": nB,
                    "cliffs_delta_A_vs_B": float(delta),
                })

        # Wenn weniger als 2 gültige Levels: keine Moderation testbar
        if len(valid_levels) < 2:
            continue

        # Pairwise Δδ zwischen Levels (Bootstrap)
        pvals = []
        temp_diffs = []

        for i in range(len(valid_levels)):
            for j in range(i + 1, len(valid_levels)):
                l1 = valid_levels[i]
                l2 = valid_levels[j]

                boot = bootstrap_delta_diff(d, mod_col, l1, l2, outcome)
                if boot is None:
                    continue

                delta_diff = float(np.mean(boot))  # Bootstrap-Mittel als Punkt-Schätzer
                ci_low = float(np.quantile(boot, 0.025))
                ci_high = float(np.quantile(boot, 0.975))

                # p_boot (approx): Anteil <=0 bzw >=0 (zweiseitig)
                p_boot = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
                p_boot = float(min(p_boot, 1.0))

                temp_diffs.append({
                    "moderator": mod_name,
                    "outcome": outcome,
                    "level1": str(l1),
                    "level2": str(l2),
                    "delta_diff(level1-level2)": delta_diff,
                    "boot_ci_low_2.5%": ci_low,
                    "boot_ci_high_97.5%": ci_high,
                    "p_boot": p_boot,
                    "n_boot": N_BOOT
                })
                pvals.append(p_boot)

        # Holm-Korrektur innerhalb (Moderator, Outcome) für die Pairwise-Tests
        if len(temp_diffs) > 0:
            p_holm = holm_correction([r["p_boot"] for r in temp_diffs])
            for r, ph in zip(temp_diffs, p_holm):
                r["p_holm"] = float(ph)
            diff_rows.extend(temp_diffs)

delta_df = pd.DataFrame(delta_rows)
diff_df = pd.DataFrame(diff_rows)

print("===== DELTAS (A vs B) je Moderator-Level =====")
print(delta_df.head(15))
print()

print("===== MODERATION Δδ (Bootstrap, pairwise) =====")
print(diff_df.head(15))
print()

# ------------------------------------------------------------
# 10) Export
# ------------------------------------------------------------

with pd.ExcelWriter(OUT_FILE) as writer:
    delta_df.to_excel(writer, sheet_name="deltas_by_level", index=False)
    diff_df.to_excel(writer, sheet_name="delta_diff_bootstrap", index=False)

print("Export:", OUT_FILE)
print("DONE.")