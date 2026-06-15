# 16_balance_and_demographics_explorative.py

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# =========================================================
# 0) PROJEKT-ROOT AUTOMATISCH FINDEN
# (Damit ist es egal, ob du das Script aus /python startest.)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# 1) DATENPFAD FESTLEGEN
# -> Standard: ../outputs/clean_data.csv
# Wenn deine Datei anders heißt: NUR diese Zeile anpassen.
# =========================================================
DATA_PATH = PROJECT_ROOT / "outputs" / "clean_data.csv"

# =========================================================
# 2) SPALTENNAMEN (ggf. anpassen)
# =========================================================
COND_COL   = "version"      # enthält "A" und "B"
GENDER_COL = "gender"
AGE_COL    = "age"
EMP_COL    = "employment"

# KI-Affinität (falls vorhanden) – wenn du KEINE solche Spalte hast, lass es so;
# das Script erkennt automatisch, ob die Spalte existiert.
AI_AFF_COL = "ai_tools_use"

# Outcomes für optionalen Subgruppen-KI-Effekt
OUTCOMES = {
    "Vertrauen": "trust_idx",
    "Loyalität": "loyal_idx",
}

# =========================================================
# 3) HELPER
# =========================================================
def chi2_or_fisher(table: pd.DataFrame):
    """Chi2, bei 2x2 + erwartete Häufigkeit <5 -> Fisher."""
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    if table.shape == (2, 2) and exp.min() < 5:
        _, p_f = stats.fisher_exact(table)
        return "Fisher", float(p_f), np.nan, np.nan
    return "Chi2", float(p), float(chi2), float(dof)

def holm_adjust(pvals):
    """Holm-Bonferroni adjust, returns adjusted p-values in original order."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    for k, idx in enumerate(order):
        adj[idx] = min((n - k) * pvals[idx], 1.0)
    for k in range(1, n):
        prev = order[k - 1]
        curr = order[k]
        adj[curr] = max(adj[curr], adj[prev])
    return adj

def cliffs_delta(x, y):
    """Cliff's delta (A vs B). Positive -> A tends to be higher than B."""
    x = np.asarray(pd.Series(x).dropna())
    y = np.asarray(pd.Series(y).dropna())
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))

# =========================================================
# 4) DATEN LADEN
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"❌ Datenfile nicht gefunden: {DATA_PATH}\n"
        f"➡️ Check: Gibt es in {PROJECT_ROOT / 'outputs'} eine clean_data.csv?\n"
        f"Wenn nicht, ändere DATA_PATH oben auf den richtigen Dateinamen."
    )

if DATA_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_PATH)
elif DATA_PATH.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(DATA_PATH)
else:
    raise ValueError(f"❌ Unbekanntes Dateiformat: {DATA_PATH.suffix}")

print("✅ Loaded:", DATA_PATH)
print("✅ Shape:", df.shape)

# Nur A/B
if COND_COL not in df.columns:
    raise KeyError(f"❌ COND_COL '{COND_COL}' nicht gefunden. Verfügbare Spalten: {list(df.columns)}")

df = df[df[COND_COL].isin(["A", "B"])].copy()
print("✅ After A/B filter:", df.shape)
print("✅ A/B counts:\n", df[COND_COL].value_counts())

# =========================================================
# 5) A) BALANCECHECK A vs B (Demografie)
# =========================================================
balance_rows = []

# Geschlecht
if GENDER_COL in df.columns:
    tab = pd.crosstab(df[COND_COL], df[GENDER_COL])
    test, p, stat, dof = chi2_or_fisher(tab)
    balance_rows.append(["Geschlecht", test, p, stat, dof, tab.shape])
else:
    balance_rows.append(["Geschlecht", "NA", np.nan, np.nan, np.nan, "Spalte fehlt"])

# Erwerbsstatus
if EMP_COL in df.columns:
    tab = pd.crosstab(df[COND_COL], df[EMP_COL])
    test, p, stat, dof = chi2_or_fisher(tab)
    balance_rows.append(["Erwerbsstatus", test, p, stat, dof, tab.shape])
else:
    balance_rows.append(["Erwerbsstatus", "NA", np.nan, np.nan, np.nan, "Spalte fehlt"])

# Alter
if AGE_COL in df.columns:
    a = df.loc[df[COND_COL] == "A", AGE_COL].dropna()
    b = df.loc[df[COND_COL] == "B", AGE_COL].dropna()
    U, p_age = stats.mannwhitneyu(a, b, alternative="two-sided")
    balance_rows.append(["Alter", "Mann-Whitney-U", float(p_age), float(U), np.nan, (len(a), len(b))])
else:
    balance_rows.append(["Alter", "NA", np.nan, np.nan, np.nan, "Spalte fehlt"])

balance_df = pd.DataFrame(balance_rows, columns=["Variable", "Test", "p", "Statistik", "df", "Info"])
out1 = PROJECT_ROOT / "outputs" / "balance_demographics_AB.xlsx"
balance_df.to_excel(out1, index=False)
print("✅ Saved:", out1)

# =========================================================
# 6) B) EXPLORATIV: KI-AFFINITÄT nach Demografie
# =========================================================
expl_rows = []

if AI_AFF_COL in df.columns:
    # Geschlecht vs KI-Affinität
    if GENDER_COL in df.columns:
        tab = pd.crosstab(df[GENDER_COL], df[AI_AFF_COL])
        test, p, stat, dof = chi2_or_fisher(tab)
        expl_rows.append(["Geschlecht", AI_AFF_COL, test, p, stat])

    # Erwerbsstatus vs KI-Affinität
    if EMP_COL in df.columns:
        tab = pd.crosstab(df[EMP_COL], df[AI_AFF_COL])
        test, p, stat, dof = chi2_or_fisher(tab)
        expl_rows.append(["Erwerbsstatus", AI_AFF_COL, test, p, stat])

    # Alter vs KI-Affinität (Spearman – nur wenn AI_AFF_COL ordinal/numerisch ist)
    if AGE_COL in df.columns:
        try:
            rho, p = stats.spearmanr(df[AGE_COL], df[AI_AFF_COL], nan_policy="omit")
            expl_rows.append(["Alter", AI_AFF_COL, "Spearman", float(p), float(rho)])
        except Exception as e:
            expl_rows.append(["Alter", AI_AFF_COL, "Spearman", np.nan, f"nicht berechenbar: {e}"])

expl_df = pd.DataFrame(expl_rows, columns=["Demografie", "Zielvariable", "Test", "p", "Statistik"])
if len(expl_df) > 0:
    expl_df["p_holm"] = holm_adjust(expl_df["p"].fillna(1.0).values)
    out2 = PROJECT_ROOT / "outputs" / "explorativ_ai_affinity_by_demographics.xlsx"
    expl_df.to_excel(out2, index=False)
    print("✅ Saved:", out2)
else:
    print("ℹ️ Keine KI-Affinitäts-Auswertung erzeugt (AI_AFF_COL nicht vorhanden oder keine Daten).")

# =========================================================
# 7) C) OPTIONAL: KI-EFFEKT INNERHALB GESCHLECHT (nur Trust/Loyalty)
# =========================================================
sub_rows = []
if GENDER_COL in df.columns:
    for outcome_name, outcome_col in OUTCOMES.items():
        if outcome_col not in df.columns:
            continue
        for g in df[GENDER_COL].dropna().unique():
            sub = df[df[GENDER_COL] == g]
            xA = sub.loc[sub[COND_COL] == "A", outcome_col]
            xB = sub.loc[sub[COND_COL] == "B", outcome_col]

            nA = len(xA.dropna()); nB = len(xB.dropna())
            if nA < 10 or nB < 10:
                sub_rows.append([outcome_name, g, nA, nB, np.nan, np.nan, np.nan, "zu kleine Zellgröße"])
                continue

            U, p = stats.mannwhitneyu(xA.dropna(), xB.dropna(), alternative="two-sided")
            d = cliffs_delta(xA, xB)
            sub_rows.append([outcome_name, g, nA, nB, float(U), float(p), float(d), "ok"])

sub_df = pd.DataFrame(sub_rows, columns=["Outcome", "Geschlecht", "n_A", "n_B", "U", "p", "delta", "Hinweis"])
if len(sub_df) > 0:
    sub_df["p_holm"] = holm_adjust(sub_df["p"].fillna(1.0).values)
    out3 = PROJECT_ROOT / "outputs" / "explorativ_KI_effect_by_gender.xlsx"
    sub_df.to_excel(out3, index=False)
    print("✅ Saved:", out3)
else:
    print("ℹ️ Keine Subgruppen-Outputs erzeugt (z.B. fehlende Spalten oder zu kleine Gruppen).")

print("\n✅ DONE. Check outputs/:")
print("- balance_demographics_AB.xlsx")
print("- explorativ_ai_affinity_by_demographics.xlsx (wenn AI_AFF_COL existiert)")
print("- explorativ_KI_effect_by_gender.xlsx (optional)")