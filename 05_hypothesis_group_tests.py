# 16_balance_and_demographics_explorative.py
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Welche Spalten MUSS der Datensatz haben? ---
COND_COL   = "version"      # A/B
GENDER_COL = "gender"
AGE_COL    = "age"
EMP_COL    = "employment"

# Optional: KI-Affinität (wenn es die Spalte gibt, wird sie ausgewertet)
AI_AFF_COL = "ai_tools_use"

# Optional: Outcomes für Subgruppen (wenn vorhanden)
OUTCOMES = {
    "Vertrauen": "trust_idx",
    "Loyalität": "loyal_idx",
}

REQUIRED_COLS = {COND_COL, GENDER_COL, AGE_COL, EMP_COL}

def chi2_or_fisher(table: pd.DataFrame):
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    if table.shape == (2, 2) and exp.min() < 5:
        _, p_f = stats.fisher_exact(table)
        return "Fisher", float(p_f), np.nan, np.nan
    return "Chi2", float(p), float(chi2), float(dof)

def holm_adjust(pvals):
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
    x = np.asarray(pd.Series(x).dropna())
    y = np.asarray(pd.Series(y).dropna())
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))

def try_load_file(fp: Path):
    try:
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        elif fp.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(fp)
        else:
            return None
        # Muss version-Spalte haben und A/B enthalten
        if COND_COL not in df.columns:
            return None
        if not set(df[COND_COL].dropna().unique()).intersection({"A", "B"}):
            return None
        # Muss die wichtigen Demografie-Spalten haben
        if not REQUIRED_COLS.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None

# --- 1) AUTOMATISCH DATENSATZ FINDEN ---
candidates = []
for ext in ("*.csv", "*.xlsx", "*.xls"):
    candidates += list(PROJECT_ROOT.rglob(ext))

# Priorisiere typische Ordner
def priority(p: Path):
    s = str(p).lower()
    score = 0
    if "processed" in s: score -= 3
    if "data" in s: score -= 2
    if "clean" in s: score -= 2
    if "raw" in s: score += 2
    if "outputs" in s: score += 1
    return score

candidates = sorted(candidates, key=priority)

df = None
used_path = None
checked = 0

for fp in candidates:
    checked += 1
    loaded = try_load_file(fp)
    if loaded is not None:
        df = loaded
        used_path = fp
        break

if df is None:
    raise FileNotFoundError(
        "❌ Ich finde keinen Datensatz mit den nötigen Spalten "
        f"{sorted(REQUIRED_COLS)} und version=A/B.\n"
        "➡️ Bitte schick mir einen Screenshot von df.columns oder sag mir, wie deine Spalten heißen.\n"
    )

print("✅ Datensatz gefunden:", used_path)
print("✅ Shape:", df.shape)
print("✅ A/B counts:\n", df[COND_COL].value_counts())

# Nur A/B
df = df[df[COND_COL].isin(["A", "B"])].copy()

# --- 2) A) BALANCECHECK A vs B ---
balance_rows = []

# Geschlecht
tab = pd.crosstab(df[COND_COL], df[GENDER_COL])
test, p, stat, dof = chi2_or_fisher(tab)
balance_rows.append(["Geschlecht", test, p, stat, dof])

# Erwerbsstatus
tab = pd.crosstab(df[COND_COL], df[EMP_COL])
test, p, stat, dof = chi2_or_fisher(tab)
balance_rows.append(["Erwerbsstatus", test, p, stat, dof])

# Alter
a = df.loc[df[COND_COL] == "A", AGE_COL].dropna()
b = df.loc[df[COND_COL] == "B", AGE_COL].dropna()
U, p_age = stats.mannwhitneyu(a, b, alternative="two-sided")
balance_rows.append(["Alter", "Mann-Whitney-U", float(p_age), float(U), np.nan])

balance_df = pd.DataFrame(balance_rows, columns=["Variable", "Test", "p", "Statistik", "df"])
out1 = OUTPUT_DIR / "balance_demographics_AB.xlsx"
balance_df.to_excel(out1, index=False)
print("✅ Saved:", out1)

# --- 3) B) EXPLORATIV: KI-AFFINITÄT nach Demografie (optional) ---
expl_rows = []
if AI_AFF_COL in df.columns:
    # Geschlecht vs KI-Affinität
    tab = pd.crosstab(df[GENDER_COL], df[AI_AFF_COL])
    test, p, stat, dof = chi2_or_fisher(tab)
    expl_rows.append(["Geschlecht", AI_AFF_COL, test, p, stat])

    # Erwerbsstatus vs KI-Affinität
    tab = pd.crosstab(df[EMP_COL], df[AI_AFF_COL])
    test, p, stat, dof = chi2_or_fisher(tab)
    expl_rows.append(["Erwerbsstatus", AI_AFF_COL, test, p, stat])

    # Alter vs KI-Affinität (Spearman – nur sinnvoll, wenn AI_AFF_COL ordinal/numerisch ist)
    try:
        rho, p = stats.spearmanr(df[AGE_COL], df[AI_AFF_COL], nan_policy="omit")
        expl_rows.append(["Alter", AI_AFF_COL, "Spearman", float(p), float(rho)])
    except Exception as e:
        expl_rows.append(["Alter", AI_AFF_COL, "Spearman", np.nan, f"nicht berechenbar: {e}"])

expl_df = pd.DataFrame(expl_rows, columns=["Demografie", "Zielvariable", "Test", "p", "Statistik"])
if len(expl_df) > 0:
    expl_df["p_holm"] = holm_adjust(expl_df["p"].fillna(1.0).values)
    out2 = OUTPUT_DIR / "explorativ_ai_affinity_by_demographics.xlsx"
    expl_df.to_excel(out2, index=False)
    print("✅ Saved:", out2)
else:
    print("ℹ️ Keine KI-Affinitäts-Auswertung (AI_AFF_COL fehlt oder keine Daten).")

# --- 4) C) OPTIONAL: KI-EFFEKT innerhalb Geschlecht (Trust/Loyalty) ---
sub_rows = []
for outcome_name, outcome_col in OUTCOMES.items():
    if outcome_col not in df.columns:
        continue
    for g in df[GENDER_COL].dropna().unique():
        sub = df[df[GENDER_COL] == g]
        xA = sub.loc[sub[COND_COL] == "A", outcome_col]
        xB = sub.loc[sub[COND_COL] == "B", outcome_col]
        nA = len(xA.dropna()); nB = len(xB.dropna())

        if nA < 10 or nB < 10:
            sub_rows.append([outcome_name, g, nA, nB, np.nan, np.nan, np.nan, "zu klein"])
            continue

        U, p = stats.mannwhitneyu(xA.dropna(), xB.dropna(), alternative="two-sided")
        d = cliffs_delta(xA, xB)
        sub_rows.append([outcome_name, g, nA, nB, float(U), float(p), float(d), "ok"])

sub_df = pd.DataFrame(sub_rows, columns=["Outcome", "Geschlecht", "n_A", "n_B", "U", "p", "delta", "Hinweis"])
if len(sub_df) > 0:
    sub_df["p_holm"] = holm_adjust(sub_df["p"].fillna(1.0).values)
    out3 = OUTPUT_DIR / "explorativ_KI_effect_by_gender.xlsx"
    sub_df.to_excel(out3, index=False)
    print("✅ Saved:", out3)
else:
    print("ℹ️ Keine Subgruppen-Outputs erzeugt (z.B. fehlende Spalten oder zu kleine Gruppen).")

print("\n✅ DONE. In outputs/ liegen jetzt:")
print("- balance_demographics_AB.xlsx")
print("- explorativ_ai_affinity_by_demographics.xlsx (falls ai_tools_use existiert)")
print("- explorativ_KI_effect_by_gender.xlsx (optional)")