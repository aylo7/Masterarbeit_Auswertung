# ============================================================
# 07_mediation_bootstrap.py
# Ziel:
# - survey_scales.xlsx laden
# - Mediation (Sensitivitätsanalyse) auf Rangtransformierten Skalen:
#     X = condition_ai (0/1; A=1, B=0)
#     M = trust_median
#     Y = loyalty_median
# - Bootstrap-Confidence-Interval für indirekten Effekt (a*b)
# - Export nach outputs/
#
# WICHTIG:
# - Das ist eine Sensitivitätsanalyse (ordinal -> Rangtransform).
# - Hauptbefund bleibt: nichtparametrische Tests + Spearman-Korrelationen.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

# ------------------------------------------------------------
# 1) Pfade
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"

OUT_SUMMARY = BASE_DIR / "outputs" / "mediation_trust_loyalty_summary.xlsx"
OUT_BOOT = BASE_DIR / "outputs" / "mediation_trust_loyalty_bootstrap.xlsx"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {DATA_FILE}\nBitte zuerst Schritt 3 ausführen.")

# ------------------------------------------------------------
# 2) Daten laden + benötigte Spalten prüfen
# ------------------------------------------------------------

df = pd.read_excel(DATA_FILE)

needed = ["condition_ai", "trust_median", "loyalty_median"]
missing = [c for c in needed if c not in df.columns]
if len(missing) > 0:
    raise ValueError("Diese Spalten fehlen im Datensatz: " + ", ".join(missing))

# Wir wählen nur die relevanten Spalten und entfernen fehlende Werte
d = df[needed].dropna().copy()

print("===== DATEN (Mediation) =====")
print("N:", d.shape[0])
print("condition_ai counts:\n", d["condition_ai"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3) Rangtransform (ordinal-freundlicher)
# ------------------------------------------------------------

# X bleibt 0/1 (Gruppe)
X = d["condition_ai"].astype(float).to_numpy()

# M und Y werden in Ränge umgewandelt (1..N)
M = rankdata(d["trust_median"].to_numpy(), method="average")
Y = rankdata(d["loyalty_median"].to_numpy(), method="average")

# ------------------------------------------------------------
# 4) OLS-Helferfunktion (ohne statsmodels)
# ------------------------------------------------------------

def ols_beta(Xmat, y):
    """
    Einfache OLS-Schätzung: beta = (X'X)^(-1) X'y
    Xmat: Matrix (n x p), y: Vektor (n,)
    """
    XtX = Xmat.T @ Xmat
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xmat.T @ y)
    return beta

def add_intercept(x):
    """Fügt eine Intercept-Spalte (1en) hinzu."""
    return np.column_stack([np.ones(len(x)), x])

# ------------------------------------------------------------
# 5) Pfadeffekte berechnen (auf Rangdaten)
#    a: M ~ X
#    c: Y ~ X
#    b & c': Y ~ X + M
# ------------------------------------------------------------

# a-Pfad: M ~ X
Xa = add_intercept(X)
beta_a = ols_beta(Xa, M)
a = beta_a[1]  # Koeffizient von X

# c (total): Y ~ X
Xc = add_intercept(X)
beta_c = ols_beta(Xc, Y)
c_total = beta_c[1]

# b und c' (direct): Y ~ X + M
Xbc = np.column_stack([np.ones(len(X)), X, M])  # Intercept, X, M
beta_bc = ols_beta(Xbc, Y)
c_prime = beta_bc[1]  # Effekt X, kontrolliert M
b = beta_bc[2]        # Effekt M, kontrolliert X

indirect = a * b

print("===== PFADEFFEKTE (Rank-OLS) =====")
print("a (X->M):", a)
print("b (M->Y | X):", b)
print("c_total (X->Y):", c_total)
print("c_prime (X->Y | M):", c_prime)
print("indirect (a*b):", indirect)
print()

# ------------------------------------------------------------
# 6) Bootstrap für indirekten Effekt
# ------------------------------------------------------------

n_boot = 5000  # 2000 wäre schneller, 5000 stabiler
rng = np.random.default_rng(42)

boot_indirect = np.zeros(n_boot)

n = len(X)

for i in range(n_boot):
    idx = rng.integers(0, n, size=n)  # Resampling mit Zurücklegen

    Xb = X[idx]
    Mb = M[idx]
    Yb = Y[idx]

    # a: Mb ~ Xb
    beta_a_b = ols_beta(add_intercept(Xb), Mb)
    a_b = beta_a_b[1]

    # b: Yb ~ Xb + Mb
    Xbc_b = np.column_stack([np.ones(len(Xb)), Xb, Mb])
    beta_bc_b = ols_beta(Xbc_b, Yb)
    b_b = beta_bc_b[2]

    boot_indirect[i] = a_b * b_b

# Percentile CI
ci_low = float(np.quantile(boot_indirect, 0.025))
ci_high = float(np.quantile(boot_indirect, 0.975))

# "p-like": Anteil Bootstrap-Werte <=0 bzw >=0 (zweiseitig grob)
p_boot = 2 * min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0))
p_boot = float(min(p_boot, 1.0))

print("===== BOOTSTRAP INDIRECT EFFECT =====")
print("Indirect (a*b):", float(indirect))
print("95% CI:", (ci_low, ci_high))
print("p_boot (approx):", p_boot)
print()

# ------------------------------------------------------------
# 7) Export
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "n": int(n),
    "a_X_to_M": float(a),
    "b_M_to_Y_given_X": float(b),
    "c_total_X_to_Y": float(c_total),
    "c_prime_X_to_Y_given_M": float(c_prime),
    "indirect_a_times_b": float(indirect),
    "boot_ci_low_2.5%": ci_low,
    "boot_ci_high_97.5%": ci_high,
    "boot_p_approx": p_boot,
    "note": "Sensitivitätsanalyse: Rangtransform + OLS + Bootstrap"
}])

boot_df = pd.DataFrame({"boot_indirect": boot_indirect})

summary.to_excel(OUT_SUMMARY, index=False)
boot_df.to_excel(OUT_BOOT, index=False)

print("Export Summary:", OUT_SUMMARY)
print("Export Bootstrap Dist:", OUT_BOOT)
print("DONE.")