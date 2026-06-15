import pandas as pd
from pathlib import Path

# --- Pfad robust finden (egal ob du aus Root oder /python startest) ---
p1 = Path("../data/processed/survey_scales.xlsx")   # wenn du im python-Ordner bist
p2 = Path("data/processed/survey_scales.xlsx")      # wenn du im Root bist

data_path = p1 if p1.exists() else p2
if not data_path.exists():
    raise FileNotFoundError(f"survey_scales.xlsx nicht gefunden. Geprüft: {p1} und {p2}")

df = pd.read_excel(data_path)
N = len(df)

print("N =", N)
print("Datei:", data_path.resolve())

def dist(col):
    s = df[col].copy()
    s = s.where(~s.isna(), "Missing")
    vc = s.value_counts(dropna=False)
    out = vc.reset_index()
    out.columns = ["Kategorie", "n"]
    out["%"] = (out["n"] / N * 100).round(1)
    return out

cols = [
    ("gender", "Geschlecht"),
    ("age_group", "Alter"),
    ("education", "Bildungsabschluss"),
    ("employment", "Erwerbsstatus"),
    ("dhp_use_12m", "Nutzung digitaler Gesundheitsplattformen (12M)"),
    ("ai_tools_use_12m", "Nutzung KI-Tools (12M)"),
    ("dhp_freq", "Nutzungshäufigkeit digitale Gesundheitsplattformen"),
]

# --- 1) Prozent-Tabellen ausgeben ---
for c, t in cols:
    print(f"\n=== {t} ({c}) ===")
    if c in df.columns:
        print(dist(c).to_string(index=False))
    else:
        print("FEHLT in Daten")

# --- 2) Mini-Textbausteine für 4.1 (automatisch aus Gender/DHP/KI) ---
def pct_of(col, label):
    tab = dist(col)
    row = tab[tab["Kategorie"].astype(str) == label]
    if len(row) == 0:
        return None
    return float(row["%"].iloc[0])

print("\n\n===== TEXTBAUSTEINE für 4.1 (copy/paste) =====")

# Gender (sofern Kategorien so heißen)
if "gender" in df.columns:
    tab_g = dist("gender")
    # Wir nehmen die Top 3 Kategorien + Missing, falls vorhanden
    top = tab_g.head(3)
    miss = tab_g[tab_g["Kategorie"] == "Missing"]
    lines = []
    for _, r in top.iterrows():
        lines.append(f"{r['Kategorie']} {r['%']}% (n={int(r['n'])})")
    if len(miss) > 0:
        r = miss.iloc[0]
        lines.append(f"Missing {r['%']}% (n={int(r['n'])})")
    print("Geschlecht:", "; ".join(lines) + ".")

# DHP use
if "dhp_use_12m" in df.columns:
    tab = dist("dhp_use_12m")
    print("DHP-Nutzung (12M):", "; ".join([f"{r['Kategorie']} {r['%']}% (n={int(r['n'])})" for _, r in tab.iterrows()]) + ".")

# KI Tools use
if "ai_tools_use_12m" in df.columns:
    tab = dist("ai_tools_use_12m")
    print("KI-Tools-Nutzung (12M):", "; ".join([f"{r['Kategorie']} {r['%']}% (n={int(r['n'])})" for _, r in tab.iterrows()]) + ".")