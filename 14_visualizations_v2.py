import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # wichtig in Codespaces/Headless

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "survey_scales.xlsx"
OUT_DIR = BASE_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Nicht gefunden: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

print("===== DATEN GELADEN =====")
print("ROWS,COLS:", df.shape)
print("Version counts:\n", df["version"].value_counts(dropna=False))
print("OUT_DIR:", OUT_DIR)
print()

def save_fig(fig, filename_base):
    png_path = OUT_DIR / f"{filename_base}.png"
    pdf_path = OUT_DIR / f"{filename_base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path.name, "and", pdf_path.name)

def percent_overall(col):
    # counts inkl. Missing
    counts = df[col].value_counts(dropna=False)

    # Index robust zu String machen und Missing sauber labeln
    idx = []
    for v in counts.index.tolist():
        if pd.isna(v):
            idx.append("Missing")
        else:
            idx.append(str(v))
    counts.index = idx

    pct = counts / len(df) * 100
    tab = pd.DataFrame({"pct": pct})
    return tab.sort_values("pct", ascending=False)

def percent_by_version(col):
    # crosstab inkl. Missing
    ct = pd.crosstab(df[col], df["version"], dropna=False)

    # Index robust zu String machen und Missing sauber labeln
    idx = []
    for v in ct.index.tolist():
        if pd.isna(v):
            idx.append("Missing")
        else:
            idx.append(str(v))
    ct.index = idx

    pct = ct.div(ct.sum(axis=0), axis=1) * 100

    for v in ["A", "B"]:
        if v not in pct.columns:
            pct[v] = 0.0

    return pct[["A", "B"]]

def plot_bar_overall(col, title, fname):
    if col not in df.columns:
        print("Skip (missing column):", col)
        return
    tab = percent_overall(col)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(tab.index, tab["pct"].values)
    ax.set_title(title)
    ax.set_ylabel("Anteil (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_fig(fig, fname)

def plot_bar_by_version(col, title, fname):
    if col not in df.columns:
        print("Skip (missing column):", col)
        return
    tab = percent_by_version(col)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cats = tab.index.tolist()
    x = np.arange(len(cats))
    w = 0.4
    ax.bar(x - w/2, tab["A"].values, w, label="A (KI)")
    ax.bar(x + w/2, tab["B"].values, w, label="B (ohne KI)")
    ax.set_title(title)
    ax.set_ylabel("Anteil innerhalb Version (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, fname)

def boxplot_by_version(col, title, fname):
    if col not in df.columns:
        print("Skip (missing column):", col)
        return
    a = pd.to_numeric(df[df["version"]=="A"][col], errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(df[df["version"]=="B"][col], errors="coerce").dropna().to_numpy()
    if len(a)==0 or len(b)==0:
        print("Skip (no data):", col)
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([a, b], labels=["A (KI)", "B (ohne KI)"], showfliers=True)

    jitter = 0.08
    ax.scatter(1 + np.random.uniform(-jitter, jitter, size=len(a)), a, s=10, alpha=0.5)
    ax.scatter(2 + np.random.uniform(-jitter, jitter, size=len(b)), b, s=10, alpha=0.5)

    ax.set_title(title)
    ax.set_ylabel("Skalenwert (Likert 1–7)")
    ax.set_ylim(1, 7)
    fig.tight_layout()
    save_fig(fig, fname)

demo_cols = [
    ("gender", "Geschlecht", "dem_gender"),
    ("age_group", "Alter", "dem_age"),
    ("education", "Bildung", "dem_education"),
    ("employment", "Erwerbsstatus", "dem_employment"),
    ("dhp_use_12m", "DHP Nutzung (12M)", "dem_dhp_use_12m"),
    ("ai_tools_use_12m", "KI-Tools Nutzung (12M)", "dem_ai_tools_use_12m"),
    ("dhp_freq", "DHP Nutzungshäufigkeit", "dem_dhp_freq"),
]

for col, label, base in demo_cols:
    plot_bar_overall(col, f"{label} – Gesamt", f"{base}_overall")
    plot_bar_by_version(col, f"{label} – A vs B", f"{base}_by_version")

scale_cols = [
    ("cx_median", "Customer Experience (Median) – A vs B", "box_cx"),
    ("trust_median", "Vertrauen (Median) – A vs B", "box_trust"),
    ("satisfaction_median", "Zufriedenheit (Median) – A vs B", "box_satisfaction"),
    ("loyalty_median", "Loyalität (Median) – A vs B", "box_loyalty"),
    ("personalization_median", "Personalisierung (Median) – A vs B", "box_personalization"),
    ("transparency_median", "Transparenz (Median) – A vs B", "box_transparency"),
]

for col, title, fname in scale_cols:
    boxplot_by_version(col, title, fname)

print("\nDONE. Dateien sollten jetzt hier liegen:", OUT_DIR)