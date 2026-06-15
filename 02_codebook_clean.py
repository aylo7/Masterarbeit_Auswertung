# ============================================================
# 02_codebook_clean.py
# Zweck:
#   - Rohdaten (CSV) laden
#   - Spalten sauber & kurz umbenennen (rename_map)
#   - Zeit-Items als cx_time_save_1 und cx_time_save_2 führen
#   - Zeitersparnis-Index bilden: cx_time_save_index = mean(time1, time2)
#   - Konstrukte (CX, Personalisierung, Trust, etc.) als Item-Listen definieren
#     -> CX nutzt den Index statt beide Zeit-Items (keine Doppelzählung)
#   - Likert-Items in Zahlen umwandeln und Wertebereich 1-7 prüfen
#   - Codebook exportieren (Originalname -> neuer Name + Konstrukt)
#   - Cleaned Dataset exportieren
# ============================================================

import pandas as pd              # Tabellen/CSV/Excel verarbeiten
import numpy as np               # numerische Hilfsfunktionen (NaN etc.)
from pathlib import Path         # saubere Dateipfade


# ----------------------------------------
# 1) Pfade definieren
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # Projekt-Root (Repo-Hauptordner)

RAW_FILE = BASE_DIR / "data" / "raw" / "2026_02_28_Liste_Umfrageteilnahmen_.csv"  # CSV Input
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Output Ordner für Daten
OUTPUT_DIR = BASE_DIR / "outputs"                # Output Ordner für Codebook etc.

# Ordner anlegen, falls sie noch nicht existieren
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check: Existiert die CSV-Datei wirklich?
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"CSV-Datei nicht gefunden: {RAW_FILE}\n"
        "Bitte stelle sicher, dass die Datei in data/raw/ liegt und der Name exakt stimmt."
    )


# ----------------------------
# 2) CSV laden (robust: erst ; dann ,)
# ----------------------------

# Wir versuchen zuerst deutsches CSV-Format (Semikolon)
try:
    df_raw = pd.read_csv(RAW_FILE, sep=";", encoding="utf-8-sig")
except Exception:
    # Wenn das nicht klappt, versuchen wir Komma-Separator
    df_raw = pd.read_csv(RAW_FILE, sep=",", encoding="utf-8-sig")

# Arbeitskopie
df = df_raw.copy()

print("===== DATENÜBERBLICK =====")
print("Form (Zeilen, Spalten):", df.shape)
print()


# ----------------------------
# 3) Optional: nur vollständige Datensätze (beendet=True)
# ----------------------------

# In deinem Export steht "beendet" als Textwert "wahr"
if "beendet" in df.columns:
    beendet_norm = df["beendet"].astype(str).str.strip().str.lower()

    print("===== CHECK 'beendet' (roh, normalisiert) =====")
    print(beendet_norm.value_counts(dropna=False).head(20))
    print()

    completed_values = ["true", "1", "yes", "ja", "wahr"]
    df_filtered = df[beendet_norm.isin(completed_values)].copy()

    # Safety: Wenn Filter alles löschen würde, nicht filtern
    if df_filtered.shape[0] == 0:
        print("⚠️ WARNUNG: Filter auf 'beendet' würde 0 Zeilen ergeben -> Filter wird übersprungen.\n")
    else:
        df = df_filtered
        print("✅ Filter angewendet: beendet in", completed_values)
        print("Form nach Filter:", df.shape, "\n")
else:
    print("Hinweis: Spalte 'beendet' nicht gefunden -> kein Filter angewendet.\n")


# ----------------------------
# 4) Spalten umbenennen (kurz & konsistent)
# ----------------------------

rename_map = {
    # ---------- Meta/Technisch ----------
    "id": "id",
    "Zugeordnete Version": "version",      # A oder B
    "Gestartet am": "started_at",
    "Beendet am": "ended_at",
    "beendet": "completed",
    "attention_check": "attention_check",
    "created_at": "created_at",
    "consent": "consent",
    "Sprache": "language",
    "scenario_confirmed": "scenario_confirmed",

    # ---------- Demografie ----------
    "Alter": "age_group",
    "Geschlecht": "gender",
    "Höchster Bildungsabschluss": "education",
    "Erwerbstatus": "employment",
    "Region (PLZ)": "plz",

    # ---------- Nutzung/Vorerfahrung ----------
    "Nutzungshäufigkeit digitaler Gesundheitsplattformen": "dhp_freq",
    "Nutzung digitaNutzung digitaler Gesundheitsplattformen (letzte 12 Monate)ler Gesundheitsplattformen": "dhp_use_12m",
    "Nutzung KI-basierter Tools (letzte 12 Monate)": "ai_tools_use_12m",

    # ---------- Likert-Items (1–7) ----------
    "Ich habe den Eindruck, dass die Plattform insgesamt einfach verständlich ist. *": "cx_understandable",
    "Ich habe den Eindruck, dass ich mit der Plattform relevante Informationen schnell finden würde.": "cx_find_fast",
    "Ich habe den Eindruck, dass mir die Plattform dabei helfen würde, sinnvolle nächste Schritte abzuleiten.": "cx_next_steps",

    # Doppelt im Export: wir führen bewusst als zwei Variablen
    "Ich habe den Eindruck, dass mir die Plattform Zeit bei der Orientierung und Informationssuche sparen würde.": "cx_time_save_1",
    "Ich habe den Eindruck, dass mir die Plattform Zeit bei der Orientierung und Informationssuche sparen würde.2": "cx_time_save_2",

    "Ich habe den Eindruck, dass die Plattform passende Hinweise oder Empfehlungen für mein Anliegen geben würde.": "pers_recommend",
    "Ich habe den Eindruck, dass die Plattform für mich relevante Informationen priorisieren würde.": "pers_prioritize",
    "Ich habe den Eindruck, dass die Plattform mich durch klare Schritte oder Rückfragen unterstützen würde.": "pers_guidance",

    "Ich habe den Eindruck, dass die Plattform zuverlässig ist.": "trust_reliable",
    "Ich habe den Eindruck, dass ich den Informationen/Ergebnissen der Plattform vertrauen würde.": "trust_info",
    "Ich habe den Eindruck, dass die Plattform kompetent ist.": "trust_competent",
    "Ich habe den Eindruck, dass ich mich bei der Nutzung einer solchen Plattform sicher fühlen würde (z. B. hinsichtlich Datenschutz/Datensicherheit).": "trust_safe_privacy",
    "Ich habe den Eindruck, dass die Plattform verantwortungsvoll handelt.": "trust_responsible",

    "Ich habe den Eindruck, dass für mich nachvollziehbar ist, wie Inhalte oder Empfehlungen zustande kommen.": "transparency_explainable",

    "Ich wäre insgesamt mit einer solchen Plattform zufrieden.": "sat_overall",
    "Die Plattform entspricht meinen Erwartungen an eine digitale Gesundheitsplattform.": "sat_expectations",
    "Mein Gesamteindruck der Plattform ist positiv.": "sat_impression",

    "Ich würde eine solche Plattform künftig nutzen.": "loy_use_future",
    "Ich würde eine solche Plattform weiterempfehlen.": "loy_recommend",
    "Wenn ich die Wahl hatte, würde ich diese Plattform ähnlichen Angeboten vorziehen.": "loy_prefer",

    # ---------- Manipulationschecks (1–7) ----------
    "Ich habe den Eindruck, dass die Plattform digitale Assistenzfunktionen enthält (z. B. Chat/Assistent, geführte Schritte, automatisierte Hinweise).": "manip_assist_functions",
    "Ich habe den Eindruck, dass auf der Plattform künstliche Intelligenz eingesetzt wird.": "manip_ai",
}

# Prüfen, ob alle erwarteten Spalten existieren
missing_cols = [col for col in rename_map.keys() if col not in df.columns]
if len(missing_cols) > 0:
    raise ValueError(
        "Diese erwarteten Spalten fehlen im CSV-Export:\n"
        + "\n".join(missing_cols)
        + "\n\n=> Wenn Spaltennamen minimal abweichen, passe ich rename_map 1:1 an."
    )

# Umbenennen anwenden
df = df.rename(columns=rename_map)

print("===== UMBENENNUNG OK =====")
print("Beispiel neue Spalten:", list(df.columns)[:12])
print()


# ----------------------------
# 5) Likert-Items numerisch machen + Wertebereich prüfen
# ----------------------------

# Alle Likert-Spalten, die wir numerisch brauchen (inkl. Manipulationscheck)
likert_cols = [
    "cx_understandable", "cx_find_fast", "cx_next_steps", "cx_time_save_1", "cx_time_save_2",
    "pers_recommend", "pers_prioritize", "pers_guidance",
    "trust_reliable", "trust_info", "trust_competent", "trust_safe_privacy", "trust_responsible",
    "transparency_explainable",
    "sat_overall", "sat_expectations", "sat_impression",
    "loy_use_future", "loy_recommend", "loy_prefer",
    "manip_assist_functions", "manip_ai",
]

# In Zahlen umwandeln (falls als Text importiert)
for col in likert_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Wertebereich 1–7 prüfen
out_of_range = {}
for col in likert_cols:
    s = df[col].dropna()
    if len(s) > 0 and ((s.min() < 1) or (s.max() > 7)):
        out_of_range[col] = (float(s.min()), float(s.max()))

print("===== LIKERT-CHECK (1–7) =====")
if len(out_of_range) == 0:
    print("OK: Alle Likert-Werte liegen im Bereich 1–7.\n")
else:
    print("ACHTUNG: Werte außerhalb 1–7 gefunden:")
    for col, (mn, mx) in out_of_range.items():
        print(f" - {col}: min={mn}, max={mx}")
    print()


# ------------------------------------------------------------
# 6) Zeitersparnis-Index erzeugen (keine Doppelzählung in CX)
# ------------------------------------------------------------

# Wir kombinieren die beiden sehr ähnlichen Zeit-Items zu einem Index,
# damit Zeitersparnis nicht doppelt in die CX-Skala eingeht.
df["cx_time_save_index"] = df[["cx_time_save_1", "cx_time_save_2"]].mean(axis=1)


# ----------------------------
# 7) Konstrukte bündeln (Item-Listen)
# ----------------------------

constructs = {
    # CX/Usability/Effizienz -> nutzt den Index statt beide Zeit-Items
    "cx_usability_efficiency": [
        "cx_understandable",
        "cx_find_fast",
        "cx_next_steps",
        "cx_time_save_index",
    ],

    # Personalisierung/Support
    "personalization_support": [
        "pers_recommend",
        "pers_prioritize",
        "pers_guidance",
    ],

    # Vertrauen
    "trust": [
        "trust_reliable",
        "trust_info",
        "trust_competent",
        "trust_safe_privacy",
        "trust_responsible",
    ],

    # Transparenz (Single-Item)
    "transparency": [
        "transparency_explainable",
    ],

    # Zufriedenheit
    "satisfaction": [
        "sat_overall",
        "sat_expectations",
        "sat_impression",
    ],

    # Loyalität
    "loyalty": [
        "loy_use_future",
        "loy_recommend",
        "loy_prefer",
    ],

    # Manipulationscheck (nicht Teil der Hypothesenskalen)
    "manipulation_check": [
        "manip_assist_functions",
        "manip_ai",
    ],
}

print("===== KONSTRUKTE =====")
for con_name, items in constructs.items():
    print(con_name, "->", items)
print()


# ----------------------------
# 8) Version A/B in 0/1 codieren
# ----------------------------

df["condition_ai"] = df["version"].astype(str).str.strip().map({"A": 1, "B": 0})

if df["condition_ai"].isna().any():
    raise ValueError("Version konnte nicht sauber in 0/1 umcodiert werden. Bitte version-Spalte prüfen (A/B).")

print("===== VERSION CODING =====")
print(df["version"].value_counts(dropna=False))
print(df["condition_ai"].value_counts(dropna=False))
print()


# ----------------------------
# 9) Codebook bauen
# ----------------------------

codebook_rows = []

# rename_map dokumentieren
for original_name, new_name in rename_map.items():
    block = "meta"
    construct = ""

    # Demografie
    if new_name in ["age_group", "gender", "education", "employment", "plz"]:
        block = "demographics"

    # Nutzung
    if new_name in ["dhp_freq", "dhp_use_12m", "ai_tools_use_12m", "scenario_confirmed"]:
        block = "usage_experience"

    # Likert / Konstrukte
    for con, items in constructs.items():
        if new_name in items:
            block = "likert_item"
            construct = con

    codebook_rows.append({
        "original_column": original_name,
        "new_column": new_name,
        "block": block,
        "construct": construct
    })

# Zusätzlich: Index-Spalte dokumentieren
codebook_rows.append({
    "original_column": "DERIVED: mean(cx_time_save_1, cx_time_save_2)",
    "new_column": "cx_time_save_index",
    "block": "likert_item",
    "construct": "cx_usability_efficiency"
})

codebook = pd.DataFrame(codebook_rows)

print("===== CODEBOOK PREVIEW =====")
print(codebook.head(12))
print()


# ----------------------------
# 10) Speichern: Cleaned Dataset + Codebook + Constructs
# ----------------------------

clean_path = PROCESSED_DIR / "survey_clean.xlsx"
codebook_path = OUTPUT_DIR / "codebook.xlsx"
constructs_path = OUTPUT_DIR / "constructs.json"

df.to_excel(clean_path, index=False)
codebook.to_excel(codebook_path, index=False)

# constructs als JSON speichern
pd.Series(constructs).to_json(constructs_path)

print("===== EXPORT FERTIG =====")
print("Cleaned data:", clean_path)
print("Codebook:", codebook_path)
print("Constructs JSON:", constructs_path)
print("\nDONE.")