# ============================================================
# AMR DATA CHALLENGE 2026 — SAVE PROFILE DATA AS CSV FILES
# ============================================================
# Saves every section of the profile report as a separate CSV
# Output folder: amr_profile_csvs/
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

ATLAS_PATH  = "atlas_vivli_2004_2024.csv"
GEARS_PATH  = "Venatorx surveillance data_2024_06_06.xlsx"
OUT_DIR     = "amr_profile_csvs"
os.makedirs(OUT_DIR, exist_ok=True)

def save(df, name):
    path = os.path.join(OUT_DIR, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df)} rows)")

TARGET_ORGANISMS = [
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
    "Enterococcus faecium",
    "Staphylococcus epidermidis",
    "Serratia marcescens",
    "Citrobacter koseri",
    "Streptococcus pneumoniae",
    "Haemophilus influenzae",
    "Salmonella spp"
]

CNS_NOSOCOMIAL_SOURCES_ATLAS = [
    "Blood", "CSF", "Brain", "CNS: Other", "Spinal Cord",
    "Head", "Peripheral Nerves",
    "Endotracheal aspirate", "Bronchoalveolar lavage", "Trachea", "Bronchus",
    "Catheters", "Drains",
    "Abscess", "Bone", "Bone Marrow",
    "Pleural Fluid", "Thoracentesis Fluid", "Peritoneal Fluid",
    "Abdominal Fluid", "Tissue Fluid", "Bodily Fluids", "Aspirate",
    "Lungs", "Respiratory: Other"
]

CNS_NOSOCOMIAL_SOURCES_GEARS = [
    "CVS: Blood",
    "Respiratory: Endotracheal aspirate",
    "Respiratory: Bronchoalveolar lavage",
    "Respiratory: Sputum",
    "Respiratory: Bronchial brushing",
    "Respiratory: Other",
    "Respiratory: Lungs",
    "Respiratory: Bronchials",
    "Bodily Fluids: Peritoneal",
    "Bodily Fluids: Thoracentesis",
    "Bodily Fluids: Abscess / Pus",
    "Bodily Fluids: Tissue",
    "GI: Abscess",
    "INT: Abscess",
    "INT: Wound"
]

VALID_SPECIALITIES_ATLAS = [
    "Medicine General", "Surgery General", "Medicine ICU",
    "Emergency Room", "Surgery ICU", "Pediatric General",
    "Clinic / Office", "Pediatric ICU", "Other",
    "General Unspecified ICU", "Nursing Home / Rehab"
]

VALID_FACILITIES_GEARS = [
    "Medicine General", "Medicine ICU", "Surgery General",
    "Surgery ICU", "Emergency Room", "General Unspecified ICU",
    "Pediatric ICU", "Pediatric General", "Other"
]

GENE_COLS = [
    "ACC","ACT/MIR","AMPC","CMY I/MOX","CMY2",
    "CTX-M-1","CTX-M-2","CTX-M-8/25","CTX-M-9",
    "DHA","FOX","GES","GIM","IMP","KPC","NDM",
    "OXA","PER","SHV","SPM","TEM","VEB","VIM"
]

# ============================================================
print("\n" + "="*60)
print("LOADING ATLAS...")
print("="*60)
atlas = pd.read_csv(ATLAS_PATH, low_memory=False,
                    na_values=["", "NA", "-", " ", "N/A"])
print(f"ATLAS loaded: {atlas.shape[0]:,} rows x {atlas.shape[1]} cols")

# Resolve column names
def resolve(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None

AGE_COL  = resolve(atlas, ["Age Group","Age.Group"])
SPEC_COL = resolve(atlas, ["Speciality","Specialty"])
SRC_COL  = resolve(atlas, ["Source"])
SP_COL   = resolve(atlas, ["Species"])
CTRY_COL = resolve(atlas, ["Country"])
YR_COL   = resolve(atlas, ["Year"])
GEN_COL  = resolve(atlas, ["Gender"])
STD_COL  = resolve(atlas, ["Study"])

interp_cols = [c for c in atlas.columns if c.endswith("_I")]

print("\nSaving ATLAS CSVs...")

# --------------------------------------------------
# A01 — Basic Dimensions
# --------------------------------------------------
df = pd.DataFrame({
    "Metric": [
        "Total rows (isolates)",
        "Total columns",
        "Year range start",
        "Year range end",
        "Total countries",
        "Total unique species"
    ],
    "Value": [
        atlas.shape[0],
        atlas.shape[1],
        int(atlas[YR_COL].min()),
        int(atlas[YR_COL].max()),
        atlas[CTRY_COL].nunique(),
        atlas[SP_COL].nunique()
    ]
})
save(df, "A01_atlas_basic_dimensions")

# --------------------------------------------------
# A02 — Column Names and Data Types
# --------------------------------------------------
df = pd.DataFrame({
    "Column_Name": atlas.columns.tolist(),
    "Data_Type":   [str(atlas[c].dtype) for c in atlas.columns],
    "Non_Null_Count": [atlas[c].notna().sum() for c in atlas.columns],
    "Null_Count":     [atlas[c].isna().sum()  for c in atlas.columns],
    "Null_Percent":   [round(atlas[c].isna().sum() / len(atlas) * 100, 2)
                       for c in atlas.columns]
})
save(df, "A02_atlas_column_names_dtypes")

# --------------------------------------------------
# A03 — Missing Values (only columns with missing)
# --------------------------------------------------
rows = []
for col in atlas.columns:
    mc = atlas[col].isna().sum()
    if mc > 0:
        rows.append({
            "Column":         col,
            "Missing_Count":  mc,
            "Missing_Percent": round(mc / len(atlas) * 100, 2)
        })
save(pd.DataFrame(rows), "A03_atlas_missing_values")

# --------------------------------------------------
# A04 — All Unique Species with Counts
# --------------------------------------------------
sp = atlas[SP_COL].value_counts(dropna=False).reset_index()
sp.columns = ["Species", "Count"]
sp["Rank"] = range(1, len(sp)+1)
save(sp, "A04_atlas_all_species_counts")

# --------------------------------------------------
# A05 — Years Covered
# --------------------------------------------------
yr = atlas[YR_COL].value_counts().sort_index().reset_index()
yr.columns = ["Year", "Isolate_Count"]
save(yr, "A05_atlas_years_covered")

# --------------------------------------------------
# A06 — Countries Covered
# --------------------------------------------------
ct = atlas[CTRY_COL].value_counts(dropna=False).reset_index()
ct.columns = ["Country", "Isolate_Count"]
ct["Rank"] = range(1, len(ct)+1)
save(ct, "A06_atlas_countries_covered")

# --------------------------------------------------
# A07 — Age Groups
# --------------------------------------------------
if AGE_COL:
    ag = atlas[AGE_COL].value_counts(dropna=False).reset_index()
    ag.columns = ["Age_Group", "Count"]
    ag["Percent"] = round(ag["Count"] / len(atlas) * 100, 2)
    save(ag, "A07_atlas_age_groups")

# --------------------------------------------------
# A08 — Gender Distribution
# --------------------------------------------------
if GEN_COL:
    gd = atlas[GEN_COL].value_counts(dropna=False).reset_index()
    gd.columns = ["Gender", "Count"]
    gd["Percent"] = round(gd["Count"] / len(atlas) * 100, 2)
    save(gd, "A08_atlas_gender_distribution")

# --------------------------------------------------
# A09 — All Specimen Sources
# --------------------------------------------------
if SRC_COL:
    sc = atlas[SRC_COL].value_counts(dropna=False).reset_index()
    sc.columns = ["Source", "Count"]
    sc["CNS_Nosocomial_Selected"] = sc["Source"].apply(
        lambda x: "YES" if x in CNS_NOSOCOMIAL_SOURCES_ATLAS else "NO"
    )
    save(sc, "A09_atlas_all_specimen_sources")

# --------------------------------------------------
# A10 — CNS/Nosocomial Sources Only (Study Scope)
# --------------------------------------------------
if SRC_COL:
    rows = []
    for src in CNS_NOSOCOMIAL_SOURCES_ATLAS:
        cnt = (atlas[SRC_COL] == src).sum()
        rows.append({"Source": src, "Count": cnt})
    df = pd.DataFrame(rows)
    df["Percent_of_Total"] = round(df["Count"] / len(atlas) * 100, 3)
    df.loc[len(df)] = ["TOTAL", df["Count"].sum(),
                       round(df["Count"].sum() / len(atlas) * 100, 2)]
    save(df, "A10_atlas_cns_nosocomial_sources_selected")

# --------------------------------------------------
# A11 — Speciality / Facility Type
# --------------------------------------------------
if SPEC_COL:
    sp2 = atlas[SPEC_COL].value_counts(dropna=False).reset_index()
    sp2.columns = ["Speciality", "Count"]
    sp2["Included_In_Study"] = sp2["Speciality"].apply(
        lambda x: "YES" if x in VALID_SPECIALITIES_ATLAS else "EXCLUDED"
    )
    sp2["Percent"] = round(sp2["Count"] / len(atlas) * 100, 2)
    save(sp2, "A11_atlas_speciality_facility_type")

# --------------------------------------------------
# A12 — Antibiotic Columns List
# --------------------------------------------------
mic_meta = {"Isolate Id","Isolate.Id","Study","Species","Country",
            "State","Gender","Age Group","Age.Group",
            "Speciality","Source","Year"}
confirmed_mic = []
for col in atlas.columns:
    if col.endswith("_I") or col in mic_meta:
        continue
    sample = atlas[col].dropna().astype(str)
    if len(sample) > 0 and sample.str.match(r'^[<>]?[0-9]').any():
        confirmed_mic.append(col)

rows = []
for col in interp_cols:
    abx = col.replace("_I","")
    na_count = atlas[col].isna().sum()
    pct_missing = round(na_count / len(atlas) * 100, 2)
    rows.append({
        "Antibiotic_Name":      abx,
        "Interpretation_Col":   col,
        "MIC_Col_Present":      "YES" if abx in confirmed_mic else "NO",
        "NA_Count":             na_count,
        "NA_Percent":           pct_missing,
        "Usability":            "UNUSABLE(100%NA)" if pct_missing == 100.0
                                else "LOW(<10%data)" if pct_missing >= 90
                                else "MODERATE" if pct_missing >= 50
                                else "GOOD"
    })
save(pd.DataFrame(rows), "A12_atlas_antibiotic_columns")

# --------------------------------------------------
# A13 — S/I/R Distribution for All 47 Antibiotics
# --------------------------------------------------
rows = []
for col in interp_cols:
    abx = col.replace("_I","")
    vc  = atlas[col].value_counts(dropna=True)
    s   = int(vc.get("Susceptible",  0))
    i   = int(vc.get("Intermediate", 0))
    r   = int(vc.get("Resistant",    0))
    na  = int(atlas[col].isna().sum())
    tot = s + i + r
    rpct = round(r / tot * 100, 2) if tot > 0 else None
    rows.append({
        "Antibiotic":        abx,
        "Susceptible":       s,
        "Intermediate":      i,
        "Resistant":         r,
        "NA":                na,
        "Total_Tested":      tot,
        "Resistance_Rate_%": rpct,
        "NA_Percent":        round(na / len(atlas) * 100, 2)
    })
save(pd.DataFrame(rows), "A13_atlas_SIR_distribution_all_antibiotics")

# --------------------------------------------------
# A14 — Study Codes
# --------------------------------------------------
if STD_COL:
    sd = atlas[STD_COL].value_counts(dropna=False).reset_index()
    sd.columns = ["Study_Code", "Count"]
    sd["Percent"] = round(sd["Count"] / len(atlas) * 100, 2)
    save(sd, "A14_atlas_study_codes")

# --------------------------------------------------
# A15 — Resistance Gene Columns
# --------------------------------------------------
gene_present = [c for c in GENE_COLS if c in atlas.columns]
rows = []
for col in gene_present:
    ne  = int(atlas[col].notna().sum())
    mis = int(atlas[col].isna().sum())
    rows.append({
        "Gene_Column":    col,
        "Non_Empty":      ne,
        "Missing":        mis,
        "Missing_Percent": round(mis / len(atlas) * 100, 2)
    })
save(pd.DataFrame(rows), "A15_atlas_resistance_gene_columns")

# --------------------------------------------------
# A16 — Target Organisms Row Counts
# --------------------------------------------------
rows = []
for org in TARGET_ORGANISMS:
    cnt = atlas[SP_COL].str.contains(org, case=False, na=False).sum()
    if   cnt >= 1000: status = "EXCELLENT"
    elif cnt >= 200:  status = "GOOD"
    elif cnt > 0:     status = "LOW"
    else:             status = "ABSENT"
    rows.append({
        "Organism":    org,
        "Total_Count": cnt,
        "Status":      status
    })
df = pd.DataFrame(rows)
df.loc[len(df)] = ["TOTAL", df["Total_Count"].sum(), ""]
save(df, "A16_atlas_target_organisms_counts")

# --------------------------------------------------
# A17 — Pediatric vs Elderly per Target Organism
# --------------------------------------------------
if AGE_COL:
    ped_df   = atlas[atlas[AGE_COL].isin(["0 - 17","0-17"])]
    elder_df = atlas[atlas[AGE_COL].isin(["61+","61 +"])]
    rows = []
    for org in TARGET_ORGANISMS:
        np_ = ped_df[SP_COL].str.contains(org, case=False, na=False).sum()
        ne_ = elder_df[SP_COL].str.contains(org, case=False, na=False).sum()
        rows.append({
            "Organism":        org,
            "Pediatric_0_17":  int(np_),
            "Elderly_61plus":  int(ne_),
            "Combined":        int(np_ + ne_),
            "Ped_Pct_of_Combined":
                round(np_ / (np_+ne_) * 100, 1) if (np_+ne_) > 0 else None
        })
    save(pd.DataFrame(rows), "A17_atlas_pediatric_vs_elderly_per_organism")

# ============================================================
print("\n" + "="*60)
print("LOADING GEARS...")
print("="*60)
gears = pd.read_excel(GEARS_PATH, sheet_name=0)
gears = gears.where(pd.notnull(gears), other=np.nan)
ages_g   = pd.to_numeric(gears['Age'], errors='coerce')
ped_g    = gears[ages_g <= 17]
elder_g  = gears[ages_g >= 61]
mic_cols_g = [c for c in gears.columns if c.endswith("_MIC")]
print(f"GEARS loaded: {gears.shape[0]:,} rows x {gears.shape[1]} cols")

print("\nSaving GEARS CSVs...")

# --------------------------------------------------
# G01 — Basic Dimensions
# --------------------------------------------------
df = pd.DataFrame({
    "Metric": [
        "Total rows",
        "Total columns",
        "Year range start",
        "Year range end",
        "Total countries",
        "Total unique organisms"
    ],
    "Value": [
        gears.shape[0],
        gears.shape[1],
        int(gears['Year'].min()),
        int(gears['Year'].max()),
        gears['Country'].nunique(),
        gears['Organism'].nunique()
    ]
})
save(df, "G01_gears_basic_dimensions")

# --------------------------------------------------
# G02 — Column Names and Data Types
# --------------------------------------------------
df = pd.DataFrame({
    "Column_Name":    gears.columns.tolist(),
    "Data_Type":      [str(gears[c].dtype) for c in gears.columns],
    "Non_Null_Count": [gears[c].notna().sum() for c in gears.columns],
    "Null_Count":     [gears[c].isna().sum()  for c in gears.columns],
    "Null_Percent":   [round(gears[c].isna().sum() / len(gears) * 100, 2)
                       for c in gears.columns]
})
save(df, "G02_gears_column_names_dtypes")

# --------------------------------------------------
# G03 — Missing Values
# --------------------------------------------------
rows = []
for col in gears.columns:
    mc = gears[col].isna().sum()
    rows.append({
        "Column":          col,
        "Missing_Count":   mc,
        "Missing_Percent": round(mc / len(gears) * 100, 2)
    })
df_miss = pd.DataFrame(rows)
df_miss = df_miss[df_miss["Missing_Count"] > 0]
if len(df_miss) == 0:
    df_miss = pd.DataFrame([{"Column":"No missing values","Missing_Count":0,"Missing_Percent":0}])
save(df_miss, "G03_gears_missing_values")

# --------------------------------------------------
# G04 — Unique Organisms
# --------------------------------------------------
go = gears['Organism'].value_counts(dropna=False).reset_index()
go.columns = ["Organism", "Count"]
go["Rank"] = range(1, len(go)+1)
go["Is_Target"] = go["Organism"].apply(
    lambda x: "YES" if any(t.lower() in str(x).lower()
                            for t in TARGET_ORGANISMS) else "NO"
)
save(go, "G04_gears_unique_organisms")

# --------------------------------------------------
# G05 — Years Covered
# --------------------------------------------------
gy = gears['Year'].value_counts().sort_index().reset_index()
gy.columns = ["Year", "Isolate_Count"]
save(gy, "G05_gears_years_covered")

# --------------------------------------------------
# G06 — Countries
# --------------------------------------------------
gc = gears['Country'].value_counts(dropna=False).reset_index()
gc.columns = ["Country", "Count"]
gc["Rank"] = range(1, len(gc)+1)
save(gc, "G06_gears_countries")

# --------------------------------------------------
# G07 — Age Distribution
# --------------------------------------------------
age_summary = pd.DataFrame({
    "Age_Group":  [
        "Neonatal (0)",
        "Pediatric 1-5",
        "Pediatric 6-17",
        "Total Pediatric (0-17)",
        "Adult 18-60",
        "Elderly (61+)",
        "Unknown/NA"
    ],
    "Count": [
        int((ages_g == 0).sum()),
        int(((ages_g >= 1) & (ages_g <= 5)).sum()),
        int(((ages_g >= 6) & (ages_g <= 17)).sum()),
        int((ages_g <= 17).sum()),
        int(((ages_g >= 18) & (ages_g <= 60)).sum()),
        int((ages_g >= 61).sum()),
        int(ages_g.isna().sum())
    ]
})
age_summary["Percent"] = round(
    age_summary["Count"] / len(gears) * 100, 2)
save(age_summary, "G07_gears_age_distribution")

# --------------------------------------------------
# G08 — Pediatric vs Elderly per Target Organism
# --------------------------------------------------
rows = []
for org in TARGET_ORGANISMS:
    total = gears['Organism'].str.contains(org, case=False, na=False).sum()
    np_   = ped_g['Organism'].str.contains(org, case=False, na=False).sum()
    ne_   = elder_g['Organism'].str.contains(org, case=False, na=False).sum()
    rows.append({
        "Organism":       org,
        "Total_in_GEARS": int(total),
        "Pediatric_0_17": int(np_),
        "Elderly_61plus": int(ne_),
        "Present_in_GEARS": "YES" if total > 0 else "ABSENT"
    })
save(pd.DataFrame(rows), "G08_gears_pediatric_vs_elderly_per_organism")

# --------------------------------------------------
# G09 — All Body Sites
# --------------------------------------------------
gbs = gears['BodySite'].value_counts(dropna=False).reset_index()
gbs.columns = ["BodySite", "Count"]
gbs["CNS_Nosocomial_Selected"] = gbs["BodySite"].apply(
    lambda x: "YES" if x in CNS_NOSOCOMIAL_SOURCES_GEARS else "NO"
)
save(gbs, "G09_gears_all_body_sites")

# --------------------------------------------------
# G10 — CNS/Nosocomial Body Sites Selected
# --------------------------------------------------
rows = []
for src in CNS_NOSOCOMIAL_SOURCES_GEARS:
    cnt = (gears['BodySite'] == src).sum()
    rows.append({"BodySite": src, "Count": cnt})
df = pd.DataFrame(rows)
df["Percent_of_Total"] = round(df["Count"] / len(gears) * 100, 3)
df.loc[len(df)] = ["TOTAL", df["Count"].sum(),
                   round(df["Count"].sum() / len(gears) * 100, 2)]
save(df, "G10_gears_cns_nosocomial_sources_selected")

# --------------------------------------------------
# G11 — Facility Type
# --------------------------------------------------
gf = gears['Facility'].value_counts(dropna=False).reset_index()
gf.columns = ["Facility", "Count"]
gf["Included_In_Study"] = gf["Facility"].apply(
    lambda x: "YES" if x in VALID_FACILITIES_GEARS else "EXCLUDED"
)
gf["Percent"] = round(gf["Count"] / len(gears) * 100, 2)
save(gf, "G11_gears_facility_type")

# --------------------------------------------------
# G12 — MIC Columns Summary
# --------------------------------------------------
rows = []
for col in mic_cols_g:
    vals = pd.to_numeric(gears[col], errors='coerce')
    nm   = int(vals.notna().sum())
    mis  = int(vals.isna().sum())
    rows.append({
        "MIC_Column":     col,
        "Non_Missing":    nm,
        "Missing":        mis,
        "Missing_Percent": round(mis / len(gears) * 100, 2),
        "Min":            round(vals.min(), 4)    if nm > 0 else None,
        "Max":            round(vals.max(), 4)    if nm > 0 else None,
        "Median":         round(vals.median(), 4) if nm > 0 else None,
        "Mean":           round(vals.mean(), 4)   if nm > 0 else None
    })
save(pd.DataFrame(rows), "G12_gears_mic_columns_summary")

# --------------------------------------------------
# G13 — Target Organisms in GEARS
# --------------------------------------------------
rows = []
for org in TARGET_ORGANISMS:
    total = gears['Organism'].str.contains(org, case=False, na=False).sum()
    np_   = ped_g['Organism'].str.contains(org, case=False, na=False).sum()
    ne_   = elder_g['Organism'].str.contains(org, case=False, na=False).sum()
    if   total >= 500: status = "EXCELLENT"
    elif total >= 100: status = "GOOD"
    elif total > 0:    status = "LOW"
    else:              status = "ABSENT"
    rows.append({
        "Organism":       org,
        "Total_in_GEARS": int(total),
        "Pediatric_0_17": int(np_),
        "Elderly_61plus": int(ne_),
        "Status":         status
    })
df = pd.DataFrame(rows)
df.loc[len(df)] = ["TOTAL", df["Total_in_GEARS"].sum(),
                   df["Pediatric_0_17"].sum(),
                   df["Elderly_61plus"].sum(), ""]
save(df, "G13_gears_target_organisms")

# ============================================================
# SUMMARY INDEX FILE
# ============================================================
index_rows = [
    # ATLAS
    ("A01","atlas","Basic dimensions (rows, cols, year range, countries, species)"),
    ("A02","atlas","All column names with data types and null counts"),
    ("A03","atlas","Missing values — columns with any missing data"),
    ("A04","atlas","All 400 unique species with isolate counts"),
    ("A05","atlas","Isolate counts per year (2004-2024)"),
    ("A06","atlas","Isolate counts per country (83 countries)"),
    ("A07","atlas","Age group distribution"),
    ("A08","atlas","Gender distribution"),
    ("A09","atlas","All 97 specimen sources with CNS/nosocomial flag"),
    ("A10","atlas","CNS/nosocomial-relevant sources selected for study"),
    ("A11","atlas","Speciality/facility type with included/excluded flag"),
    ("A12","atlas","All 47 antibiotic columns with usability classification"),
    ("A13","atlas","S/I/R distribution and resistance rate for all 47 antibiotics"),
    ("A14","atlas","Study codes (ATLAS, TEST, Inform)"),
    ("A15","atlas","Resistance gene columns with non-empty counts"),
    ("A16","atlas","Target organism row counts with status"),
    ("A17","atlas","Pediatric (0-17) vs elderly (61+) per target organism"),
    # GEARS
    ("G01","gears","Basic dimensions"),
    ("G02","gears","All column names with data types and null counts"),
    ("G03","gears","Missing values"),
    ("G04","gears","All 42 unique organisms with counts and target flag"),
    ("G05","gears","Isolate counts per year (2018-2022)"),
    ("G06","gears","Isolate counts per country (59 countries)"),
    ("G07","gears","Age distribution summary"),
    ("G08","gears","Pediatric vs elderly per target organism"),
    ("G09","gears","All body sites with CNS/nosocomial flag"),
    ("G10","gears","CNS/nosocomial-relevant body sites selected for study"),
    ("G11","gears","Facility type with included/excluded flag"),
    ("G12","gears","MIC columns: min, max, median, missing stats"),
    ("G13","gears","Target organisms in GEARS with pediatric/elderly breakdown"),
]
idx_df = pd.DataFrame(index_rows,
                       columns=["File_ID","Dataset","Description"])
idx_df["Filename"] = idx_df["File_ID"] + "_*.csv"
save(idx_df, "INDEX_all_csv_files")

print("\n" + "="*60)
print(f"ALL CSVs SAVED TO FOLDER: {OUT_DIR}/")
print(f"Total files: {len(index_rows) + 1}")
print("="*60)
