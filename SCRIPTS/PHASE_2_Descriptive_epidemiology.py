# ============================================================
# AMR DATA CHALLENGE 2026 - PHASE 2: DESCRIPTIVE EPIDEMIOLOGY
# Nosocomial Neuroinvasive Bacterial AMR Study
# ============================================================
# REFERENCES:
#   MDR/XDR/PDR: Magiorakos et al. CMI 2012;18(3):268-281
#   Resistance rate formula: WHO GLASS 2023, ECDC EARS-Net 2023
#   Minimum threshold: WHO GLASS 2023 (>=10 isolates)
#   WHO Regions: WHO Global Health Observatory official list
#   Ladder: WHO Priority Pathogens 2024 + Magiorakos 2012
# ============================================================
# INPUT:  phase1_outputs/atlas_filtered_mdr_flagged.csv
#         phase1_outputs/gears_filtered_mdr_flagged.csv
# OUTPUT: phase2_outputs/ (all CSVs listed at end)
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
ATLAS_PATH = "phase1_outputs/atlas_filtered_mdr_flagged.csv"
GEARS_PATH = "phase1_outputs/gears_filtered_mdr_flagged.csv"
OUT_DIR    = "phase2_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_ISOLATES = 10  # WHO GLASS 2023 minimum threshold

def save(df, name):
    path = os.path.join(OUT_DIR, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df):,} rows)")

# ============================================================
# STUDY PARAMETERS
# ============================================================

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

GRAM_NEGATIVE = [
    "Klebsiella pneumoniae", "Acinetobacter baumannii",
    "Escherichia coli", "Pseudomonas aeruginosa",
    "Enterobacter cloacae", "Serratia marcescens",
    "Citrobacter koseri", "Haemophilus influenzae",
    "Salmonella spp"
]

GRAM_POSITIVE = [
    "Staphylococcus aureus", "Enterococcus faecium",
    "Staphylococcus epidermidis", "Streptococcus pneumoniae"
]

# ============================================================
# MDR CLASS MAP (15 classes)
# Non-susceptible = Resistant OR Intermediate
# ============================================================
MDR_CLASSES = {
    "Penicillins": [
        "Ampicillin_I","Amoxycillin clavulanate_I",
        "Ampicillin sulbactam_I","Piperacillin tazobactam_I","Penicillin_I"
    ],
    "Cephalosporins": [
        "Cefepime_I","Ceftazidime_I","Ceftriaxone_I","Ceftaroline_I",
        "Ceftazidime avibactam_I","Ceftolozane tazobactam_I"
    ],
    "Monobactams":      ["Aztreonam_I"],
    "Carbapenems":      ["Meropenem_I","Imipenem_I","Doripenem_I"],
    "Fluoroquinolones": ["Levofloxacin_I","Ciprofloxacin_I"],
    "Aminoglycosides":  ["Amikacin_I","Gentamicin_I"],
    "Glycopeptides":    ["Vancomycin_I","Teicoplanin_I"],
    "Polymyxins":       ["Colistin_I"],
    "Tetracyclines":    ["Minocycline_I","Tigecycline_I"],
    "Trimethoprim_Sulfonamides": ["Trimethoprim sulfa_I"],
    "Oxazolidinones":   ["Linezolid_I"],
    "Lipopeptides":     ["Daptomycin_I"],
    "Penicillinase_Resistant_Penicillins": ["Oxacillin_I"],
    "Lincosamides":     ["Clindamycin_I"],
    "Macrolides":       ["Erythromycin_I"]
}
TOTAL_CLASSES = len(MDR_CLASSES)  # 15

# ============================================================
# WHO OFFICIAL COUNTRY → REGION MAPPING
# Source: WHO Global Health Observatory
# ============================================================
WHO_REGION_MAP = {
    # AFRO
    "Nigeria":"AFRO","South Africa":"AFRO","Kenya":"AFRO",
    "Cameroon":"AFRO","Ivory Coast":"AFRO","Uganda":"AFRO",
    "Malawi":"AFRO","Ghana":"AFRO","Namibia":"AFRO",
    "Mauritius":"AFRO",
    # AMRO
    "United States":"AMRO","Canada":"AMRO","Mexico":"AMRO",
    "Brazil":"AMRO","Argentina":"AMRO","Colombia":"AMRO",
    "Chile":"AMRO","Venezuela":"AMRO","Guatemala":"AMRO",
    "Panama":"AMRO","Costa Rica":"AMRO","Dominican Republic":"AMRO",
    "Honduras":"AMRO","El Salvador":"AMRO","Jamaica":"AMRO",
    "Nicaragua":"AMRO","Puerto Rico":"AMRO","Ecuador":"AMRO",
    # EMRO
    "Egypt":"EMRO","Jordan":"EMRO","Kuwait":"EMRO",
    "Lebanon":"EMRO","Morocco":"EMRO","Oman":"EMRO",
    "Pakistan":"EMRO","Qatar":"EMRO","Saudi Arabia":"EMRO",
    "Tunisia":"EMRO",
    # EURO
    "Spain":"EURO","France":"EURO","Germany":"EURO",
    "Italy":"EURO","Belgium":"EURO","Portugal":"EURO",
    "United Kingdom":"EURO","Czech Republic":"EURO","Hungary":"EURO",
    "Greece":"EURO","Poland":"EURO","Russia":"EURO",
    "Croatia":"EURO","Romania":"EURO","Netherlands":"EURO",
    "Denmark":"EURO","Ireland":"EURO","Switzerland":"EURO",
    "Sweden":"EURO","Austria":"EURO","Lithuania":"EURO",
    "Finland":"EURO","Latvia":"EURO","Ukraine":"EURO",
    "Bulgaria":"EURO","Slovenia":"EURO","Slovak Republic":"EURO",
    "Serbia":"EURO","Norway":"EURO","Estonia":"EURO",
    "Israel":"EURO","Turkey":"EURO",
    # SEARO
    "India":"SEARO","Thailand":"SEARO","Indonesia":"SEARO",
    "Bangladesh":"SEARO",
    # WPRO
    "China":"WPRO","Japan":"WPRO","Korea, South":"WPRO",
    "Taiwan":"WPRO","Australia":"WPRO","Philippines":"WPRO",
    "Malaysia":"WPRO","Hong Kong":"WPRO","Singapore":"WPRO",
    "New Zealand":"WPRO","Vietnam":"WPRO",
}

# ============================================================
# MDR PROGRESSION LADDER DEFINITIONS
# Reference: WHO Priority Pathogens 2024 + Magiorakos 2012
# Non-susceptible = R or I
# ============================================================

def classify_ladder_gn(row, cols):
    """
    Gram-negative ladder.
    Step 0: Susceptible to all tested classes
    Step 1: 3rd-gen Cephalosporin non-susceptible (ESBL-like)
    Step 2: Step1 + Fluoroquinolone non-susceptible
    Step 3: Step2 + Carbapenem non-susceptible (CR)
    Step 4: Step3 + Polymyxin non-susceptible (last resort)
    Returns highest step reached.
    """
    def ns(drug_cols):
        available = [c for c in drug_cols if c in cols]
        vals = [row[c] for c in available if pd.notna(row[c])]
        return any(v in ["Resistant","Intermediate"] for v in vals)

    ceph_cols  = ["Ceftriaxone_I","Ceftazidime_I","Cefepime_I"]
    fq_cols    = ["Levofloxacin_I","Ciprofloxacin_I"]
    carb_cols  = ["Meropenem_I","Imipenem_I","Doripenem_I"]
    poly_cols  = ["Colistin_I"]

    if ns(poly_cols) and ns(carb_cols) and ns(ceph_cols):
        return 4
    elif ns(carb_cols) and ns(ceph_cols):
        return 3
    elif ns(fq_cols) and ns(ceph_cols):
        return 2
    elif ns(ceph_cols):
        return 1
    else:
        return 0

def classify_ladder_gp(row, cols):
    """
    Gram-positive ladder.
    Step 0: Susceptible to all tested classes
    Step 1: Penicillin non-susceptible
    Step 2: Step1 + Methicillin/Oxacillin non-susceptible (MRSA/MRSE)
    Step 3: Step2 + Fluoroquinolone non-susceptible
    Step 4: Step3 + Glycopeptide non-susceptible (VRE/VRSA)
    """
    def ns(drug_cols):
        available = [c for c in drug_cols if c in cols]
        vals = [row[c] for c in available if pd.notna(row[c])]
        return any(v in ["Resistant","Intermediate"] for v in vals)

    pen_cols  = ["Ampicillin_I","Penicillin_I","Amoxycillin clavulanate_I"]
    meth_cols = ["Oxacillin_I"]
    fq_cols   = ["Levofloxacin_I","Ciprofloxacin_I"]
    glyc_cols = ["Vancomycin_I","Teicoplanin_I"]

    if ns(glyc_cols) and ns(fq_cols) and ns(meth_cols) and ns(pen_cols):
        return 4
    elif ns(fq_cols) and ns(meth_cols) and ns(pen_cols):
        return 3
    elif ns(meth_cols) and ns(pen_cols):
        return 2
    elif ns(pen_cols):
        return 1
    else:
        return 0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resistance_rate(r, s, i):
    """WHO GLASS formula: R / (R+S+I) * 100"""
    total = r + s + i
    if total < MIN_ISOLATES:
        return None  # flagged insufficient
    return round(r / total * 100, 2)

def ns_count(row, drug_cols, df_cols):
    """Count non-susceptible (R or I) for given drug columns."""
    available = [c for c in drug_cols if c in df_cols]
    vals = [row[c] for c in available if pd.notna(row[c])]
    return sum(1 for v in vals if v in ["Resistant","Intermediate"])

def compute_mxp(row, df_cols):
    """
    Compute MDR/XDR/PDR flags per isolate.
    Magiorakos 2012: non-susceptible = R or I.
    MDR: non-susceptible in >= 3 categories
    XDR: non-susceptible in all but <= 2 categories
         (susceptible to only 1 or 2 categories)
    PDR: non-susceptible in ALL categories
    Returns: (mdr, xdr, pdr, ns_class_count, ns_classes)
    """
    ns_classes = []
    skipped    = []
    for cls, drugs in MDR_CLASSES.items():
        available = [d for d in drugs if d in df_cols]
        if not available:
            skipped.append(cls)
            continue
        vals = [row[d] for d in available if pd.notna(row[d])]
        if not vals:
            skipped.append(cls)
            continue
        if any(v in ["Resistant","Intermediate"] for v in vals):
            ns_classes.append(cls)

    tested_classes = TOTAL_CLASSES - len(skipped)
    ns_count_val   = len(ns_classes)
    s_count        = tested_classes - ns_count_val

    mdr = 1 if ns_count_val >= 3 else 0
    xdr = 1 if (tested_classes > 0 and s_count <= 2 and ns_count_val >= 1) else 0
    pdr = 1 if (tested_classes > 0 and s_count == 0 and ns_count_val > 0) else 0

    # XDR must be MDR first (Magiorakos: XDR is subset of MDR)
    if xdr == 1 and mdr == 0:
        xdr = 0

    return mdr, xdr, pdr, ns_count_val, ";".join(ns_classes)

def get_organism_label(species_val):
    for org in TARGET_ORGANISMS:
        if org.lower() in str(species_val).lower():
            return org
    return str(species_val)

# ============================================================
# LOAD PHASE 1 OUTPUTS
# ============================================================
print("\n" + "="*65)
print("LOADING PHASE 1 OUTPUTS")
print("="*65)

atlas = pd.read_csv(ATLAS_PATH, low_memory=False)
gears = pd.read_csv(GEARS_PATH, low_memory=False)

print(f"  ATLAS loaded: {len(atlas):,} rows x {atlas.shape[1]} cols")
print(f"  GEARS loaded: {len(gears):,} rows x {gears.shape[1]} cols")

# Resolve column names
def resolve(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None

SP_COL   = resolve(atlas, ["Species"])
AGE_COL  = resolve(atlas, ["Age Group","Age.Group"])
SRC_COL  = resolve(atlas, ["Source"])
CTRY_COL = resolve(atlas, ["Country"])
YR_COL   = resolve(atlas, ["Year"])

# Standardize organism labels
atlas["Organism_Label"] = atlas[SP_COL].apply(get_organism_label)
gears["Organism_Label"] = gears["Organism"].apply(get_organism_label)

# Add WHO region
atlas["WHO_Region"] = atlas[CTRY_COL].map(WHO_REGION_MAP).fillna("Unclassified")
gears["WHO_Region"] = gears["Country"].map(WHO_REGION_MAP).fillna("Unclassified")

atlas_cols = set(atlas.columns)

# ============================================================
# STEP 1  RECOMPUTE MDR/XDR/PDR 
# ============================================================
print("\n" + "="*65)
print("STEP 1 — RECOMPUTING MDR/XDR/PDR (Magiorakos 2012, R+I=non-susceptible)")
print("="*65)
print("  Processing... (may take a few minutes)")

mxp_results = atlas.apply(
    lambda row: compute_mxp(row, atlas_cols), axis=1
)
atlas["MDR_v2"]          = mxp_results.apply(lambda x: x[0])
atlas["XDR"]             = mxp_results.apply(lambda x: x[1])
atlas["PDR"]             = mxp_results.apply(lambda x: x[2])
atlas["NS_Class_Count"]  = mxp_results.apply(lambda x: x[3])
atlas["NS_Classes"]      = mxp_results.apply(lambda x: x[4])

print(f"  MDR (v2, R+I): {atlas['MDR_v2'].sum():,} ({atlas['MDR_v2'].mean()*100:.1f}%)")
print(f"  XDR          : {atlas['XDR'].sum():,} ({atlas['XDR'].mean()*100:.1f}%)")
print(f"  PDR          : {atlas['PDR'].sum():,} ({atlas['PDR'].mean()*100:.1f}%)")

# ============================================================
# STEP 2  ADD LADDER CLASSIFICATION
# ============================================================
print("\n" + "="*65)
print("STEP 2 — MDR PROGRESSION LADDER CLASSIFICATION")
print("="*65)

def assign_ladder(row):
    org = str(row["Organism_Label"])
    if any(g.lower() in org.lower() for g in GRAM_POSITIVE):
        return classify_ladder_gp(row, atlas_cols)
    else:
        return classify_ladder_gn(row, atlas_cols)

atlas["Ladder_Step"] = atlas.apply(assign_ladder, axis=1)

for step in range(5):
    cnt = (atlas["Ladder_Step"] == step).sum()
    print(f"  Step {step}: {cnt:,} ({cnt/len(atlas)*100:.1f}%)")

# ============================================================
# STEP 3  RESISTANCE RATES: ALL ANTIBIOTICS × ALL ORGANISMS
# Reference: WHO GLASS formula R/(R+S+I)*100
# ============================================================
print("\n" + "="*65)
print("STEP 3 — RESISTANCE RATES: ALL ANTIBIOTICS × ALL ORGANISMS")
print("="*65)

interp_cols = [c for c in atlas.columns if c.endswith("_I") and
               c not in ["Aztreonam avibactam_I","Cefoperazone sulbactam_I",
                          "Gatifloxacin_I","Sulbactam_I"]]

# 3a — Wide format: organisms as rows, antibiotics as columns
rows_wide = []
for org in TARGET_ORGANISMS:
    sub = atlas[atlas["Organism_Label"] == org]
    row_data = {"Organism": org, "Total_Isolates": len(sub)}
    for col in interp_cols:
        abx = col.replace("_I","")
        r   = (sub[col] == "Resistant").sum()
        s   = (sub[col] == "Susceptible").sum()
        i   = (sub[col] == "Intermediate").sum()
        rate = resistance_rate(r, s, i)
        row_data[f"{abx}_R"] = r
        row_data[f"{abx}_S"] = s
        row_data[f"{abx}_I_count"] = i
        row_data[f"{abx}_Tested"] = r+s+i
        row_data[f"{abx}_Rate_%"] = rate if rate is not None else "InsufficientData"
    rows_wide.append(row_data)

df_wide = pd.DataFrame(rows_wide)
save(df_wide, "FULL_resistance_all_antibiotics_wide")

# 3b — Long format
rows_long = []
for org in TARGET_ORGANISMS:
    sub = atlas[atlas["Organism_Label"] == org]
    for col in interp_cols:
        abx  = col.replace("_I","")
        r    = int((sub[col] == "Resistant").sum())
        s    = int((sub[col] == "Susceptible").sum())
        i    = int((sub[col] == "Intermediate").sum())
        na   = int(sub[col].isna().sum())
        tot  = r + s + i
        rate = resistance_rate(r, s, i)
        rows_long.append({
            "Organism":        org,
            "Antibiotic":      abx,
            "Resistant":       r,
            "Susceptible":     s,
            "Intermediate":    i,
            "NA":              na,
            "Total_Tested":    tot,
            "Resistance_Rate_%": rate,
            "Sufficient_Data": "YES" if tot >= MIN_ISOLATES else "NO"
        })

df_long = pd.DataFrame(rows_long)
save(df_long, "FULL_resistance_all_antibiotics_long")
print(f"  Resistance rates computed for {len(TARGET_ORGANISMS)} organisms × {len(interp_cols)} antibiotics")

# ============================================================
# STEP 4  PEDIATRIC vs ELDERLY COMPARISON
# Reference: WHO GLASS age-stratified analysis
# ============================================================
print("\n" + "="*65)
print("STEP 4 — PEDIATRIC vs ELDERLY RESISTANCE COMPARISON")
print("="*65)

rows_pe = []
for org in TARGET_ORGANISMS:
    for age_cat in ["Pediatric","Elderly"]:
        sub = atlas[(atlas["Organism_Label"] == org) &
                    (atlas["Age_Category"] == age_cat)]
        for col in interp_cols:
            abx  = col.replace("_I","")
            r    = int((sub[col] == "Resistant").sum())
            s    = int((sub[col] == "Susceptible").sum())
            i    = int((sub[col] == "Intermediate").sum())
            tot  = r + s + i
            rate = resistance_rate(r, s, i)
            rows_pe.append({
                "Organism":        org,
                "Age_Group":       age_cat,
                "Antibiotic":      abx,
                "Resistant":       r,
                "Susceptible":     s,
                "Intermediate":    i,
                "Total_Tested":    tot,
                "Resistance_Rate_%": rate,
                "Sufficient_Data": "YES" if tot >= MIN_ISOLATES else "NO"
            })

df_pe_long = pd.DataFrame(rows_pe)
save(df_pe_long, "FULL_resistance_pediatric_vs_elderly_long")

# Wide pivot: organism × antibiotic, Ped rate vs Elderly rate side by side
pivot_ped = df_pe_long[df_pe_long["Age_Group"]=="Pediatric"][
    ["Organism","Antibiotic","Resistance_Rate_%","Total_Tested"]
].rename(columns={"Resistance_Rate_%":"Ped_Rate_%","Total_Tested":"Ped_N"})

pivot_eld = df_pe_long[df_pe_long["Age_Group"]=="Elderly"][
    ["Organism","Antibiotic","Resistance_Rate_%","Total_Tested"]
].rename(columns={"Resistance_Rate_%":"Elderly_Rate_%","Total_Tested":"Elderly_N"})

df_pe_wide = pd.merge(pivot_ped, pivot_eld, on=["Organism","Antibiotic"])
df_pe_wide["Rate_Difference_Ped_minus_Elderly"] = df_pe_wide.apply(
    lambda r: round(float(r["Ped_Rate_%"]) - float(r["Elderly_Rate_%"]), 2)
    if (r["Ped_Rate_%"] not in [None,"InsufficientData"] and
        r["Elderly_Rate_%"] not in [None,"InsufficientData"]) else None, axis=1
)
save(df_pe_wide, "FULL_resistance_pediatric_vs_elderly_wide")
print(f"  Pediatric vs Elderly comparison: {len(df_pe_wide):,} organism-antibiotic pairs")

# ============================================================
# STEP 5  RESISTANCE TRENDS 2004–2024
# ============================================================
print("\n" + "="*65)
print("STEP 5 — RESISTANCE TRENDS BY YEAR (2004-2024)")
print("="*65)

# Key antibiotics for trend analysis — one per class, best coverage
KEY_ANTIBIOTICS = {
    "Ceftriaxone_I":        "Cephalosporins",
    "Meropenem_I":          "Carbapenems",
    "Levofloxacin_I":       "Fluoroquinolones",
    "Amikacin_I":           "Aminoglycosides",
    "Vancomycin_I":         "Glycopeptides",
    "Colistin_I":           "Polymyxins",
    "Piperacillin tazobactam_I": "Penicillins",
    "Tigecycline_I":        "Tetracyclines",
    "Trimethoprim sulfa_I": "Trimethoprim",
    "Linezolid_I":          "Oxazolidinones"
}

rows_trend = []
for org in TARGET_ORGANISMS:
    for year in sorted(atlas[YR_COL].dropna().unique()):
        sub = atlas[(atlas["Organism_Label"] == org) &
                    (atlas[YR_COL] == year)]
        for col, cls in KEY_ANTIBIOTICS.items():
            if col not in atlas.columns:
                continue
            r    = int((sub[col] == "Resistant").sum())
            s    = int((sub[col] == "Susceptible").sum())
            i    = int((sub[col] == "Intermediate").sum())
            tot  = r + s + i
            rate = resistance_rate(r, s, i)
            rows_trend.append({
                "Organism":          org,
                "Year":              int(year),
                "Antibiotic":        col.replace("_I",""),
                "Antibiotic_Class":  cls,
                "Resistant":         r,
                "Susceptible":       s,
                "Intermediate":      i,
                "Total_Tested":      tot,
                "Resistance_Rate_%": rate,
                "Sufficient_Data":   "YES" if tot >= MIN_ISOLATES else "NO"
            })

df_trend = pd.DataFrame(rows_trend)
save(df_trend, "FULL_resistance_trends_2004_2024")
print(f"  Trend rows: {len(df_trend):,}")

# ============================================================
# STEP 6  MDR/XDR/PDR BURDEN BY ORGANISM AND AGE GROUP
# ============================================================
print("\n" + "="*65)
print("STEP 6 — MDR/XDR/PDR BURDEN BY ORGANISM AND AGE GROUP")
print("="*65)

rows_mxp = []
for org in TARGET_ORGANISMS:
    for age_cat in ["Pediatric","Elderly","All"]:
        if age_cat == "All":
            sub = atlas[atlas["Organism_Label"] == org]
        else:
            sub = atlas[(atlas["Organism_Label"] == org) &
                        (atlas["Age_Category"] == age_cat)]
        n   = len(sub)
        mdr = int(sub["MDR_v2"].sum())
        xdr = int(sub["XDR"].sum())
        pdr = int(sub["PDR"].sum())
        rows_mxp.append({
            "Organism":     org,
            "Age_Group":    age_cat,
            "Total_N":      n,
            "MDR_Count":    mdr,
            "XDR_Count":    xdr,
            "PDR_Count":    pdr,
            "MDR_Rate_%":   round(mdr/n*100,2) if n > 0 else None,
            "XDR_Rate_%":   round(xdr/n*100,2) if n > 0 else None,
            "PDR_Rate_%":   round(pdr/n*100,2) if n > 0 else None,
            "Non_MDR_Count": n - mdr
        })

df_mxp = pd.DataFrame(rows_mxp)
save(df_mxp, "FULL_MDR_XDR_PDR_burden_organism_age")
print(f"  MDR/XDR/PDR burden table: {len(df_mxp):,} rows")

# ============================================================
# STEP 7  MDR LADDER BY ORGANISM AND AGE GROUP
# ============================================================
print("\n" + "="*65)
print("STEP 7 — MDR PROGRESSION LADDER BY ORGANISM AND AGE")
print("="*65)

rows_ladder = []
for org in TARGET_ORGANISMS:
    for age_cat in ["Pediatric","Elderly","All"]:
        if age_cat == "All":
            sub = atlas[atlas["Organism_Label"] == org]
        else:
            sub = atlas[(atlas["Organism_Label"] == org) &
                        (atlas["Age_Category"] == age_cat)]
        n = len(sub)
        if n == 0:
            continue
        gram_type = "Gram-Positive" if org in GRAM_POSITIVE else "Gram-Negative"
        for step in range(5):
            cnt = int((sub["Ladder_Step"] == step).sum())
            rows_ladder.append({
                "Organism":   org,
                "Gram_Type":  gram_type,
                "Age_Group":  age_cat,
                "Ladder_Step": step,
                "Step_Label": [
                    "Step0_Susceptible",
                    "Step1_CephR_ESBLlike" if org in GRAM_NEGATIVE else "Step1_PenR",
                    "Step2_CephR_FQR"      if org in GRAM_NEGATIVE else "Step2_MRSA_MRSE",
                    "Step3_CarbapenemR"    if org in GRAM_NEGATIVE else "Step3_FQR",
                    "Step4_ColistinR"      if org in GRAM_NEGATIVE else "Step4_GlycopeptideR"
                ][step],
                "Count":      cnt,
                "Percent":    round(cnt/n*100, 2),
                "Total_N":    n
            })

df_ladder = pd.DataFrame(rows_ladder)
save(df_ladder, "FULL_MDR_ladder_organism_age")

# ============================================================
# STEP 8  COUNTRY × ORGANISM RESISTANCE HEATMAP DATA
# 83 countries × 13 organisms × antibiotics
# ============================================================
print("\n" + "="*65)
print("STEP 8 — COUNTRY × ORGANISM RESISTANCE HEATMAP DATA")
print("="*65)

rows_cty = []
for country in sorted(atlas[CTRY_COL].dropna().unique()):
    for org in TARGET_ORGANISMS:
        sub = atlas[(atlas[CTRY_COL] == country) &
                    (atlas["Organism_Label"] == org)]
        n = len(sub)
        if n == 0:
            continue
        mdr = int(sub["MDR_v2"].sum())
        xdr = int(sub["XDR"].sum())
        who_reg = WHO_REGION_MAP.get(country, "Unclassified")
        row = {
            "Country":        country,
            "WHO_Region":     who_reg,
            "Organism":       org,
            "Total_Isolates": n,
            "MDR_Count":      mdr,
            "XDR_Count":      xdr,
            "MDR_Rate_%":     round(mdr/n*100,2) if n > 0 else None,
            "XDR_Rate_%":     round(xdr/n*100,2) if n > 0 else None,
            "Sufficient_Data":"YES" if n >= MIN_ISOLATES else "NO"
        }
        # Add key antibiotic rates
        for col, cls in KEY_ANTIBIOTICS.items():
            if col not in atlas.columns:
                continue
            r   = int((sub[col] == "Resistant").sum())
            s   = int((sub[col] == "Susceptible").sum())
            i   = int((sub[col] == "Intermediate").sum())
            rate = resistance_rate(r, s, i)
            row[f"{col.replace('_I','')}_{cls}_Rate_%"] = rate
        rows_cty.append(row)

df_cty = pd.DataFrame(rows_cty)
save(df_cty, "FULL_country_organism_heatmap_data")
print(f"  Country×Organism rows: {len(df_cty):,}")

# ============================================================
# STEP 9  CNS-DIRECT SENSITIVITY ANALYSIS
# Compare Blood isolates vs pure CNS (CSF+Brain+SpinalCord+CNS:Other)
# ============================================================
print("\n" + "="*65)
print("STEP 9 — CNS-DIRECT SENSITIVITY ANALYSIS")
print("="*65)

cns_direct = ["CSF","Brain","CNS: Other","Spinal Cord","Head","Peripheral Nerves"]
blood_only  = ["Blood"]

rows_cns = []
for org in TARGET_ORGANISMS:
    for src_group, src_list in [("Blood_only", blood_only),
                                  ("CNS_direct", cns_direct)]:
        sub = atlas[(atlas["Organism_Label"] == org) &
                    (atlas[SRC_COL].isin(src_list))]
        n   = len(sub)
        mdr = int(sub["MDR_v2"].sum()) if n > 0 else 0
        rows_cns.append({
            "Organism":     org,
            "Source_Group": src_group,
            "Total_N":      n,
            "MDR_Count":    mdr,
            "MDR_Rate_%":   round(mdr/n*100,2) if n > 0 else None
        })

df_cns = pd.DataFrame(rows_cns)
save(df_cns, "FULL_CNS_sensitivity_analysis")

# ============================================================
# STEP 10  MDR LADDER BY WHO REGION
# ============================================================
print("\n" + "="*65)
print("STEP 10 — MDR LADDER BY WHO REGION")
print("="*65)

rows_who = []
for region in ["AFRO","AMRO","EMRO","EURO","SEARO","WPRO","Unclassified"]:
    for org in TARGET_ORGANISMS:
        sub = atlas[(atlas["WHO_Region"] == region) &
                    (atlas["Organism_Label"] == org)]
        n = len(sub)
        if n == 0:
            continue
        gram_type = "Gram-Positive" if org in GRAM_POSITIVE else "Gram-Negative"
        for step in range(5):
            cnt = int((sub["Ladder_Step"] == step).sum())
            rows_who.append({
                "WHO_Region":  region,
                "Organism":    org,
                "Gram_Type":   gram_type,
                "Ladder_Step": step,
                "Count":       cnt,
                "Percent":     round(cnt/n*100,2),
                "Total_N":     n,
                "Sufficient":  "YES" if n >= MIN_ISOLATES else "NO"
            })

df_who = pd.DataFrame(rows_who)
save(df_who, "FULL_MDR_ladder_WHO_region")

# ============================================================
# STEP 11  MDR LADDER TREND 2004–2024
# ============================================================
print("\n" + "="*65)
print("STEP 11 — MDR LADDER TREND 2004-2024")
print("="*65)

rows_lt = []
for org in TARGET_ORGANISMS:
    for year in sorted(atlas[YR_COL].dropna().unique()):
        sub = atlas[(atlas["Organism_Label"] == org) &
                    (atlas[YR_COL] == year)]
        n = len(sub)
        if n == 0:
            continue
        for step in range(5):
            cnt = int((sub["Ladder_Step"] == step).sum())
            rows_lt.append({
                "Organism":    org,
                "Year":        int(year),
                "Ladder_Step": step,
                "Count":       cnt,
                "Percent":     round(cnt/n*100,2),
                "Total_N":     n,
                "Sufficient":  "YES" if n >= MIN_ISOLATES else "NO"
            })

df_lt = pd.DataFrame(rows_lt)
save(df_lt, "FULL_MDR_ladder_trend_2004_2024")

# ============================================================
# STEP 12  WHO REGION × YEAR COMBINED
# ============================================================
print("\n" + "="*65)
print("STEP 12 — WHO REGION × YEAR COMBINED MDR RATES")
print("="*65)

rows_ry = []
for region in ["AFRO","AMRO","EMRO","EURO","SEARO","WPRO"]:
    for year in sorted(atlas[YR_COL].dropna().unique()):
        sub = atlas[(atlas["WHO_Region"] == region) &
                    (atlas[YR_COL] == year)]
        n   = len(sub)
        mdr = int(sub["MDR_v2"].sum()) if n > 0 else 0
        xdr = int(sub["XDR"].sum())    if n > 0 else 0
        rows_ry.append({
            "WHO_Region":  region,
            "Year":        int(year),
            "Total_N":     n,
            "MDR_Count":   mdr,
            "XDR_Count":   xdr,
            "MDR_Rate_%":  round(mdr/n*100,2) if n > 0 else None,
            "XDR_Rate_%":  round(xdr/n*100,2) if n > 0 else None,
            "Sufficient":  "YES" if n >= MIN_ISOLATES else "NO"
        })

df_ry = pd.DataFrame(rows_ry)
save(df_ry, "FULL_WHO_region_year_MDR_rates")

# ============================================================
# STEP 13  ANTIBIOTIC COVERAGE SUMMARY
# ============================================================
print("\n" + "="*65)
print("STEP 13 — FULL ANTIBIOTIC COVERAGE SUMMARY")
print("="*65)

rows_cov = []
for col in interp_cols:
    abx = col.replace("_I","")
    r   = int((atlas[col] == "Resistant").sum())
    s   = int((atlas[col] == "Susceptible").sum())
    i   = int((atlas[col] == "Intermediate").sum())
    na  = int(atlas[col].isna().sum())
    tot = r + s + i
    rows_cov.append({
        "Antibiotic":      abx,
        "Interpretation_Col": col,
        "Susceptible":     s,
        "Intermediate":    i,
        "Resistant":       r,
        "NA":              na,
        "Total_Tested":    tot,
        "Coverage_%":      round(tot/len(atlas)*100,2),
        "Resistance_Rate_%": round(r/tot*100,2) if tot > 0 else None,
        "In_MDR_Classes":  "YES" if any(col in v for v in MDR_CLASSES.values()) else "NO"
    })

df_cov = pd.DataFrame(rows_cov)
save(df_cov, "FULL_antibiotic_coverage_summary")

# ============================================================
# STEP 14  CONFIRMED COUNTS TABLE
# ============================================================
print("\n" + "="*65)
print("STEP 14 — CONFIRMED COUNTS TABLE")
print("="*65)

rows_conf = []
for org in TARGET_ORGANISMS:
    sub = atlas[atlas["Organism_Label"] == org]
    ped = sub[sub["Age_Category"] == "Pediatric"]
    eld = sub[sub["Age_Category"] == "Elderly"]
    rows_conf.append({
        "Organism":            org,
        "Gram_Type":           "Gram-Positive" if org in GRAM_POSITIVE else "Gram-Negative",
        "Total_Isolates":      len(sub),
        "Pediatric_N":         len(ped),
        "Elderly_N":           len(eld),
        "Countries_Covered":   sub[CTRY_COL].nunique(),
        "Years_Covered":       f"{int(sub[YR_COL].min())}-{int(sub[YR_COL].max())}",
        "MDR_Count":           int(sub["MDR_v2"].sum()),
        "XDR_Count":           int(sub["XDR"].sum()),
        "PDR_Count":           int(sub["PDR"].sum()),
        "MDR_Rate_%":          round(sub["MDR_v2"].mean()*100,2),
        "XDR_Rate_%":          round(sub["XDR"].mean()*100,2),
        "PDR_Rate_%":          round(sub["PDR"].mean()*100,2),
        "Blood_N":             (sub[SRC_COL]=="Blood").sum(),
        "CSF_N":               (sub[SRC_COL]=="CSF").sum(),
        "Brain_SpinalCord_N":  sub[SRC_COL].isin(["Brain","Spinal Cord","CNS: Other"]).sum()
    })

df_conf = pd.DataFrame(rows_conf)
save(df_conf, "FULL_confirmed_counts_table")

# ============================================================
# STEP 15  GEARS SUPPLEMENTARY SUMMARY (Option A)
# Labeled clearly as GEARS-derived
# ============================================================
print("\n" + "="*65)
print("STEP 15 — GEARS SUPPLEMENTARY SUMMARY")
print("="*65)

gears_interp = [c for c in gears.columns if c.endswith("_I") and
                c not in ["TIM_I"]]

# GEARS MDR/XDR/PDR — recompute with correct definition
GEARS_MDR_CLASSES = {
    "Penicillins":      ["TZP_I"],
    "Cephalosporins":   ["CAZ_I","C_I","FEP_I"],
    "Carbapenems":      ["IPM_I","MEM_I"],
    "Fluoroquinolones": ["CIP_I","LVX_I"],
    "Aminoglycosides":  ["GM_I"],
    "Polymyxins":       ["CL_I"],
    "Tetracyclines":    ["MI_I"],
    "Trimethoprim_Sulfonamides": ["SXT_I"]
}
GEARS_TOTAL_CLASSES = len(GEARS_MDR_CLASSES)
gears_cols = set(gears.columns)

def compute_mxp_gears(row):
    ns_classes = []
    skipped    = []
    for cls, drugs in GEARS_MDR_CLASSES.items():
        available = [d for d in drugs if d in gears_cols]
        if not available:
            skipped.append(cls); continue
        vals = [row[d] for d in available if pd.notna(row[d])]
        if not vals:
            skipped.append(cls); continue
        if any(v in ["Resistant","Intermediate"] for v in vals):
            ns_classes.append(cls)
    tested = GEARS_TOTAL_CLASSES - len(skipped)
    ns     = len(ns_classes)
    s      = tested - ns
    mdr = 1 if ns >= 3 else 0
    xdr = 1 if (tested > 0 and s <= 2 and ns >= 1 and mdr == 1) else 0
    pdr = 1 if (tested > 0 and s == 0 and ns > 0) else 0
    return mdr, xdr, pdr, ns

gears_mxp = gears.apply(compute_mxp_gears, axis=1)
gears["MDR_v2"] = gears_mxp.apply(lambda x: x[0])
gears["XDR"]    = gears_mxp.apply(lambda x: x[1])
gears["PDR"]    = gears_mxp.apply(lambda x: x[2])
gears["NS_Class_Count"] = gears_mxp.apply(lambda x: x[3])

# GEARS resistance summary by organism and age
rows_g = []
for org in TARGET_ORGANISMS:
    for age_cat in ["Pediatric","Elderly","All"]:
        if age_cat == "All":
            sub = gears[gears["Organism_Label"] == org]
        else:
            sub = gears[(gears["Organism_Label"] == org) &
                        (gears["Age_Category"] == age_cat)]
        n = len(sub)
        if n == 0:
            continue
        mdr = int(sub["MDR_v2"].sum())
        xdr = int(sub["XDR"].sum())
        row = {
            "Dataset":      "GEARS",
            "Organism":     org,
            "Age_Group":    age_cat,
            "Total_N":      n,
            "MDR_Count":    mdr,
            "XDR_Count":    xdr,
            "MDR_Rate_%":   round(mdr/n*100,2) if n > 0 else None,
            "XDR_Rate_%":   round(xdr/n*100,2) if n > 0 else None,
            "Years":        "2018-2022",
            "Note":         "GEARS-derived. 8 of 15 MDR classes available."
        }
        for col in gears_interp:
            if col in gears.columns:
                abx  = col.replace("_I","")
                r    = int((sub[col] == "Resistant").sum())
                s    = int((sub[col] == "Susceptible").sum())
                i    = int((sub[col] == "Intermediate").sum())
                rate = resistance_rate(r, s, i)
                row[f"{abx}_Rate_%"] = rate
        rows_g.append(row)

df_gears_sum = pd.DataFrame(rows_g)
save(df_gears_sum, "GEARS_supplementary_MDR_summary")

# GEARS country × organism
rows_gcty = []
for country in sorted(gears["Country"].dropna().unique()):
    for org in TARGET_ORGANISMS:
        sub = gears[(gears["Country"] == country) &
                    (gears["Organism_Label"] == org)]
        n = len(sub)
        if n == 0:
            continue
        mdr = int(sub["MDR_v2"].sum())
        rows_gcty.append({
            "Dataset":      "GEARS",
            "Country":      country,
            "WHO_Region":   WHO_REGION_MAP.get(country,"Unclassified"),
            "Organism":     org,
            "Total_N":      n,
            "MDR_Count":    mdr,
            "MDR_Rate_%":   round(mdr/n*100,2) if n > 0 else None,
            "Sufficient":   "YES" if n >= MIN_ISOLATES else "NO"
        })

df_gcty = pd.DataFrame(rows_gcty)
save(df_gcty, "GEARS_country_organism_MDR")

# ============================================================
# FINAL SUMMARY PRINT
# ============================================================
print("\n" + "="*65)
print("PHASE 2 COMPLETE — ALL OUTPUTS")
print("="*65)
print(f"""
  ATLAS working dataset : {len(atlas):,} isolates
  MDR (R+I, corrected)  : {atlas['MDR_v2'].sum():,} ({atlas['MDR_v2'].mean()*100:.1f}%)
  XDR                   : {atlas['XDR'].sum():,} ({atlas['XDR'].mean()*100:.1f}%)
  PDR                   : {atlas['PDR'].sum():,} ({atlas['PDR'].mean()*100:.1f}%)

  Files saved to: phase2_outputs/
   01. FULL_resistance_all_antibiotics_wide.csv
   02. FULL_resistance_all_antibiotics_long.csv
   03. FULL_resistance_pediatric_vs_elderly_long.csv
   04. FULL_resistance_pediatric_vs_elderly_wide.csv
   05. FULL_resistance_trends_2004_2024.csv
   06. FULL_MDR_XDR_PDR_burden_organism_age.csv
   07. FULL_MDR_ladder_organism_age.csv
   08. FULL_country_organism_heatmap_data.csv
   09. FULL_CNS_sensitivity_analysis.csv
   10. FULL_MDR_ladder_WHO_region.csv
   11. FULL_MDR_ladder_trend_2004_2024.csv
   12. FULL_WHO_region_year_MDR_rates.csv
   13. FULL_antibiotic_coverage_summary.csv
   14. FULL_confirmed_counts_table.csv
   15. GEARS_supplementary_MDR_summary.csv
   16. GEARS_country_organism_MDR.csv
""")
