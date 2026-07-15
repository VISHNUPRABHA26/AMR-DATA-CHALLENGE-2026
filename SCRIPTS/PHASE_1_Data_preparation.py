# ============================================================
# AMR DATA CHALLENGE 2026 - PHASE 1: DATA PREPARATION
# Nosocomial Neuroinvasive Bacterial AMR Study
# ============================================================
# REFERENCE: MDR definition - Magiorakos et al. 2012,
#   Clinical Microbiology and Infection (ECDC/CDC standard)
# EUCAST 2024 breakpoints for GEARS MIC conversion
# ============================================================
# OUTPUT FILES (saved to phase1_outputs/):
#   atlas_filtered.csv        — ATLAS filtered dataset
#   gears_filtered.csv        — GEARS filtered dataset
#   atlas_mdr_flagged.csv     — ATLAS with MDR flag
#   gears_mdr_flagged.csv     — GEARS with MDR flag
#   phase1_summary.csv        — Summary counts of all steps
#   mdr_class_counts.csv      — Per-isolate class resistance counts
#   gears_breakpoint_log.csv  — EUCAST conversion log for GEARS
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS — adjust if filenames differ
# ============================================================
ATLAS_PATH  = "atlas_vivli_2004_2024.csv"
GEARS_PATH  = "Venatorx surveillance data_2024_06_06.xlsx"
OUT_DIR     = "phase1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save(df, name):
    path = os.path.join(OUT_DIR, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df):,} rows)")

# ============================================================
# STUDY PARAMETERS — confirmed by Vishnu Prabha
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

# Direct CNS + Bloodstream sources only (confirmed)
CNS_BLOOD_SOURCES_ATLAS = [
    "Blood",
    "CSF",
    "Brain",
    "CNS: Other",
    "Spinal Cord",
    "Head",
    "Peripheral Nerves"
]

# GEARS equivalent — CVS: Blood only (no CSF/Brain in GEARS confirmed from profile)
CNS_BLOOD_SOURCES_GEARS = [
    "CVS: Blood"
]

# Age groups (exact strings from ATLAS)
ATLAS_PEDIATRIC = ["0 - 17"]
ATLAS_ELDERLY   = ["61+"]

# ============================================================
# MDR CLASS MAP  
# Only antibiotics with <90% missing data included
# ============================================================
MDR_CLASSES = {
    "Penicillins": [
        "Ampicillin_I",
        "Amoxycillin clavulanate_I",
        "Ampicillin sulbactam_I",
        "Piperacillin tazobactam_I",
        "Penicillin_I"
    ],
    "Cephalosporins": [
        "Cefepime_I",
        "Ceftazidime_I",
        "Ceftriaxone_I",
        "Ceftaroline_I",
        "Ceftazidime avibactam_I",
        "Ceftolozane tazobactam_I"
    ],
    "Monobactams": [
        "Aztreonam_I"
    ],
    "Carbapenems": [
        "Meropenem_I",
        "Imipenem_I",
        "Doripenem_I"
    ],
    "Fluoroquinolones": [
        "Levofloxacin_I",
        "Ciprofloxacin_I"
    ],
    "Aminoglycosides": [
        "Amikacin_I",
        "Gentamicin_I"
    ],
    "Glycopeptides": [
        "Vancomycin_I",
        "Teicoplanin_I"
    ],
    "Polymyxins": [
        "Colistin_I"
    ],
    "Tetracyclines": [
        "Minocycline_I",
        "Tigecycline_I"
    ],
    "Trimethoprim_Sulfonamides": [
        "Trimethoprim sulfa_I"
    ],
    "Oxazolidinones": [
        "Linezolid_I"
    ],
    "Lipopeptides": [
        "Daptomycin_I"
    ],
    "Penicillinase_Resistant_Penicillins": [
        "Oxacillin_I"
    ],
    "Lincosamides": [
        "Clindamycin_I"
    ],
    "Macrolides": [
        "Erythromycin_I"
    ]
}

# ============================================================
# EUCAST 2024 BREAKPOINTS FOR GEARS MIC CONVERSION
# Source: EUCAST Clinical Breakpoint Tables v14.0, 2024
# Format: {organism_keyword: {drug_col: (S<=, R>)}}
# S <= breakpoint = Susceptible
# R >  breakpoint = Resistant
# Between S and R = Intermediate
# "None" means no breakpoint defined for that combination
# ============================================================
# GEARS MIC column mapping to antibiotic names:
# CAZ_MIC  = Ceftazidime
# C_MIC    = Ceftazidime avibactam (C/A in GEARS context — see note)
# CIP_MIC  = Ciprofloxacin
# CL_MIC   = Colistin
# FEP_MIC  = Cefepime
# GM_MIC   = Gentamicin
# IPM_MIC  = Imipenem
# LVX_MIC  = Levofloxacin
# MEM_MIC  = Meropenem
# MI_MIC   = Minocycline
# SXT_MIC  = Trimethoprim-sulfamethoxazole
# TIM_MIC  = Ticarcillin-clavulanate (not in MDR classes — no breakpoint applied)
# TZP_MIC  = Piperacillin-tazobactam
#
# NOTE on C_MIC: In Venatorx GEARS, C_MIC = ceftazidime-avibactam.
# EUCAST 2024 breakpoint for ceftazidime-avibactam Enterobacterales: S<=8, R>8
# ============================================================

EUCAST_BREAKPOINTS = {
    # ---- Enterobacterales (E.coli, Klebsiella, Enterobacter,
    #       Serratia, Citrobacter, Salmonella, Proteus) ----
    "Enterobacterales": {
        "organisms": [
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Enterobacter cloacae",
            "Serratia marcescens",
            "Citrobacter koseri",
            "Salmonella spp"
        ],
        "breakpoints": {
            "CAZ_MIC": (1,   4),     # Ceftazidime S<=1, R>4
            "C_MIC":   (8,   8),     # Ceftazidime-avibactam S<=8, R>8
            "CIP_MIC": (0.25, 0.5),  # Ciprofloxacin S<=0.25, R>0.5
            "CL_MIC":  (2,   2),     # Colistin S<=2, R>2
            "FEP_MIC": (1,   4),     # Cefepime S<=1, R>4
            "GM_MIC":  (2,   4),     # Gentamicin S<=2, R>4
            "IPM_MIC": (2,   8),     # Imipenem S<=2, R>8
            "LVX_MIC": (1,   2),     # Levofloxacin S<=1, R>2
            "MEM_MIC": (2,   8),     # Meropenem S<=2, R>8
            "MI_MIC":  (2,   8),     # Minocycline S<=2, R>8
            "SXT_MIC": (2,   4),     # Trimethoprim-sulfa S<=2, R>4
            "TZP_MIC": (8,  16),     # Piperacillin-tazobactam S<=8, R>16
            "TIM_MIC": (None, None), # Not in MDR classes — skip
        }
    },
    # ---- Pseudomonas aeruginosa ----
    "Pseudomonas aeruginosa": {
        "organisms": ["Pseudomonas aeruginosa"],
        "breakpoints": {
            "CAZ_MIC": (8,   8),
            "C_MIC":   (8,   8),
            "CIP_MIC": (0.5, 1),
            "CL_MIC":  (2,   2),
            "FEP_MIC": (8,   8),
            "GM_MIC":  (4,   4),
            "IPM_MIC": (4,   4),
            "LVX_MIC": (1,   2),
            "MEM_MIC": (2,   8),
            "MI_MIC":  (None, None),
            "SXT_MIC": (None, None),
            "TZP_MIC": (16,  16),
            "TIM_MIC": (None, None),
        }
    },
    # ---- Acinetobacter baumannii ----
    "Acinetobacter baumannii": {
        "organisms": ["Acinetobacter baumannii"],
        "breakpoints": {
            "CAZ_MIC": (8,   8),
            "C_MIC":   (None, None),
            "CIP_MIC": (1,   1),
            "CL_MIC":  (2,   2),
            "FEP_MIC": (8,   8),
            "GM_MIC":  (4,   4),
            "IPM_MIC": (2,   8),
            "LVX_MIC": (1,   2),
            "MEM_MIC": (2,   8),
            "MI_MIC":  (2,   8),
            "SXT_MIC": (2,   4),
            "TZP_MIC": (None, None),
            "TIM_MIC": (None, None),
        }
    },
    # ---- Staphylococcus aureus ----
    "Staphylococcus aureus": {
        "organisms": ["Staphylococcus aureus", "Staphylococcus aureus, MSSA"],
        "breakpoints": {
            "CAZ_MIC": (None, None),
            "C_MIC":   (None, None),
            "CIP_MIC": (1,   1),
            "CL_MIC":  (None, None),
            "FEP_MIC": (None, None),
            "GM_MIC":  (1,   1),
            "IPM_MIC": (None, None),
            "LVX_MIC": (1,   2),
            "MEM_MIC": (None, None),
            "MI_MIC":  (0.5, 1),
            "SXT_MIC": (2,   4),
            "TZP_MIC": (None, None),
            "TIM_MIC": (None, None),
        }
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def flag_mdr(row, class_map, df_columns):
    """
    For one isolate (row), count how many antibiotic classes
    it is Resistant to.
    Rules:
      - If ANY drug in a class = 'Resistant' → class is Resistant
      - If all drugs in class are NA → class is skipped (not counted)
      - Missing values are skipped, not counted as resistant
    Returns: (mdr_flag, resistant_class_count, resistant_classes_list)
    """
    resistant_classes = []
    for class_name, drugs in class_map.items():
        # Only use drugs that exist in this dataframe
        available = [d for d in drugs if d in df_columns]
        if not available:
            continue
        values = [row[d] for d in available if pd.notna(row[d]) and row[d] != ""]
        if not values:
            # All NA for this class — skip
            continue
        if any(v == "Resistant" for v in values):
            resistant_classes.append(class_name)
    count = len(resistant_classes)
    mdr   = 1 if count >= 3 else 0
    return mdr, count, ";".join(resistant_classes)

def apply_eucast(row, breakpoints_dict):
    """
    Convert MIC values to S/I/R for one GEARS row
    using EUCAST 2024 breakpoints.
    Returns a dict of {col_name: SIR_label}
    """
    organism = str(row.get("Organism", ""))
    result   = {}

    # Identify which breakpoint group applies
    bp_group = None
    for group_name, group_data in breakpoints_dict.items():
        if any(org.lower() in organism.lower()
               for org in group_data["organisms"]):
            bp_group = group_data["breakpoints"]
            break

    if bp_group is None:
        # No breakpoint defined for this organism
        for col in ["CAZ_MIC","C_MIC","CIP_MIC","CL_MIC","FEP_MIC",
                    "GM_MIC","IPM_MIC","LVX_MIC","MEM_MIC","MI_MIC",
                    "SXT_MIC","TIM_MIC","TZP_MIC"]:
            result[col.replace("_MIC","_I")] = np.nan
        return result

    for mic_col, (s_bp, r_bp) in bp_group.items():
        sir_col = mic_col.replace("_MIC", "_I")
        if s_bp is None:
            result[sir_col] = np.nan
            continue
        mic_val = pd.to_numeric(row.get(mic_col, np.nan), errors='coerce')
        if pd.isna(mic_val):
            result[sir_col] = np.nan
        elif mic_val <= s_bp:
            result[sir_col] = "Susceptible"
        elif mic_val > r_bp:
            result[sir_col] = "Resistant"
        else:
            result[sir_col] = "Intermediate"

    return result

# ============================================================
# SECTION 1  ATLAS PROCESSING
# ============================================================
print("\n" + "="*65)
print("STEP 1 — LOADING ATLAS")
print("="*65)

atlas = pd.read_csv(ATLAS_PATH, low_memory=False,
                    na_values=["", "NA", "-", " ", "N/A"])
print(f"  Loaded: {atlas.shape[0]:,} rows x {atlas.shape[1]} columns")

# Resolve column names
AGE_COL  = resolve_col(atlas, ["Age Group", "Age.Group"])
SRC_COL  = resolve_col(atlas, ["Source"])
SP_COL   = resolve_col(atlas, ["Species"])
CTRY_COL = resolve_col(atlas, ["Country"])
YR_COL   = resolve_col(atlas, ["Year"])
SPEC_COL = resolve_col(atlas, ["Speciality", "Specialty"])

print(f"  Age col: '{AGE_COL}' | Source col: '{SRC_COL}' | Species col: '{SP_COL}'")

summary_rows = []

# ---- Step 1a: Filter organisms ----
print("\nSTEP 1a — Filtering to 13 target organisms")
org_mask = atlas[SP_COL].apply(
    lambda x: any(org.lower() in str(x).lower()
                  for org in TARGET_ORGANISMS)
)
atlas_org = atlas[org_mask].copy()
print(f"  Rows after organism filter: {len(atlas_org):,}")
summary_rows.append({
    "Step": "1a_organism_filter",
    "Dataset": "ATLAS",
    "Description": "Filter to 13 target organisms",
    "Rows_Remaining": len(atlas_org)
})

# ---- Step 1b: Filter age groups ----
print("\nSTEP 1b — Filtering to Pediatric (0-17) and Elderly (61+)")
age_mask = atlas_org[AGE_COL].isin(ATLAS_PEDIATRIC + ATLAS_ELDERLY)
atlas_age = atlas_org[age_mask].copy()
atlas_age["Age_Category"] = atlas_age[AGE_COL].apply(
    lambda x: "Pediatric" if x in ATLAS_PEDIATRIC else "Elderly"
)
print(f"  Rows after age filter: {len(atlas_age):,}")
print(f"    Pediatric (0-17): {(atlas_age['Age_Category']=='Pediatric').sum():,}")
print(f"    Elderly   (61+) : {(atlas_age['Age_Category']=='Elderly').sum():,}")
summary_rows.append({
    "Step": "1b_age_filter",
    "Dataset": "ATLAS",
    "Description": "Filter to Pediatric(0-17) and Elderly(61+)",
    "Rows_Remaining": len(atlas_age)
})

# ---- Step 1c: Filter specimen sources ----
print("\nSTEP 1c — Filtering to CNS/Blood specimen sources")
src_mask   = atlas_age[SRC_COL].isin(CNS_BLOOD_SOURCES_ATLAS)
atlas_src  = atlas_age[src_mask].copy()
print(f"  Rows after source filter: {len(atlas_src):,}")
print("  Source breakdown:")
for src, cnt in atlas_src[SRC_COL].value_counts().items():
    print(f"    {src:<30} : {cnt:,}")
summary_rows.append({
    "Step": "1c_source_filter",
    "Dataset": "ATLAS",
    "Description": "Filter to CNS+Blood sources only",
    "Rows_Remaining": len(atlas_src)
})

# ---- Step 1d: Drop rows where ALL antibiotic columns are NA ----
print("\nSTEP 1d — Dropping rows with all antibiotic values missing")
all_abx_cols = [c for c in atlas_src.columns if c.endswith("_I")]
before = len(atlas_src)
atlas_clean = atlas_src.dropna(subset=all_abx_cols, how="all").copy()
dropped = before - len(atlas_clean)
print(f"  Dropped {dropped:,} rows (all antibiotics NA)")
print(f"  Rows remaining: {len(atlas_clean):,}")
summary_rows.append({
    "Step": "1d_drop_all_na_antibiotics",
    "Dataset": "ATLAS",
    "Description": "Drop rows where ALL antibiotic cols are NA",
    "Rows_Remaining": len(atlas_clean)
})

# ---- Step 2: MDR Flag Engineering for ATLAS ----
print("\nSTEP 2 — MDR FLAG ENGINEERING (ATLAS)")
print("  Applying Magiorakos 2012 MDR definition (15 classes)")
print("  Rule: Resistant to ANY drug in a class = class counted")
print("  Rule: Missing data = skipped (not counted as resistant)")
print("  Threshold: MDR if resistant to >= 3 classes")
print("  Processing... (may take a few minutes for large dataset)")

atlas_cols = set(atlas_clean.columns)
mdr_results = atlas_clean.apply(
    lambda row: flag_mdr(row, MDR_CLASSES, atlas_cols), axis=1
)

atlas_clean["MDR_Flag"]              = mdr_results.apply(lambda x: x[0])
atlas_clean["Resistant_Class_Count"] = mdr_results.apply(lambda x: x[1])
atlas_clean["Resistant_Classes"]     = mdr_results.apply(lambda x: x[2])

mdr_count = atlas_clean["MDR_Flag"].sum()
print(f"\n  MDR isolates     : {mdr_count:,}")
print(f"  Non-MDR isolates : {(atlas_clean['MDR_Flag']==0).sum():,}")
print(f"  MDR rate         : {mdr_count/len(atlas_clean)*100:.1f}%")

print("\n  MDR by organism:")
for org in TARGET_ORGANISMS:
    sub = atlas_clean[atlas_clean[SP_COL].str.contains(org, case=False, na=False)]
    if len(sub) > 0:
        m = sub["MDR_Flag"].sum()
        print(f"    {org:<45} : {m:,}/{len(sub):,} ({m/len(sub)*100:.1f}%)")

print("\n  MDR by age group:")
for cat in ["Pediatric", "Elderly"]:
    sub = atlas_clean[atlas_clean["Age_Category"] == cat]
    if len(sub) > 0:
        m = sub["MDR_Flag"].sum()
        print(f"    {cat:<15} : {m:,}/{len(sub):,} ({m/len(sub)*100:.1f}%)")

summary_rows.append({
    "Step": "2_mdr_flag",
    "Dataset": "ATLAS",
    "Description": "MDR flag engineered (>=3 resistant classes)",
    "Rows_Remaining": len(atlas_clean)
})

# Save ATLAS outputs
print("\n  Saving ATLAS outputs...")
save(atlas_clean, "atlas_filtered_mdr_flagged")

# MDR class breakdown per isolate (separate file for reference)
mdr_class_df = atlas_clean[
    ["Isolate Id" if "Isolate Id" in atlas_clean.columns else atlas_clean.columns[0],
     SP_COL, CTRY_COL, YR_COL, AGE_COL, "Age_Category",
     "MDR_Flag", "Resistant_Class_Count", "Resistant_Classes"]
].copy()
save(mdr_class_df, "atlas_mdr_class_breakdown")

# ============================================================
# SECTION 2  GEARS PROCESSING
# ============================================================
print("\n" + "="*65)
print("STEP 3 — LOADING GEARS")
print("="*65)

gears = pd.read_excel(GEARS_PATH, sheet_name=0)
gears = gears.where(pd.notnull(gears), other=np.nan)
print(f"  Loaded: {gears.shape[0]:,} rows x {gears.shape[1]} columns")

# ---- Step 3a: Filter organisms ----
print("\nSTEP 3a — Filtering GEARS to 13 target organisms")
gears_org_mask = gears["Organism"].apply(
    lambda x: any(org.lower() in str(x).lower()
                  for org in TARGET_ORGANISMS)
)
gears_org = gears[gears_org_mask].copy()
print(f"  Rows after organism filter: {len(gears_org):,}")
print("  Organisms found:")
for org, cnt in gears_org["Organism"].value_counts().items():
    print(f"    {org:<50}: {cnt:,}")
summary_rows.append({
    "Step": "3a_organism_filter",
    "Dataset": "GEARS",
    "Description": "Filter to 13 target organisms",
    "Rows_Remaining": len(gears_org)
})

# ---- Step 3b: Filter age groups ----
print("\nSTEP 3b — Filtering GEARS to Pediatric (0-17) and Elderly (61+)")
ages_g     = pd.to_numeric(gears_org["Age"], errors='coerce')
age_mask_g = (ages_g <= 17) | (ages_g >= 61)
gears_age  = gears_org[age_mask_g].copy()
ages_filt  = pd.to_numeric(gears_age["Age"], errors='coerce')
gears_age["Age_Category"] = ages_filt.apply(
    lambda x: "Pediatric" if x <= 17 else "Elderly"
)
print(f"  Rows after age filter: {len(gears_age):,}")
print(f"    Pediatric (0-17): {(gears_age['Age_Category']=='Pediatric').sum():,}")
print(f"    Elderly   (61+) : {(gears_age['Age_Category']=='Elderly').sum():,}")
summary_rows.append({
    "Step": "3b_age_filter",
    "Dataset": "GEARS",
    "Description": "Filter to Pediatric(0-17) and Elderly(61+)",
    "Rows_Remaining": len(gears_age)
})

# ---- Step 3c: Filter body sites ----
print("\nSTEP 3c — Filtering GEARS to CVS: Blood only")
src_mask_g  = gears_age["BodySite"].isin(CNS_BLOOD_SOURCES_GEARS)
gears_src   = gears_age[src_mask_g].copy()
print(f"  Rows after source filter: {len(gears_src):,}")
print("  BodySite breakdown:")
for bs, cnt in gears_src["BodySite"].value_counts().items():
    print(f"    {bs:<40}: {cnt:,}")
summary_rows.append({
    "Step": "3c_source_filter",
    "Dataset": "GEARS",
    "Description": "Filter to CVS: Blood only",
    "Rows_Remaining": len(gears_src)
})

# ---- Step 4: Apply EUCAST 2024 Breakpoints to GEARS ----
print("\nSTEP 4 — APPLYING EUCAST 2024 BREAKPOINTS TO GEARS MIC VALUES")
print("  Converting MIC -> S/I/R for each organism-drug combination")
print("  Reference: EUCAST Clinical Breakpoint Tables v14.0, 2024")
print("  Processing...")

sir_results = gears_src.apply(
    lambda row: apply_eucast(row, EUCAST_BREAKPOINTS), axis=1
)
sir_df = pd.DataFrame(sir_results.tolist(), index=gears_src.index)
gears_sir = pd.concat([gears_src, sir_df], axis=1)

# Log how many values were converted per column
print("\n  Conversion results per antibiotic:")
bp_log_rows = []
mic_to_sir = {
    "CAZ_MIC":"CAZ_I","C_MIC":"C_I","CIP_MIC":"CIP_I",
    "CL_MIC":"CL_I","FEP_MIC":"FEP_I","GM_MIC":"GM_I",
    "IPM_MIC":"IPM_I","LVX_MIC":"LVX_I","MEM_MIC":"MEM_I",
    "MI_MIC":"MI_I","SXT_MIC":"SXT_I","TIM_MIC":"TIM_I",
    "TZP_MIC":"TZP_I"
}
for mic_col, sir_col in mic_to_sir.items():
    if sir_col in gears_sir.columns:
        vc = gears_sir[sir_col].value_counts(dropna=True)
        s  = int(vc.get("Susceptible",  0))
        i  = int(vc.get("Intermediate", 0))
        r  = int(vc.get("Resistant",    0))
        na = int(gears_sir[sir_col].isna().sum())
        tot = s + i + r
        rpct = round(r/tot*100,1) if tot > 0 else None
        print(f"    {sir_col:<8}: S={s:,} I={i:,} R={r:,} NA={na:,}  R%={rpct}")
        bp_log_rows.append({
            "MIC_Column": mic_col,
            "SIR_Column": sir_col,
            "Susceptible": s,
            "Intermediate": i,
            "Resistant": r,
            "NA": na,
            "Resistance_Rate_Pct": rpct
        })

save(pd.DataFrame(bp_log_rows), "gears_eucast_breakpoint_log")

# ---- Step 5: MDR Flag for GEARS ----
print("\nSTEP 5 — MDR FLAG ENGINEERING (GEARS)")

# Map GEARS SIR column names to MDR class map
GEARS_MDR_CLASSES = {
    "Penicillins":      ["TZP_I"],
    "Cephalosporins":   ["CAZ_I", "C_I", "FEP_I"],
    "Carbapenems":      ["IPM_I", "MEM_I"],
    "Fluoroquinolones": ["CIP_I", "LVX_I"],
    "Aminoglycosides":  ["GM_I"],
    "Polymyxins":       ["CL_I"],
    "Tetracyclines":    ["MI_I"],
    "Trimethoprim_Sulfonamides": ["SXT_I"]
}
# Note: Monobactams, Glycopeptides, Oxazolidinones, Lipopeptides,
# Penicillinase-resistant penicillins, Lincosamides, Macrolides
# are not present in GEARS MIC panel — only 8 classes available

print("  Note: GEARS has 8 of 15 MDR classes available")
print("  (Remaining 7 classes not in GEARS antibiotic panel)")

gears_cols = set(gears_sir.columns)
mdr_results_g = gears_sir.apply(
    lambda row: flag_mdr(row, GEARS_MDR_CLASSES, gears_cols), axis=1
)
gears_sir["MDR_Flag"]              = mdr_results_g.apply(lambda x: x[0])
gears_sir["Resistant_Class_Count"] = mdr_results_g.apply(lambda x: x[1])
gears_sir["Resistant_Classes"]     = mdr_results_g.apply(lambda x: x[2])

mdr_count_g = gears_sir["MDR_Flag"].sum()
print(f"\n  MDR isolates     : {mdr_count_g:,}")
print(f"  Non-MDR isolates : {(gears_sir['MDR_Flag']==0).sum():,}")
if len(gears_sir) > 0:
    print(f"  MDR rate         : {mdr_count_g/len(gears_sir)*100:.1f}%")

print("\n  MDR by organism (GEARS):")
for org, cnt in gears_sir["Organism"].value_counts().items():
    sub = gears_sir[gears_sir["Organism"] == org]
    m   = sub["MDR_Flag"].sum()
    print(f"    {org:<50}: {m:,}/{len(sub):,} ({m/len(sub)*100:.1f}%)")

summary_rows.append({
    "Step": "5_mdr_flag",
    "Dataset": "GEARS",
    "Description": "MDR flag engineered (>=3 resistant classes, 8 classes available)",
    "Rows_Remaining": len(gears_sir)
})

# Save GEARS outputs
print("\n  Saving GEARS outputs...")
save(gears_sir, "gears_filtered_mdr_flagged")

gears_mdr_class_df = gears_sir[[
    "Isolate", "Organism", "Country", "Year",
    "Age", "Age_Category",
    "MDR_Flag", "Resistant_Class_Count", "Resistant_Classes"
]].copy()
save(gears_mdr_class_df, "gears_mdr_class_breakdown")

# ============================================================
# PHASE 1 SUMMARY
# ============================================================
print("\n" + "="*65)
print("PHASE 1 COMPLETE — SUMMARY")
print("="*65)

summary_df = pd.DataFrame(summary_rows)
save(summary_df, "phase1_summary")

print(f"""
  ATLAS:
    Original dataset        : 1,011,168 rows
    After organism filter   : {atlas_org.shape[0]:,} rows
    After age filter        : {atlas_age.shape[0]:,} rows
    After source filter     : {atlas_src.shape[0]:,} rows
    After drop all-NA abx   : {atlas_clean.shape[0]:,} rows
    MDR isolates            : {atlas_clean['MDR_Flag'].sum():,}
    MDR rate                : {atlas_clean['MDR_Flag'].sum()/len(atlas_clean)*100:.1f}%

  GEARS:
    Original dataset        : 29,365 rows
    After organism filter   : {gears_org.shape[0]:,} rows
    After age filter        : {gears_age.shape[0]:,} rows
    After source filter     : {gears_src.shape[0]:,} rows
    MDR isolates            : {gears_sir['MDR_Flag'].sum():,}
    MDR rate                : {gears_sir['MDR_Flag'].sum()/len(gears_sir)*100:.1f}% (if any rows remain)

  Output files in: phase1_outputs/
    atlas_filtered_mdr_flagged.csv
    atlas_mdr_class_breakdown.csv
    gears_filtered_mdr_flagged.csv
    gears_mdr_class_breakdown.csv
    gears_eucast_breakpoint_log.csv
    phase1_summary.csv
""")

print("Phase 1 complete.")
