# ============================================================
# AMR DATA CHALLENGE 2026 — PHASE 3
# SPATIOTEMPORAL MAP 
#
# WHAT THIS SCRIPT DOES:
#   Step 1 — Generates the temporal CSV from Phase 1 ATLAS data,
#             now including XDR, PDR, and Age_Category columns.
#             XDR/PDR defined per Magiorakos et al. CMI 2012.
#   Step 2 — Builds spatiotemporal_layout.html with:
#             - Organism dropdown  (existing)
#             - Year dropdown      (existing)
#             - Resistance Type dropdown: MDR | XDR | PDR  (NEW)
#             - Age Category dropdown: All | Pediatric | Elderly  (NEW)
#
# DEFINITIONS (Magiorakos et al. CMI 2012):
#   MDR: non-susceptible (R or I) to >= 3 antibiotic classes
#   XDR: non-susceptible to >= 1 agent in all but <= 2 classes
#        (i.e., susceptible to <= 2 classes only)
#   PDR: non-susceptible to all agents in all tested classes
#        (zero susceptible classes)
#
# Age Category (from Phase 1):
#   Pediatric : Age_Category == "Pediatric"  (0–17)
#   Elderly   : Age_Category == "Elderly"    (61+)
#   All       : Pediatric + Elderly pooled (Option A — confirmed)
#
# REFERENCES:
#   Magiorakos et al. CMI 2012;18(3):268-281
#   WHO GLASS 2023 — min 10 isolates threshold
#   ECDC AMR Threat Report 2022 — 50% crisis threshold
#
# INPUTS:
#   PHASE1_outputs/atlas_filtered_mdr_flagged.csv
#
# OUTPUT:
#   phase3_outputs/layout/spatiotemporal_layout.html
#   phase3_outputs/csv_outputs/temporal_mxp_age_country_year.csv
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
import urllib.request
warnings.filterwarnings("ignore")

import geopandas as gpd
import plotly.graph_objects as go

# ============================================================
# PATHS
# ============================================================
ATLAS_PATH = "phase1_outputs/atlas_filtered_mdr_flagged.csv"
OUT_DIR    = "phase3_outputs"
CSV_DIR    = os.path.join(OUT_DIR, "csv_outputs")
LAY_DIR    = os.path.join(OUT_DIR, "layout")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(LAY_DIR, exist_ok=True)

MIN_ISOLATES = 10   # WHO GLASS 2023

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

# Magiorakos 2012 - 15 antibiotic classes
MDR_CLASSES = {
    "Penicillins": [
        "Ampicillin_I", "Amoxycillin clavulanate_I",
        "Ampicillin sulbactam_I", "Piperacillin tazobactam_I",
        "Penicillin_I"],
    "Cephalosporins": [
        "Cefepime_I", "Ceftazidime_I", "Ceftriaxone_I",
        "Ceftaroline_I", "Ceftazidime avibactam_I",
        "Ceftolozane tazobactam_I"],
    "Monobactams":      ["Aztreonam_I"],
    "Carbapenems":      ["Meropenem_I", "Imipenem_I", "Doripenem_I"],
    "Fluoroquinolones": ["Levofloxacin_I", "Ciprofloxacin_I"],
    "Aminoglycosides":  ["Amikacin_I", "Gentamicin_I"],
    "Glycopeptides":    ["Vancomycin_I", "Teicoplanin_I"],
    "Polymyxins":       ["Colistin_I"],
    "Tetracyclines":    ["Minocycline_I", "Tigecycline_I"],
    "Trimethoprim_Sulfonamides": ["Trimethoprim sulfa_I"],
    "Oxazolidinones":   ["Linezolid_I"],
    "Lipopeptides":     ["Daptomycin_I"],
    "Penicillinase_Resistant_Penicillins": ["Oxacillin_I"],
    "Lincosamides":     ["Clindamycin_I"],
    "Macrolides":       ["Erythromycin_I"]
}
TOTAL_CLASSES = len(MDR_CLASSES)  # 15

WORLD_BANK_INCOME = {
    "United States": "High Income", "Canada": "High Income",
    "Germany": "High Income", "France": "High Income",
    "United Kingdom": "High Income", "Italy": "High Income",
    "Spain": "High Income", "Australia": "High Income",
    "Japan": "High Income", "Korea, South": "High Income",
    "Netherlands": "High Income", "Belgium": "High Income",
    "Sweden": "High Income", "Denmark": "High Income",
    "Switzerland": "High Income", "Norway": "High Income",
    "Finland": "High Income", "Austria": "High Income",
    "Ireland": "High Income", "New Zealand": "High Income",
    "Israel": "High Income", "Singapore": "High Income",
    "Hong Kong": "High Income", "Taiwan": "High Income",
    "Czech Republic": "High Income", "Slovenia": "High Income",
    "Slovak Republic": "High Income", "Estonia": "High Income",
    "Lithuania": "High Income", "Latvia": "High Income",
    "Portugal": "High Income", "Greece": "High Income",
    "Hungary": "High Income", "Poland": "High Income",
    "Qatar": "High Income", "Kuwait": "High Income",
    "Saudi Arabia": "High Income", "Oman": "High Income",
    "Puerto Rico": "High Income", "Croatia": "High Income",
    "China": "Upper Middle Income", "Brazil": "Upper Middle Income",
    "Mexico": "Upper Middle Income", "Argentina": "Upper Middle Income",
    "Colombia": "Upper Middle Income", "Turkey": "Upper Middle Income",
    "Russia": "Upper Middle Income", "Malaysia": "Upper Middle Income",
    "South Africa": "Upper Middle Income",
    "Thailand": "Upper Middle Income",
    "Romania": "Upper Middle Income", "Bulgaria": "Upper Middle Income",
    "Serbia": "Upper Middle Income", "Ukraine": "Upper Middle Income",
    "Jordan": "Upper Middle Income", "Lebanon": "Upper Middle Income",
    "Venezuela": "Upper Middle Income", "Chile": "Upper Middle Income",
    "Ecuador": "Upper Middle Income",
    "Dominican Republic": "Upper Middle Income",
    "Costa Rica": "Upper Middle Income",
    "Panama": "Upper Middle Income",
    "Jamaica": "Upper Middle Income",
    "Mauritius": "Upper Middle Income",
    "India": "Lower Middle Income",
    "Philippines": "Lower Middle Income",
    "Vietnam": "Lower Middle Income",
    "Indonesia": "Lower Middle Income",
    "Morocco": "Lower Middle Income",
    "Egypt": "Lower Middle Income",
    "Nigeria": "Lower Middle Income",
    "Kenya": "Lower Middle Income",
    "Ghana": "Lower Middle Income",
    "Cameroon": "Lower Middle Income",
    "Ivory Coast": "Lower Middle Income",
    "Tunisia": "Lower Middle Income",
    "Pakistan": "Lower Middle Income",
    "Honduras": "Lower Middle Income",
    "El Salvador": "Lower Middle Income",
    "Guatemala": "Lower Middle Income",
    "Nicaragua": "Lower Middle Income",
    "Uganda": "Low Income",
    "Malawi": "Low Income",
    "Namibia": "Low Income",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_org_label(sp):
    for org in TARGET_ORGANISMS:
        if org.lower() in str(sp).lower():
            return org
    return None

def get_income(country):
    return WORLD_BANK_INCOME.get(country, "Unclassified")

def compute_mxp(row, df_cols):
    """
    Magiorakos et al. CMI 2012 — MDR, XDR, PDR per isolate.
    Non-susceptible (NS) = Resistant OR Intermediate.
    A class is 'tested' only if >= 1 drug in that class has a non-NA value.
    A class is 'NS' if any drug in that class is NS.
    A class is 'susceptible' if all tested drugs are Susceptible.

    MDR: NS in >= 3 classes
    XDR: susceptible to <= 2 classes (NS in all but <= 2)
    PDR: NS in every tested class (0 susceptible classes)

    Returns: (mdr, xdr, pdr, ns_count, tested_count)
    """
    ns_classes  = []
    sus_classes = []
    skip_classes = []

    for cls, drugs in MDR_CLASSES.items():
        available = [d for d in drugs if d in df_cols]
        if not available:
            skip_classes.append(cls)
            continue
        vals = [row[d] for d in available if pd.notna(row[d])]
        if not vals:
            skip_classes.append(cls)
            continue
        if any(v in ["Resistant", "Intermediate"] for v in vals):
            ns_classes.append(cls)
        else:
            # all tested drugs are Susceptible
            sus_classes.append(cls)

    ns    = len(ns_classes)
    sus   = len(sus_classes)
    tested = ns + sus

    mdr = 1 if ns >= 3 else 0
    # XDR: susceptible to <= 2 classes AND is MDR (tested > 0)
    xdr = 1 if (mdr == 1 and tested > 0 and sus <= 2) else 0
    # PDR: no susceptible classes at all (tested > 0)
    pdr = 1 if (tested > 0 and sus == 0 and ns > 0) else 0

    return mdr, xdr, pdr, ns, tested

# ============================================================
# STEP 1  LOAD ATLAS AND COMPUTE MDR/XDR/PDR PER ISOLATE
# ============================================================
print("=" * 65)
print("STEP 1 — LOADING PHASE 1 ATLAS DATA")
print("=" * 65)

atlas = pd.read_csv(ATLAS_PATH, low_memory=False)
print(f"  Loaded: {len(atlas):,} rows")

# Normalize column names: dots -> spaces (ATLAS uses spaces)
atlas.columns = [c.replace(".", " ").strip() for c in atlas.columns]

# Check required columns
required = ["Species", "Country", "Year", "Age_Category"]
missing  = [c for c in required if c not in atlas.columns]
if missing:
    print(f"  ERROR: Missing columns: {missing}")
    print(f"  Available columns (first 30): {list(atlas.columns[:30])}")
    raise SystemExit("Cannot proceed — fix column names")

atlas["Organism_Label"] = atlas["Species"].apply(get_org_label)
atlas = atlas[atlas["Organism_Label"].notna()].copy()
print(f"  Rows with target organisms: {len(atlas):,}")

# Verify Age_Category values
print(f"  Age_Category values: {atlas['Age_Category'].unique()}")

# Compute MDR/XDR/PDR per isolate if not already present
atlas_cols = set(atlas.columns)

if "MDR_v2" in atlas.columns and "XDR" in atlas.columns and "PDR" in atlas.columns:
    print("  MDR_v2, XDR, PDR already present — using directly")
else:
    print("  Computing MDR_v2, XDR, PDR per isolate...")
    print("  (Magiorakos et al. CMI 2012 definition)")
    results = atlas.apply(lambda row: compute_mxp(row, atlas_cols), axis=1)
    atlas["MDR_v2"] = results.apply(lambda x: x[0])
    atlas["XDR"]    = results.apply(lambda x: x[1])
    atlas["PDR"]    = results.apply(lambda x: x[2])
    print(f"  MDR rate: {atlas['MDR_v2'].mean()*100:.1f}%")
    print(f"  XDR rate: {atlas['XDR'].mean()*100:.1f}%")
    print(f"  PDR rate: {atlas['PDR'].mean()*100:.1f}%")

# ============================================================
# STEP 2  BUILD TEMPORAL CSV
# Rows: Organism × Country × Year × Age_Category
# Age categories: Pediatric, Elderly (All is computed at render time)
# ============================================================
print("\n" + "=" * 65)
print("STEP 2  BUILDING TEMPORAL CSV WITH XDR, PDR, AGE CATEGORY")
print("=" * 65)

AGE_CATS = ["Pediatric", "Elderly"]   # "All" is pooled in JS, not stored separately

rows = []
for org in TARGET_ORGANISMS:
    sub_org = atlas[atlas["Organism_Label"] == org]
    if len(sub_org) == 0:
        continue
    for year in sorted(sub_org["Year"].dropna().unique()):
        sub_yr = sub_org[sub_org["Year"] == year]
        for country in sorted(sub_yr["Country"].dropna().unique()):
            sub_c = sub_yr[sub_yr["Country"] == country]

            for age_cat in AGE_CATS:
                sub_a = sub_c[sub_c["Age_Category"] == age_cat]
                n = len(sub_a)
                if n < MIN_ISOLATES:
                    continue
                mdr_c = int(sub_a["MDR_v2"].sum())
                xdr_c = int(sub_a["XDR"].sum())
                pdr_c = int(sub_a["PDR"].sum())
                rows.append({
                    "Organism":    org,
                    "Country":     country,
                    "Year":        int(year),
                    "Age_Category": age_cat,
                    "N_Isolates":  n,
                    "MDR_Count":   mdr_c,
                    "XDR_Count":   xdr_c,
                    "PDR_Count":   pdr_c,
                    "MDR_Rate_%":  round(mdr_c / n * 100, 2),
                    "XDR_Rate_%":  round(xdr_c / n * 100, 2),
                    "PDR_Rate_%":  round(pdr_c / n * 100, 2),
                    "Income_Group": get_income(country),
                    "MDR_Definition": "Magiorakos2012_R+I"
                })

df_mxp = pd.DataFrame(rows)
csv_path = os.path.join(CSV_DIR, "temporal_mxp_age_country_year.csv")
df_mxp.to_csv(csv_path, index=False)
print(f"  Saved:temporal_mxp_age_country_year.csv ({len(df_mxp):,} rows)")
print(f"  Organisms covered: {df_mxp['Organism'].nunique()}")
print(f"  Countries covered: {df_mxp['Country'].nunique()}")
print(f"  Years covered: {sorted(df_mxp['Year'].unique())}")
print(f"  Age categories: {df_mxp['Age_Category'].unique()}")

# Quick sanity check — rows per age category
for ac in AGE_CATS:
    n = (df_mxp["Age_Category"] == ac).sum()
    print(f"  {ac}: {n:,} rows (country-year-organism cells >= {MIN_ISOLATES} isolates)")

# ============================================================
# STEP 3  LOAD SHAPEFILE FOR ISO3 MAPPING
# ============================================================
print("\n" + "=" * 65)
print("STEP 3  LOADING SHAPEFILE")
print("=" * 65)

SHP = "ne_110m_admin_0_countries.zip"
if not os.path.exists(SHP):
    print("  Downloading Natural Earth 110m shapefile...")
    urllib.request.urlretrieve(
        "https://naciscdn.org/naturalearth/110m/cultural/"
        "ne_110m_admin_0_countries.zip", SHP)
    print("  Downloaded.")
else:
    print("  Shapefile found locally.")

world = gpd.read_file(f"zip://{SHP}")
NAME_MAP = {
    "United States of America": "United States",
    "Republic of Korea":        "Korea, South",
    "Czechia":                  "Czech Republic",
    "Russian Federation":       "Russia",
    "Slovakia":                 "Slovak Republic",
    "Dominican Rep.":           "Dominican Republic",
    "Côte d'Ivoire":            "Ivory Coast",
}
world["Country_Std"] = world["NAME"].apply(lambda x: NAME_MAP.get(x, x))
iso3_map = dict(zip(world["Country_Std"], world["ADM0_A3"]))
world["rep"] = world.geometry.representative_point()
centroids = {r["Country_Std"]: (r["rep"].x, r["rep"].y)
             for _, r in world.iterrows()}
print(f"  {len(world)} polygons loaded")

# Apply ISO3 to temporal dataframe
df_mxp["ISO3"] = df_mxp["Country"].map(iso3_map)
df_mxp_valid = df_mxp[df_mxp["ISO3"].notna()].copy()
unmapped = df_mxp[df_mxp["ISO3"].isna()]["Country"].unique()
if len(unmapped) > 0:
    print(f"  WARNING: {len(unmapped)} countries not mapped to ISO3: {unmapped}")

# ============================================================
# STEP 4  BUILD JS DATA TABLE
#
# Structure: DATA[orgIdx][resTypeIdx][ageCatIdx][yearIdx]
#   resTypeIdx: 0=MDR, 1=XDR, 2=PDR
#   ageCatIdx:  0=All (pooled), 1=Pediatric, 2=Elderly
#   Each cell: {locations, z, text}
#
# For "All" (ageCatIdx=0): pool Pediatric + Elderly rows
#   per country, sum N_Isolates and Counts, recompute rate
# ============================================================
print("\n" + "=" * 65)
print("STEP 4  BUILDING JS LOOKUP TABLE")
print("=" * 65)

orgs_avail = [o for o in TARGET_ORGANISMS
              if o in df_mxp_valid["Organism"].unique()]
all_years  = sorted(df_mxp_valid["Year"].unique())
RES_TYPES  = ["MDR", "XDR", "PDR"]
AGE_LABELS = ["All", "Pediatric", "Elderly"]   # 0=All, 1=Pediatric, 2=Elderly

print(f"  Organisms: {len(orgs_avail)}")
print(f"  Years: {all_years[0]}–{all_years[-1]} ({len(all_years)} years)")
print(f"  Resistance types: {RES_TYPES}")
print(f"  Age categories: {AGE_LABELS}")

RATE_COL = {"MDR": "MDR_Rate_%", "XDR": "XDR_Rate_%", "PDR": "PDR_Rate_%"}
CNT_COL  = {"MDR": "MDR_Count",  "XDR": "XDR_Count",  "PDR": "PDR_Count"}


def build_cell(sub, res_type):
    """
    sub: filtered dataframe for a specific org/year/country combination.
    Returns {locations, z, text} dict for one JS cell.
    """
    if len(sub) == 0:
        return {"locations": [], "z": [], "text": []}
    locs, zvals, texts = [], [], []
    for _, r in sub.iterrows():
        rate = r[RATE_COL[res_type]]
        locs.append(r["ISO3"])
        zvals.append(round(float(rate), 2))
        texts.append(
            f"<b>{r['Country']}</b><br>"
            f"Organism: {r['Organism']}<br>"
            f"Year: {int(r['Year'])}<br>"
            f"{res_type} Rate: {rate:.1f}%<br>"
            f"Isolates: {int(r['N_Isolates']):,}<br>"
            f"Income: {r['Income_Group']}"
        )
    return {"locations": locs, "z": zvals, "text": texts}


def build_all_cell(sub_ped, sub_eld, res_type):
    """
    Pool Pediatric + Elderly for the same org/year.
    Aggregate per country: sum N and Count, recompute rate.
    Only include countries where pooled N >= MIN_ISOLATES.
    """
    combined = pd.concat([sub_ped, sub_eld], ignore_index=True)
    if len(combined) == 0:
        return {"locations": [], "z": [], "text": []}

    cnt_col = CNT_COL[res_type]
    agg = combined.groupby(["Country", "ISO3", "Organism",
                             "Year", "Income_Group"]).agg(
        N_Isolates=("N_Isolates", "sum"),
        Count=(cnt_col, "sum")
    ).reset_index()
    agg = agg[agg["N_Isolates"] >= MIN_ISOLATES].copy()
    agg["Rate"] = agg["Count"] / agg["N_Isolates"] * 100

    if len(agg) == 0:
        return {"locations": [], "z": [], "text": []}

    locs, zvals, texts = [], [], []
    for _, r in agg.iterrows():
        rate = r["Rate"]
        locs.append(r["ISO3"])
        zvals.append(round(float(rate), 2))
        texts.append(
            f"<b>{r['Country']}</b><br>"
            f"Organism: {r['Organism']}<br>"
            f"Year: {int(r['Year'])}<br>"
            f"{res_type} Rate (All ages): {rate:.1f}%<br>"
            f"Isolates: {int(r['N_Isolates']):,}<br>"
            f"Income: {r['Income_Group']}"
        )
    return {"locations": locs, "z": zvals, "text": texts}


# DATA[orgIdx][resTypeIdx][ageCatIdx][yearIdx]
data_table = []

for org in orgs_avail:
    org_block = []
    sub_org = df_mxp_valid[df_mxp_valid["Organism"] == org]

    for res_type in RES_TYPES:
        res_block = []

        for age_label in AGE_LABELS:
            age_block = []

            if age_label == "All":
                sub_ped = sub_org[sub_org["Age_Category"] == "Pediatric"]
                sub_eld = sub_org[sub_org["Age_Category"] == "Elderly"]
                for yr in all_years:
                    s_ped = sub_ped[sub_ped["Year"] == yr]
                    s_eld = sub_eld[sub_eld["Year"] == yr]
                    age_block.append(build_all_cell(s_ped, s_eld, res_type))
            else:
                sub_age = sub_org[sub_org["Age_Category"] == age_label]
                for yr in all_years:
                    s = sub_age[sub_age["Year"] == yr]
                    age_block.append(build_cell(s, res_type))

            res_block.append(age_block)
        org_block.append(res_block)
    data_table.append(org_block)

print("  Data table built.")
print(f"  Dimensions: {len(orgs_avail)} orgs × {len(RES_TYPES)} res × "
      f"{len(AGE_LABELS)} ages × {len(all_years)} years")

# ============================================================
# STEP 5  BUILD PLOTLY BASE FIGURE
# ONE trace per organism (first year, MDR, All ages)
# Country label trace always on top
# ============================================================
print("\n" + "=" * 65)
print("STEP 5  BUILDING PLOTLY BASE FIGURE")
print("=" * 65)

COLORSCALE = [
    [0.0,  "#ffffcc"],
    [0.25, "#fecc5c"],
    [0.5,  "#fd8d3c"],
    [0.75, "#f03b20"],
    [1.0,  "#bd0026"]
]

base_traces = []
for i, org in enumerate(orgs_avail):
    # Seed with MDR / All / first year
    cell = data_table[i][0][0][0]   # [org][MDR=0][All=0][year0]
    tr = go.Choropleth(
        locations=cell["locations"],
        z=cell["z"],
        zmin=0, zmax=100,
        colorscale=COLORSCALE,
        showscale=False,
        text=cell["text"],
        hovertemplate="%{text}<extra></extra>",
        visible=(i == 0)
    )
    base_traces.append(tr)

# Country label scatter
all_ctries = sorted(df_mxp_valid["Country"].unique())
lons, lats, names = [], [], []
for c in all_ctries:
    if c in centroids:
        lon, lat = centroids[c]
        lons.append(lon); lats.append(lat); names.append(c)

label_trace = go.Scattergeo(
    lon=lons, lat=lats, text=names, mode="text",
    textfont=dict(size=6.5, color="#222222", family="Arial"),
    hoverinfo="skip", showlegend=False, visible=True,
    name="country_labels"
)
base_traces.append(label_trace)

fig = go.Figure(data=base_traces)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="white",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
        bgcolor="#d6eaf8",
        landcolor="#f0f0f0",
        showland=True, showcountries=True,
        countrycolor="#888888",
        coastlinecolor="#555555",
        showocean=True, oceancolor="#d6eaf8"
    ),
    updatemenus=[],
    sliders=[]
)

fig_json = fig.to_json()
print("  Plotly figure built.")

# ============================================================
# STEP 6  BUILD HTML
# ============================================================
print("\n" + "=" * 65)
print("STEP 6  BUILDING HTML")
print("=" * 65)

n_orgs    = len(orgs_avail)
data_js   = json.dumps(data_table)
years_js  = json.dumps([int(y) for y in all_years])
norgs_js  = str(n_orgs)
res_js    = json.dumps(RES_TYPES)       # ["MDR","XDR","PDR"]
age_js    = json.dumps(AGE_LABELS)      # ["All","Pediatric","Elderly"]

# HTML option elements
org_opts = "\n".join(
    f'          <option value="{i}">{org}</option>'
    for i, org in enumerate(orgs_avail))

year_opts = "\n".join(
    f'          <option value="{int(yr)}">{int(yr)}</option>'
    for yr in all_years)

res_opts = "\n".join(
    f'          <option value="{i}">{rt}</option>'
    for i, rt in enumerate(RES_TYPES))

age_opts = "\n".join(
    f'          <option value="{i}">{al}</option>'
    for i, al in enumerate(AGE_LABELS))

PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f4f8;
    min-height: 100vh;
}
.page-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 18px 18px 28px;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.title-bar {
    background: linear-gradient(135deg, #1a3a5c 0%, #2471a3 100%);
    color: #ffffff;
    text-align: center;
    padding: 16px 24px;
    border-radius: 10px 10px 0 0;
    border-bottom: 3px solid #1a5276;
}
.title-bar h1 { font-size: 1.45rem; font-weight: 800; letter-spacing: 0.5px; line-height: 1.3; }
.title-bar p  { font-size: 0.85rem; opacity: 0.82; margin-top: 4px; }
.main-row {
    display: flex;
    background: #ffffff;
    border-left: 1px solid #d5dde5;
    border-right: 1px solid #d5dde5;
}
.map-col {
    flex: 1 1 0%;
    min-width: 0;
    padding: 0;
    position: relative;
}
.ctrl-panel {
    width: 230px;
    flex-shrink: 0;
    background: #f7f9fc;
    border-left: 2px solid #d5dde5;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px 14px 14px;
}
.ctrl-panel .section-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: #5d6d7e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}
.btn-row { display: flex; gap: 10px; }
.btn-play, .btn-pause {
    flex: 1; padding: 9px 0;
    border: none; border-radius: 6px;
    font-size: 0.90rem; font-weight: 700;
    cursor: pointer; transition: opacity 0.15s;
}
.btn-play  { background: #27ae60; color: #fff; }
.btn-pause { background: #e74c3c; color: #fff; }
.btn-play:hover  { opacity: 0.87; }
.btn-pause:hover { opacity: 0.87; }
.ctrl-select {
    width: 100%;
    padding: 8px 10px;
    border: 2px solid #e6a817;
    border-radius: 6px;
    background: #fff3cd;
    font-size: 0.86rem;
    font-weight: 700;
    color: #5a3e00;
    cursor: pointer;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%235a3e00'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    padding-right: 28px;
}
.ctrl-select.blue-select {
    border-color: #2980b9;
    background-color: #d6eaf8;
    color: #1a252f;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%231a252f'/%3E%3C/svg%3E");
}
.ctrl-select.green-select {
    border-color: #1e8449;
    background-color: #d5f5e3;
    color: #145a32;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%23145a32'/%3E%3C/svg%3E");
}
.ctrl-select.purple-select {
    border-color: #7d3c98;
    background-color: #e8daef;
    color: #4a235a;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%234a235a'/%3E%3C/svg%3E");
}
.ctrl-select:focus { box-shadow: 0 0 0 3px rgba(41,128,185,0.25); }
.colorbar-row {
    background: #ffffff;
    border-left: 1px solid #d5dde5;
    border-right: 1px solid #d5dde5;
    border-top: 1px solid #e8edf2;
    padding: 10px 16px 8px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.cb-label { font-size: 0.78rem; color: #333; font-weight: 600; white-space: nowrap; }
.cb-gradient {
    display: block; width: 100%; min-width: 200px; height: 22px; border-radius: 4px;
    background: linear-gradient(to right, #ffffcc 0%, #fecc5c 25%, #fd8d3c 50%, #f03b20 75%, #bd0026 100%) !important;
    border: 1px solid #ccc;
}
.cb-ticks {
    display: flex; justify-content: space-between; width: 100%;
    font-size: 0.72rem; color: #555; margin-top: 3px;
}
.cb-wrap { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.desc-box {
    background: #f8f9fa;
    border-left: 7px solid #2980b9;
    border-radius: 0 0 10px 10px;
    border-top: none;
    padding: 16px 22px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #1a252f;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    border-bottom: 1px solid #d5dde5;
}
.desc-box strong { color: #1a3a5c; }
#res-label-display {
    font-size: 0.78rem; font-weight: 700;
    color: #555; text-align: center;
    margin-top: 2px; letter-spacing: 0.5px;
}
"""

JS = """
// ── embedded data ──────────────────────────────────────────────
// DATA[orgIdx][resTypeIdx][ageCatIdx][yearIdx] = {locations, z, text}
const DATA      = DATA_PLACEHOLDER;
const YEARS     = YEARS_PLACEHOLDER;
const NORGS     = NORGS_PLACEHOLDER;
const RES_TYPES = RES_PLACEHOLDER;
const AGE_CATS  = AGE_PLACEHOLDER;

let _animTimer = null;
let _orgIdx    = 0;
let _yearIdx   = 0;
let _resIdx    = 0;   // 0=MDR, 1=XDR, 2=PDR
let _ageIdx    = 0;   // 0=All, 1=Pediatric, 2=Elderly

function applySelection() {
    const d = DATA[_orgIdx][_resIdx][_ageIdx][_yearIdx];

    // Update choropleth data on the active trace
    Plotly.restyle('plotly-map',
        { locations: [d.locations], z: [d.z], text: [d.text] },
        [_orgIdx]
    );

    // Show only active organism trace + label trace (index NORGS)
    const vis = Array(NORGS + 1).fill(false);
    vis[_orgIdx] = true;
    vis[NORGS]   = true;
    Plotly.restyle('plotly-map', { visible: vis });

    // Update colorbar label
    const resLabel = RES_TYPES[_resIdx];
    const ageLabel = AGE_CATS[_ageIdx];
    document.getElementById('res-label-display').textContent =
        resLabel + ' Rate (%) — ' + ageLabel;
}

function setYear(idx) {
    _yearIdx = idx;
    document.getElementById('year-select').value = YEARS[idx];
    applySelection();
}

function setOrganism(idx) {
    _orgIdx = idx;
    applySelection();
}

function setResType(idx) {
    _resIdx = idx;
    applySelection();
}

function setAgeCategory(idx) {
    _ageIdx = idx;
    applySelection();
}

function playAnimation() {
    if (_animTimer) return;
    _animTimer = setInterval(() => {
        _yearIdx = (_yearIdx + 1) % YEARS.length;
        document.getElementById('year-select').value = YEARS[_yearIdx];
        applySelection();
    }, 700);
}

function pauseAnimation() {
    clearInterval(_animTimer);
    _animTimer = null;
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('org-select').addEventListener('change', function() {
        setOrganism(parseInt(this.value));
    });
    document.getElementById('year-select').addEventListener('change', function() {
        const idx = YEARS.indexOf(parseInt(this.value));
        if (idx >= 0) setYear(idx);
    });
    document.getElementById('res-select').addEventListener('change', function() {
        setResType(parseInt(this.value));
    });
    document.getElementById('age-select').addEventListener('change', function() {
        setAgeCategory(parseInt(this.value));
    });
    // Initialise label
    document.getElementById('res-label-display').textContent =
        RES_TYPES[0] + ' Rate (%) — ' + AGE_CATS[0];
});
"""

js_code = (JS
           .replace("DATA_PLACEHOLDER",  data_js)
           .replace("YEARS_PLACEHOLDER", years_js)
           .replace("NORGS_PLACEHOLDER", norgs_js)
           .replace("RES_PLACEHOLDER",   res_js)
           .replace("AGE_PLACEHOLDER",   age_js))

DESCRIPTION = """
<strong>About this map.</strong>
This is the <strong>spatiotemporal evolution</strong> view of antimicrobial resistance
across countries from 2004 to 2024 for nosocomial neuroinvasive bacterial pathogens.
Select an organism and press <strong>Play</strong> to animate year by year and observe
how resistance intensifies, spreads, or recedes over two decades.
<br><br>
<strong>Resistance type</strong> (green dropdown): choose between
<em>MDR</em> (multi-drug resistant - non-susceptible to ≥3 antibiotic classes),
<em>XDR</em> (extensively drug resistant - susceptible to ≤2 classes only), or
<em>PDR</em> (pan-drug resistant - non-susceptible to all tested classes).
All definitions follow Magiorakos <em>et al.</em> (CMI 2012; 18:268–281)
using R + I = non-susceptible.
<br><br>
<strong>Age category</strong> (purple dropdown):
<em>All</em> = Pediatric + Elderly pooled; <em>Pediatric</em> = 0–17 years;
<em>Elderly</em> = 61+ years.
Only country–year–organism–age combinations with <strong>≥10 isolates</strong>
are shown (WHO GLASS 2023 minimum threshold).
<br><br>
<strong>Controls:</strong>
<em>Organism</em> (amber) = one of 13 WHO priority nosocomial pathogens. &nbsp;|&nbsp;
<em>Year</em> (blue) = jump to any year 2004–2024. &nbsp;|&nbsp;
<em>Resistance Type</em> (green) = MDR / XDR / PDR. &nbsp;|&nbsp;
<em>Age Category</em> (purple) = All / Pediatric / Elderly. &nbsp;|&nbsp;
<em>Play / Pause</em> = animate the map automatically.
Hover any coloured country for detailed statistics.
<br><br>
<strong>Data source:</strong> Pfizer ATLAS Surveillance Programme 2004–2024.
<strong>MDR definition:</strong> Magiorakos <em>et al.</em> CMI 2012.
<strong>Income classification:</strong> World Bank Country &amp; Lending Groups FY2026.
"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Spatiotemporal MDR/XDR/PDR Evolution - All Organisms (2004–2024)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{PAGE_CSS}
</style>
</head>
<body>
<div class="page-wrap">

  <!-- TITLE -->
  <div class="title-bar">
    <h1>Spatiotemporal Resistance Evolution &mdash; All Organisms (2004&ndash;2024)</h1>
    <p>AMR Data Challenge 2026 &middot; Phase 3 Geographic Analysis &middot; Nosocomial Neuroinvasive Bacterial AMR Study</p>
  </div>

  <!-- MAIN ROW -->
  <div class="main-row">

    <!-- MAP -->
    <div class="map-col">
      <div id="plotly-map" style="width:100%;height:540px;"></div>
    </div>

    <!-- CONTROLS -->
    <div class="ctrl-panel">

      <!-- Play / Pause -->
      <div>
        <div class="section-label">Animate</div>
        <div class="btn-row">
          <button class="btn-play"  onclick="playAnimation()">&#9654; Play</button>
          <button class="btn-pause" onclick="pauseAnimation()">&#9646;&#9646; Pause</button>
        </div>
      </div>

      <!-- Organism -->
      <div>
        <div class="section-label">Organism</div>
        <select id="org-select" class="ctrl-select">
{org_opts}
        </select>
      </div>

      <!-- Year -->
      <div>
        <div class="section-label">Year</div>
        <select id="year-select" class="ctrl-select blue-select">
{year_opts}
        </select>
      </div>

      <!-- Resistance Type (NEW) -->
      <div>
        <div class="section-label">Resistance Type</div>
        <select id="res-select" class="ctrl-select green-select">
{res_opts}
        </select>
      </div>

      <!-- Age Category (NEW) -->
      <div>
        <div class="section-label">Age Category</div>
        <select id="age-select" class="ctrl-select purple-select">
{age_opts}
        </select>
      </div>

      <!-- Mini legend -->
      <div style="margin-top:auto;font-size:0.74rem;color:#7f8c8d;line-height:1.6;">
        <strong>MDR:</strong> ≥3 classes non-susceptible<br>
        <strong>XDR:</strong> susceptible to ≤2 classes<br>
        <strong>PDR:</strong> 0 susceptible classes<br>
        <em style="font-size:0.70rem;">Magiorakos et al. CMI 2012</em><br><br>
        <strong>Min. isolates:</strong> 10&nbsp;(WHO GLASS 2023)
      </div>

    </div><!-- /ctrl-panel -->
  </div><!-- /main-row -->

  <!-- COLOUR BAR -->
  <div class="colorbar-row">
    <span class="cb-label">Rate (%):</span>
    <div class="cb-wrap">
      <div class="cb-gradient"></div>
      <div class="cb-ticks">
        <span>0%</span><span>25%</span><span>50%</span>
        <span>75%</span><span>100%</span>
      </div>
    </div>
    <div id="res-label-display">MDR Rate (%) — All</div>
  </div>

  <!-- DESCRIPTION -->
  <div class="desc-box">
{DESCRIPTION}
  </div>

</div><!-- /page-wrap -->

<div style="text-align:center;font-family:'Segoe UI',Arial,sans-serif;
font-size:12px;color:#7f8c8d;padding:10px 0 18px 0;
border-top:1px solid #dddddd;max-width:1280px;margin:0 auto;">
&copy; 2026 PSG Institute of Medical Sciences &amp; Research
</div>

<script>
const figData = {fig_json};
Plotly.newPlot('plotly-map',
    figData.data,
    figData.layout,
    {{ responsive: true, displayModeBar: false }}
);

{js_code}
</script>
</body>
</html>"""

out_path = os.path.join(LAY_DIR, "spatiotemporal_layout.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"  Saved: {out_path}")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 65)
print("PHASE 3 SPATIOTEMPORAL COMPLETE")
print("=" * 65)
print(f"""
  Outputs:
    CSV : phase3_outputs/csv_outputs/temporal_mxp_age_country_year.csv
    HTML: phase3_outputs/layout/spatiotemporal_layout.html

  New dropdowns added:
    Resistance Type : MDR | XDR | PDR
    Age Category    : All | Pediatric | Elderly

  Dropdown colours:
    Organism         - amber  (existing)
    Year             - blue   (existing)
    Resistance Type  - green  (new)
    Age Category     - purple (new)

  JS lookup table: DATA[orgIdx][resTypeIdx][ageCatIdx][yearIdx]
  "All" age = Pediatric + Elderly pooled per country at build time,
  recomputed rate from summed counts (Option A confirmed).

  MDR/XDR/PDR definition: Magiorakos et al. CMI 2012
  Minimum isolates: 10 (WHO GLASS 2023)
""")


# ============================================================
# AMR DATA CHALLENGE 2026 — PHASE 3
# LMIC vs HIC EQUITY GAP — DUAL Y-AXIS CHART 
# FIX: HTML now built via string concatenation — not nested
#      f-strings — so JS curly braces do not break Python
#      f-string parsing.
#
# COLOUR SCHEME:
#   Resistance family : red (#c0392b) + blue (#2980b9) + pink fill
#   Expenditure family: dark green (#1e8449) + purple (#7d3c98) + green fill
#
# LAYOUT:
#   Single panel, dual Y-axis
#   Plotly legend removed (showlegend:false)
#   Right panel static HTML legend only
#   Scrollable description box below chart (max-height 200px)
#
# INPUTS:
#   phase3_outputs/csv_outputs/temporal_mxp_age_country_year.csv
#   phase3_outputs/csv_outputs/equity_gap_mxp_age_by_year.csv
#   SYB68_325_202511_Expenditure on health.csv
#
# OUTPUT:
#   phase3_outputs/layout/equity_gap_dual_axis.html
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
TEMPORAL_CSV    = "phase3_outputs/csv_outputs/temporal_mxp_age_country_year.csv"
EQUITY_CSV      = "phase3_outputs/csv_outputs/equity_gap_mxp_age_by_year.csv"
EXPENDITURE_CSV = "SYB68_325_202511_Expenditure on health.csv"

OUT_DIR = "phase3_outputs"
CSV_DIR = os.path.join(OUT_DIR, "csv_outputs")
LAY_DIR = os.path.join(OUT_DIR, "layout")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(LAY_DIR, exist_ok=True)

MIN_ISOLATES  = 10
MIN_YR_POINTS = 3

TARGET_ORGANISMS = [
    "Klebsiella pneumoniae", "Acinetobacter baumannii",
    "Escherichia coli", "Staphylococcus aureus",
    "Pseudomonas aeruginosa", "Enterobacter cloacae",
    "Enterococcus faecium", "Staphylococcus epidermidis",
    "Serratia marcescens", "Citrobacter koseri",
    "Streptococcus pneumoniae", "Haemophilus influenzae",
    "Salmonella spp"
]

HIC_COUNTRIES = {
    "United States", "Canada", "Germany", "France", "United Kingdom",
    "Italy", "Spain", "Australia", "Japan", "Korea, South",
    "Netherlands", "Belgium", "Sweden", "Denmark", "Switzerland",
    "Norway", "Finland", "Austria", "Ireland", "New Zealand",
    "Israel", "Singapore", "Hong Kong", "Taiwan", "Czech Republic",
    "Slovenia", "Slovak Republic", "Estonia", "Lithuania", "Latvia",
    "Portugal", "Greece", "Hungary", "Poland", "Qatar", "Kuwait",
    "Saudi Arabia", "Oman", "Puerto Rico", "Croatia"
}

RES_TYPES  = ["MDR", "XDR", "PDR"]
AGE_LABELS = ["All", "Pediatric", "Elderly"]
RATE_COL   = {"MDR": "MDR_Rate_%", "XDR": "XDR_Rate_%", "PDR": "PDR_Rate_%"}
COUNT_COL  = {"MDR": "MDR_Count",  "XDR": "XDR_Count",  "PDR": "PDR_Count"}

EXPEND_NAME_MAP = {
    "United States of America":
        "United States",
    "United Kingdom of Great Britain and Northern Ireland":
        "United Kingdom",
    "Republic of Korea":                 "Korea, South",
    "Czechia":                           "Czech Republic",
    "Russian Federation":                "Russia",
    "Slovakia":                          "Slovak Republic",
    "Dominican Republic":                "Dominican Republic",
    "C\u00f4te d'Ivoire":               "Ivory Coast",
    "Viet Nam":                          "Vietnam",
    "Iran (Islamic Republic of)":        "Iran",
    "Syrian Arab Republic":              "Syria",
    "Bolivia (Plurinational State of)":  "Bolivia",
    "Venezuela (Bolivarian Republic of)":"Venezuela",
    "China, Hong Kong SAR":              "Hong Kong",
    "China, Taiwan Province of China":   "Taiwan",
    "Republic of Moldova":               "Moldova",
    "North Macedonia":                   "North Macedonia",
    "T\u00fcrk\u00fcy":                 "Turkey",
    "Lao People's Democratic Republic":  "Laos",
}

HIC_EXPEND_NAMES = {
    "United States of America", "Canada", "Germany", "France",
    "United Kingdom of Great Britain and Northern Ireland",
    "Italy", "Spain", "Australia", "Japan", "Republic of Korea",
    "Netherlands", "Belgium", "Sweden", "Denmark", "Switzerland",
    "Norway", "Finland", "Austria", "Ireland", "New Zealand",
    "Israel", "Singapore", "Czechia", "Slovakia",
    "Estonia", "Lithuania", "Latvia", "Portugal", "Greece",
    "Hungary", "Poland", "Qatar", "Kuwait", "Saudi Arabia",
    "Oman", "Croatia",
    "United States", "United Kingdom", "Korea, South",
    "Czech Republic", "Slovak Republic",
}

#============================================================
# STEP 1 — LOAD TEMPORAL CSV
# ============================================================
print("=" * 65)
print("STEP 1  LOADING TEMPORAL CSV")
print("=" * 65)

df = pd.read_csv(TEMPORAL_CSV)
print(f"  Loaded: {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")

# Verify required columns
required = ["Organism", "Country", "Year", "Age_Category",
            "N_Isolates", "MDR_Count", "XDR_Count", "PDR_Count",
            "MDR_Rate_%", "XDR_Rate_%", "PDR_Rate_%", "Income_Group"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in temporal CSV: {missing}\n"
                     "Run Phase3_spatiotemporal_v4.py first.")

df["Is_HIC"] = df["Country"].apply(lambda c: c in HIC_COUNTRIES)

# ============================================================
# BUILD EQUITY GAP YEAR CSV
# For each organism × resistance type × age category × year:
#   HIC mean = weighted mean across all HIC countries
#              (sum of counts / sum of isolates)
#   Per-country rate is already in the temporal CSV
# ============================================================
print("\n" + "=" * 65)
print("STEP 2  BUILDING EQUITY GAP YEAR CSV")
print("=" * 65)

eq_rows = []

for org in TARGET_ORGANISMS:
    sub_org = df[df["Organism"] == org]
    if len(sub_org) == 0:
        continue

    for res in RES_TYPES:
        rate_col  = RATE_COL[res]
        count_col = COUNT_COL[res]

        for age_label in AGE_LABELS:

            if age_label == "All":
                # Pool Pediatric + Elderly per country per year
                sub_age = sub_org.groupby(
                    ["Organism", "Country", "Year",
                     "Income_Group", "Is_HIC"]).agg(
                    N_Isolates=("N_Isolates", "sum"),
                    Count=(count_col, "sum")
                ).reset_index()
                sub_age = sub_age[sub_age["N_Isolates"] >= MIN_ISOLATES].copy()
                sub_age["Rate"] = sub_age["Count"] / sub_age["N_Isolates"] * 100
            else:
                sub_age = sub_org[
                    sub_org["Age_Category"] == age_label].copy()
                sub_age = sub_age[sub_age["N_Isolates"] >= MIN_ISOLATES].copy()
                sub_age["Rate"] = sub_age[rate_col]

            if len(sub_age) == 0:
                continue

            for yr in sorted(sub_age["Year"].unique()):
                yr_d   = sub_age[sub_age["Year"] == yr]
                hic_d  = yr_d[yr_d["Is_HIC"] == True]
                lmic_d = yr_d[yr_d["Is_HIC"] == False]

                n_hic  = int(hic_d["N_Isolates"].sum())
                cnt_hic = int(hic_d["Count"].sum()) if "Count" in hic_d.columns \
                    else int((hic_d["Rate"] * hic_d["N_Isolates"] / 100).sum())
                hic_rate = round(cnt_hic / n_hic * 100, 2) if n_hic > 0 else None

                n_lmic  = int(lmic_d["N_Isolates"].sum())
                cnt_lmic = int(lmic_d["Count"].sum()) if "Count" in lmic_d.columns \
                    else int((lmic_d["Rate"] * lmic_d["N_Isolates"] / 100).sum())
                lmic_rate = round(cnt_lmic / n_lmic * 100, 2) if n_lmic > 0 else None

                eq_rows.append({
                    "Organism":        org,
                    "Resistance_Type": res,
                    "Age_Category":    age_label,
                    "Year":            int(yr),
                    "HIC_Rate_%":      hic_rate,
                    "LMIC_Rate_%":     lmic_rate,
                    "HIC_N_Isolates":  n_hic,
                    "LMIC_N_Isolates": n_lmic,
                    "HIC_N_Countries": int(hic_d["Country"].nunique()
                                          if "Country" in hic_d.columns else 0),
                    "LMIC_N_Countries": int(lmic_d["Country"].nunique()
                                           if "Country" in lmic_d.columns else 0),
                })

df_eq = pd.DataFrame(eq_rows)
df_eq["Equity_Gap_%"] = (df_eq["LMIC_Rate_%"] - df_eq["HIC_Rate_%"]).round(2)

csv_path = os.path.join(CSV_DIR, "equity_gap_mxp_age_by_year.csv")
df_eq.to_csv(csv_path, index=False)
print(f"  Saved: equity_gap_mxp_age_by_year.csv ({len(df_eq):,} rows)")
print(f"  Resistance types: {df_eq['Resistance_Type'].unique()}")
print(f"  Age categories:   {df_eq['Age_Category'].unique()}")
print(f"  Years covered:    {sorted(df_eq['Year'].unique())}")


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("=" * 65)
print("STEP 1 — LOADING DATA")
print("=" * 65)

df = pd.read_csv(TEMPORAL_CSV)
print(f"  Temporal CSV: {len(df):,} rows")

df_eq = pd.read_csv(EQUITY_CSV)
print(f"  Equity CSV:   {len(df_eq):,} rows")

df["Is_HIC"] = df["Country"].apply(lambda c: c in HIC_COUNTRIES)


# ============================================================
# LOAD EXPENDITURE DATA
# ============================================================
print("\n" + "=" * 65)
print("STEP 2 — LOADING EXPENDITURE DATA")
print("=" * 65)

exp_raw = pd.read_csv(
    EXPENDITURE_CSV,
    skiprows=1, header=0,
    names=["Code", "Country", "Year", "Series", "Value",
           "Footnotes", "Source"],
    encoding="latin-1"
)

exp_gdp = exp_raw[
    exp_raw["Series"] == "Current health expenditure (% of GDP)"
].copy()
exp_gdp["Value"] = pd.to_numeric(exp_gdp["Value"], errors="coerce")
exp_gdp = exp_gdp.dropna(subset=["Value"])
exp_gdp["Year"] = exp_gdp["Year"].astype(int)
exp_gdp["Country_Std"] = exp_gdp["Country"].apply(
    lambda c: EXPEND_NAME_MAP.get(c, c))
exp_gdp["Is_HIC"] = exp_gdp["Country"].apply(
    lambda c: c in HIC_EXPEND_NAMES or
              EXPEND_NAME_MAP.get(c, c) in HIC_COUNTRIES)

hic_exp_by_yr = exp_gdp[exp_gdp["Is_HIC"]].groupby("Year").agg(
    HIC_Mean=("Value", "mean"),
    N=("Country_Std", "nunique")
).reset_index()
hic_exp_lookup = dict(zip(
    hic_exp_by_yr["Year"].astype(int),
    hic_exp_by_yr["HIC_Mean"].round(2)))

exp_lookup = {}
for _, r in exp_gdp.iterrows():
    cty = r["Country_Std"]
    yr  = int(r["Year"])
    val = round(float(r["Value"]), 2)
    exp_lookup.setdefault(cty, {})[yr] = val

EXP_YEARS = sorted(exp_gdp["Year"].unique())
print(f"  Countries in GHED: {exp_gdp['Country_Std'].nunique()}")
print(f"  Years: {EXP_YEARS}")
print(f"  HIC mean expenditure:")
for yr, v in sorted(hic_exp_lookup.items()):
    print(f"    {yr}: {v:.2f}% GDP")


# ============================================================
# LOAD AND PROCESS EXPENDITURE CSV
# ============================================================
print("\n" + "=" * 65)
print("STEP 2 — LOADING HEALTH EXPENDITURE CSV (UNdata SYB68)")
print("=" * 65)

print(f"  Raw rows loaded: {len(exp_raw):,}")

# Keep only "Current health expenditure (% of GDP)"
exp_gdp = exp_raw[
    exp_raw["Series"] == "Current health expenditure (% of GDP)"
].copy()
exp_gdp["Value"] = pd.to_numeric(exp_gdp["Value"], errors="coerce")
exp_gdp = exp_gdp.dropna(subset=["Value"])
exp_gdp["Year"] = exp_gdp["Year"].astype(int)

# Apply name mapping
exp_gdp["Country_Std"] = exp_gdp["Country"].apply(
    lambda c: EXPEND_NAME_MAP.get(c, c))

# Flag HIC in expenditure data
exp_gdp["Is_HIC"] = exp_gdp["Country"].apply(
    lambda c: c in HIC_EXPEND_NAMES or
              EXPEND_NAME_MAP.get(c, c) in HIC_COUNTRIES)

print(f"  Rows after filtering to % of GDP: {len(exp_gdp):,}")
print(f"  Countries in expenditure data: {exp_gdp['Country_Std'].nunique()}")
print(f"  Years available: {sorted(exp_gdp['Year'].unique())}")
print(f"  HIC rows: {exp_gdp['Is_HIC'].sum():,}")

# Compute HIC mean expenditure per year
# Weighted mean = simple mean across HIC countries (no isolate weighting —
# expenditure is a country-level indicator, not isolate-weighted)
hic_exp = exp_gdp[exp_gdp["Is_HIC"]].groupby("Year").agg(
    HIC_Mean_Exp=("Value", "mean"),
    HIC_N_Countries=("Country_Std", "nunique")
).reset_index()
hic_exp["HIC_Mean_Exp"] = hic_exp["HIC_Mean_Exp"].round(2)
print(f"\n  HIC mean expenditure by year:")
for _, r in hic_exp.iterrows():
    print(f"    {int(r['Year'])}: {r['HIC_Mean_Exp']:.2f}% GDP "
          f"({int(r['HIC_N_Countries'])} HIC countries)")

# Save processed expenditure CSV
exp_out = exp_gdp[["Country_Std", "Year", "Value", "Is_HIC"]].copy()
exp_out.columns = ["Country", "Year", "Health_Exp_PCT_GDP", "Is_HIC"]
exp_out = exp_out.merge(
    hic_exp[["Year", "HIC_Mean_Exp"]], on="Year", how="left")
exp_out.to_csv(
    os.path.join(CSV_DIR, "health_expenditure_processed.csv"),
    index=False)
print(f"\n  Saved: health_expenditure_processed.csv "
      f"({len(exp_out):,} rows)")

# Build per-country expenditure lookup
# exp_lookup[country] = {year: value}
exp_lookup = {}
for _, r in exp_gdp.iterrows():
    cty = r["Country_Std"]
    yr  = int(r["Year"])
    val = round(float(r["Value"]), 2)
    if cty not in exp_lookup:
        exp_lookup[cty] = {}
    exp_lookup[cty][yr] = val

# HIC mean lookup: {year: hic_mean}
hic_exp_lookup = dict(zip(
    hic_exp["Year"].astype(int),
    hic_exp["HIC_Mean_Exp"].round(2)))

EXP_YEARS = sorted(exp_gdp["Year"].unique())
print(f"\n  Expenditure years: {EXP_YEARS}")

# ============================================================
# STEP 3 — BUILD JS DATA TABLE
# ============================================================
print("\n" + "=" * 65)
print("STEP 3 — BUILDING JS DATA TABLE")
print("=" * 65)

orgs_avail = [o for o in TARGET_ORGANISMS if o in df["Organism"].unique()]

org_to_ctys = {}
for org in orgs_avail:
    sub_o = df[df["Organism"] == org]
    sub_all = sub_o.groupby(["Country", "Year"]).agg(
        N=("N_Isolates", "sum")).reset_index()
    sub_all = sub_all[sub_all["N"] >= MIN_ISOLATES]
    ctys = sorted([
        c for c in sub_all["Country"].unique()
        if len(sub_all[sub_all["Country"] == c]) >= MIN_YR_POINTS
    ])
    if ctys:
        org_to_ctys[org] = ctys

org_list = [o for o in orgs_avail if o in org_to_ctys]
print(f"  Organisms: {len(org_list)}")

hic_res_lookup = {}
for org in org_list:
    hic_res_lookup[org] = {}
    for res in RES_TYPES:
        hic_res_lookup[org][res] = {}
        for age in AGE_LABELS:
            sub_eq = df_eq[
                (df_eq["Organism"] == org) &
                (df_eq["Resistance_Type"] == res) &
                (df_eq["Age_Category"] == age)
            ].dropna(subset=["HIC_Rate_%"])
            hic_res_lookup[org][res][age] = dict(
                zip(sub_eq["Year"].astype(int),
                    sub_eq["HIC_Rate_%"].round(2)))


def get_resistance_series(org, country, res, age_label):
    sub_org = df[df["Organism"] == org]
    income  = "Unclassified"
    tmp = df[df["Country"] == country]["Income_Group"]
    if len(tmp) > 0:
        income = tmp.iloc[0]
    if age_label == "All":
        sub = sub_org[sub_org["Country"] == country].groupby("Year").agg(
            N_Isolates=("N_Isolates", "sum"),
            Count=(COUNT_COL[res], "sum")
        ).reset_index()
        sub = sub[sub["N_Isolates"] >= MIN_ISOLATES].copy()
        sub["Rate"] = sub["Count"] / sub["N_Isolates"] * 100
    else:
        sub = sub_org[
            (sub_org["Country"] == country) &
            (sub_org["Age_Category"] == age_label)
        ].copy()
        sub = sub[sub["N_Isolates"] >= MIN_ISOLATES].copy()
        sub["Rate"] = sub[RATE_COL[res]]
    if len(sub) == 0:
        return [], income
    sub = sub.sort_values("Year")
    return [
        (int(r["Year"]), round(float(r["Rate"]), 2), int(r["N_Isolates"]))
        for _, r in sub.iterrows()
    ], income


def get_exp_series(country):
    cty_data = exp_lookup.get(country, {})
    if not cty_data:
        return [], [], [], [], [], False
    years_e   = sorted(cty_data.keys())
    cty_exp_v = [cty_data[y] for y in years_e]
    hic_exp_v = [round(hic_exp_lookup.get(y, np.nan), 2)
                 if y in hic_exp_lookup else None
                 for y in years_e]
    valid = [(y, c, h) for y, c, h in
             zip(years_e, cty_exp_v, hic_exp_v) if h is not None]
    if valid:
        fy  = [v[0] for v in valid]
        fce = [v[1] for v in valid]
        fhe = [v[2] for v in valid]
        fill_ex = fy + list(reversed(fy))
        fill_ey = fce + list(reversed(fhe))
    else:
        fill_ex, fill_ey = [], []
    return years_e, cty_exp_v, hic_exp_v, fill_ex, fill_ey, True


data_table = []
for org in org_list:
    org_block = []
    for res in RES_TYPES:
        res_block = []
        for age in AGE_LABELS:
            age_block = []
            hic_yr = hic_res_lookup[org][res][age]
            for cty in org_to_ctys[org]:
                series, income = get_resistance_series(org, cty, res, age)
                if len(series) < MIN_YR_POINTS:
                    age_block.append({
                        "years_r": [], "rate": [], "hic_r": [],
                        "fill_rx": [], "fill_ry": [],
                        "years_e": [], "cty_exp": [], "hic_exp": [],
                        "fill_ex": [], "fill_ey": [],
                        "country": cty, "organism": org,
                        "income": income, "has_exp": False, "n": []
                    })
                    continue
                years_r = [s[0] for s in series]
                rate_c  = [s[1] for s in series]
                n_c     = [s[2] for s in series]
                hic_r   = [round(hic_yr.get(y, np.nan), 2)
                           if not np.isnan(hic_yr.get(y, np.nan))
                           else None for y in years_r]
                valid_r = [(y, r, h) for y, r, h in
                           zip(years_r, rate_c, hic_r) if h is not None]
                if valid_r:
                    fy = [v[0] for v in valid_r]
                    fr = [v[1] for v in valid_r]
                    fh = [v[2] for v in valid_r]
                    fill_rx = fy + list(reversed(fy))
                    fill_ry = fr + list(reversed(fh))
                else:
                    fill_rx, fill_ry = [], []
                years_e, cty_exp, hic_exp, fill_ex, fill_ey, has_exp = \
                    get_exp_series(cty)
                age_block.append({
                    "years_r":  years_r,
                    "rate":     rate_c,
                    "hic_r":    hic_r,
                    "fill_rx":  fill_rx,
                    "fill_ry":  fill_ry,
                    "years_e":  years_e,
                    "cty_exp":  cty_exp,
                    "hic_exp":  hic_exp,
                    "fill_ex":  fill_ex,
                    "fill_ey":  fill_ey,
                    "country":  cty,
                    "organism": org,
                    "income":   income,
                    "has_exp":  has_exp,
                    "n":        n_c
                })
            res_block.append(age_block)
        org_block.append(res_block)
    data_table.append(org_block)

print(f"  Data table built: {len(org_list)} orgs x {len(RES_TYPES)} "
      f"res x {len(AGE_LABELS)} ages x (variable countries)")

# Serialise to JSON strings — plain strings, no f-string involvement here
data_js = json.dumps(data_table)
orgs_js = json.dumps(org_list)
ctys_js = json.dumps({o: org_to_ctys[o] for o in org_list})
res_js  = json.dumps(RES_TYPES)
age_js  = json.dumps(AGE_LABELS)

# HTML option elements — plain strings
org_opts = "\n".join(
    '          <option value="' + str(i) + '">' + org + '</option>'
    for i, org in enumerate(org_list))

first_ctys = org_to_ctys[org_list[0]] if org_list else []
cty_opts = "\n".join(
    '          <option value="' + str(i) + '">' + c + '</option>'
    for i, c in enumerate(first_ctys))

res_opts = "\n".join(
    '          <option value="' + str(i) + '">' + r + '</option>'
    for i, r in enumerate(RES_TYPES))

age_opts = "\n".join(
    '          <option value="' + str(i) + '">' + a + '</option>'
    for i, a in enumerate(AGE_LABELS))

# ============================================================
# STEP 4 — BUILD HTML
# HTML is built by string concatenation — NOT f-strings
# This avoids Python trying to parse JS curly braces as
# f-string placeholders, which caused the blank chart bug.
# ============================================================
print("\n" + "=" * 65)
print("STEP 4 — BUILDING HTML")
print("=" * 65)

PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f4f8;
    min-height: 100vh;
}
.page-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 18px 18px 28px;
    display: flex;
    flex-direction: column;
}
.title-bar {
    background: linear-gradient(135deg, #1a3a5c 0%, #2471a3 100%);
    color: #fff;
    text-align: center;
    padding: 16px 24px;
    border-radius: 10px 10px 0 0;
    border-bottom: 3px solid #1a5276;
}
.title-bar h1 { font-size: 1.40rem; font-weight: 800;
                letter-spacing: 0.5px; line-height: 1.3; }
.title-bar p  { font-size: 0.84rem; opacity: 0.82; margin-top: 4px; }
.main-row {
    display: flex;
    background: #fff;
    border-left: 1px solid #d5dde5;
    border-right: 1px solid #d5dde5;
}
.chart-col { flex: 1 1 0%; min-width: 0; padding: 8px 4px 8px 4px; }
#dual-chart { width: 100%; height: 500px; }
.ctrl-panel {
    width: 232px; flex-shrink: 0;
    background: #f7f9fc;
    border-left: 2px solid #d5dde5;
    display: flex; flex-direction: column;
    gap: 11px; padding: 16px 13px 14px;
}
.ctrl-panel .section-label {
    font-size: 0.69rem; font-weight: 700; color: #5d6d7e;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
}
.ctrl-select {
    width: 100%; padding: 7px 10px;
    border: 2px solid #e6a817; border-radius: 6px;
    background: #fff3cd; font-size: 0.83rem;
    font-weight: 700; color: #5a3e00;
    cursor: pointer; outline: none;
    appearance: none; -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%235a3e00'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center; padding-right: 28px;
}
.ctrl-select.blue-sel {
    border-color: #2980b9; background-color: #d6eaf8; color: #1a252f;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%231a252f'/%3E%3C/svg%3E");
}
.ctrl-select.green-sel {
    border-color: #1e8449; background-color: #d5f5e3; color: #145a32;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%23145a32'/%3E%3C/svg%3E");
}
.ctrl-select.purple-sel {
    border-color: #7d3c98; background-color: #e8daef; color: #4a235a;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%234a235a'/%3E%3C/svg%3E");
}
.ctrl-select:focus { box-shadow: 0 0 0 3px rgba(41,128,185,0.25); }
.legend-block {
    font-size: 0.73rem; color: #444; line-height: 1.9;
    border-top: 1px solid #d5dde5; padding-top: 10px; margin-top: 4px;
}
.leg-title { font-weight: 700; color: #1a252f; font-size: 0.75rem; margin-bottom: 5px; }
.leg-grp {
    font-size: 0.68rem; color: #888; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px; margin: 6px 0 3px;
}
.leg-line { display: flex; align-items: center; gap: 7px; margin-bottom: 2px; }
.leg-sw   { flex-shrink: 0; width: 26px; }
.leg-fill-r {
    flex-shrink: 0; width: 26px; height: 11px;
    border: 1px solid rgba(192,57,43,0.35);
    background: rgba(192,57,43,0.15);
}
"""

# JS as a plain string — NO f-string — data injected via concatenation
JS = (
    "const DATA      = " + data_js + ";\n"
    "const EQ_ORGS   = " + orgs_js + ";\n"
    "const EQ_CTYS   = " + ctys_js + ";\n"
    "const RES_TYPES = " + res_js  + ";\n"
    "const AGE_LABELS= " + age_js  + ";\n"
    """
let _orgIdx = 0;
let _ctyIdx = 0;
let _resIdx = 0;
let _ageIdx = 0;

function render() {
    // Auto-find first country with data if current has none
    let d = DATA[_orgIdx][_resIdx][_ageIdx][_ctyIdx];
    if (!d || d.years_r.length === 0) {
        const ctys = EQ_CTYS[EQ_ORGS[_orgIdx]] || [];
        for (let i = 0; i < ctys.length; i++) {
            const test = DATA[_orgIdx][_resIdx][_ageIdx][i];
            if (test && test.years_r.length > 0) {
                _ctyIdx = i;
                document.getElementById("cty-select").value = i;
                d = test;
                break;
            }
        }
    }

    if (!d || d.years_r.length === 0) {
        Plotly.react("dual-chart", [], {
            paper_bgcolor: "white", plot_bgcolor: "white",
            annotations: [{
                text: "No data available for this selection",
                x: 0.5, y: 0.5, xref: "paper", yref: "paper",
                showarrow: false, font: { size: 14, color: "#999" }
            }]
        }, { responsive: true, displayModeBar: false });
        return;
    }

    const org = d.organism;
    const cty = d.country;
    const inc = d.income;
    const rl  = RES_TYPES[_resIdx];
    const al  = AGE_LABELS[_ageIdx];

    // 1. Resistance shaded fill — left axis
    const t_rfill = {
        x: d.fill_rx, y: d.fill_ry,
        fill: "toself",
        fillcolor: "rgba(192,57,43,0.12)",
        line: { color: "rgba(0,0,0,0)", width: 0 },
        mode: "lines", type: "scatter", yaxis: "y",
        name: "Excess resistance burden",
        showlegend: false, hoverinfo: "skip"
    };

    // 2. Country resistance rate — red solid
    const t_rate = {
        x: d.years_r, y: d.rate,
        mode: "lines+markers", type: "scatter", yaxis: "y",
        name: cty + " \u2014 " + rl + " Rate (%)",
        line: { color: "#c0392b", width: 2.8 },
        marker: { size: 8, color: "#c0392b",
                  line: { color: "white", width: 1.5 } },
        hovertemplate:
            "<b>" + cty + "</b> (" + inc + ")<br>" +
            "Organism: " + org + "<br>" +
            "Age group: " + al + "<br>" +
            "Year: %{x}<br>" +
            rl + " Rate: %{y:.1f}%<extra></extra>"
    };

    // 3. HIC mean resistance — blue dashed
    const t_hic_r = {
        x: d.years_r, y: d.hic_r,
        mode: "lines+markers", type: "scatter", yaxis: "y",
        name: "HIC Mean \u2014 " + rl + " Rate (%)",
        line: { color: "#2980b9", width: 2.2, dash: "dash" },
        marker: { size: 6, color: "#2980b9", symbol: "square",
                  line: { color: "white", width: 1 } },
        hovertemplate:
            "HIC weighted mean<br>" +
            "Organism: " + org + "<br>" +
            "Age group: " + al + "<br>" +
            "Year: %{x}<br>" +
            "HIC Mean " + rl + ": %{y:.1f}%<extra></extra>"
    };

    // 4. Expenditure shaded fill — right axis, dark green
    const t_efill = d.has_exp ? {
        x: d.fill_ex, y: d.fill_ey,
        fill: "toself",
        fillcolor: "rgba(30,132,73,0.13)",
        line: { color: "rgba(0,0,0,0)", width: 0 },
        mode: "lines", type: "scatter", yaxis: "y2",
        name: "Funding gap (expenditure)",
        showlegend: false, hoverinfo: "skip"
    } : null;

    // 5. Country expenditure — dark green solid diamond
    const t_exp = d.has_exp ? {
        x: d.years_e, y: d.cty_exp,
        mode: "lines+markers", type: "scatter", yaxis: "y2",
        name: cty + " \u2014 Health Exp. (% GDP)",
        line: { color: "#1e8449", width: 2.8 },
        marker: { size: 9, color: "#1e8449", symbol: "diamond",
                  line: { color: "white", width: 1.5 } },
        hovertemplate:
            "<b>" + cty + "</b><br>" +
            "Year: %{x}<br>" +
            "Health Expenditure: %{y:.1f}% of GDP<br>" +
            "<i>WHO GHED snapshot</i><extra></extra>"
    } : null;

    // 6. HIC mean expenditure — purple dashed
    const t_hic_e = d.has_exp ? {
        x: d.years_e, y: d.hic_exp,
        mode: "lines+markers", type: "scatter", yaxis: "y2",
        name: "HIC Mean \u2014 Health Exp. (% GDP)",
        line: { color: "#7d3c98", width: 2, dash: "dash" },
        marker: { size: 6, color: "#7d3c98", symbol: "square",
                  line: { color: "white", width: 1 } },
        hovertemplate:
            "HIC mean health expenditure<br>" +
            "Year: %{x}<br>" +
            "HIC Mean: %{y:.1f}% of GDP<extra></extra>"
    } : null;

    const traces = [t_rfill, t_rate, t_hic_r];
    if (d.has_exp) { traces.push(t_efill, t_exp, t_hic_e); }

    const noExpNote = d.has_exp ? "" :
        " &nbsp;|&nbsp; <span style='color:#c0392b;'>" +
        "No WHO GHED data for " + cty + "</span>";

    const layout = {
        title: {
            text: "<b>" + rl + " Resistance & Health Expenditure \u2014 " +
                  org + " &nbsp;/&nbsp; " + cty + " vs HIC</b><br>" +
                  "<span style='font-size:11px;color:#555;font-weight:normal'>" +
                  "Age group: " + al +
                  " &nbsp;|&nbsp; Income: " + inc +
                  " &nbsp;|&nbsp; Min. 10 isolates (WHO GLASS 2023)" +
                  noExpNote + "</span>",
            x: 0.5, xanchor: "center",
            font: { size: 14, color: "#1a252f", family: "Arial Black" }
        },
        xaxis: {
            title: {
                text: "<b>Year</b>",
                font: { size: 14, color: "#1a252f", family: "Arial Black" }
            },
            tickfont: { size: 12, color: "#1a252f", family: "Arial Black" },
            dtick: 2,
            gridcolor: "#eeeeee", linecolor: "#888",
            linewidth: 1.2, showline: true, mirror: false,
            range: [2003, 2025]
        },
        yaxis: {
            title: { text: "<b>" + rl + " Rate (%)</b>",
                     font: { size: 14, color: "#c0392b", family: "Arial Black" } },
            range: [0, 105],
            tickfont: { size: 12, color: "#c0392b", family: "Arial Black" },
            gridcolor: "#eeeeee",
            linecolor: "#c0392b", linewidth: 1.5,
            showline: true, zeroline: false
        },
        yaxis2: {
            title: { text: "<b>Health Expenditure (% of GDP)</b>",
                     font: { size: 14, color: "#1e8449", family: "Arial Black" } },
            overlaying: "y", side: "right",
            range: [0, 22],
            tickfont: { size: 12, color: "#1e8449", family: "Arial Black" },
            gridcolor: "rgba(0,0,0,0)",
            linecolor: "#1e8449", linewidth: 1.5,
            showline: true, zeroline: false, showgrid: false
        },
        shapes: [{
            type: "line",
            x0: 2003, x1: 2025, y0: 50, y1: 50,
            xref: "x", yref: "y",
            line: { color: "#aaa", width: 1.2, dash: "dot" }
        }],
        annotations: [{
            text: "<i>Grey dotted = 50% resistance crisis threshold (ECDC 2022)" +
                  " &nbsp;&nbsp; Expenditure: WHO GHED snapshots 2005, 2010, " +
                  "2015, 2020\u20132022 \u2014 no interpolation applied</i>",
            x: 0.5, y: -0.13, xref: "paper", yref: "paper",
            showarrow: false, font: { size: 9.5, color: "#666" }
        }],
        showlegend: false,
        plot_bgcolor: "white",
        paper_bgcolor: "white",
        margin: { l: 68, r: 85, t: 90, b: 75 },
        autosize: true
    };

    Plotly.react("dual-chart", traces, layout,
        { responsive: true, displayModeBar: false });
}

function setOrganism(idx) {
    _orgIdx = idx;
    _ctyIdx = 0;
    const org  = EQ_ORGS[idx];
    const ctys = EQ_CTYS[org] || [];
    const sel  = document.getElementById("cty-select");
    sel.innerHTML = ctys.map((c, i) =>
        '<option value="' + i + '">' + c + '</option>').join("\\n");
    render();
}

function setCountry(idx)  { _ctyIdx = idx; render(); }
function setResType(idx)  { _resIdx = idx; render(); }
function setAgeGroup(idx) { _ageIdx = idx; render(); }

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("org-select").addEventListener("change",
        function() { setOrganism(parseInt(this.value)); });
    document.getElementById("cty-select").addEventListener("change",
        function() { setCountry(parseInt(this.value)); });
    document.getElementById("res-select").addEventListener("change",
        function() { setResType(parseInt(this.value)); });
    document.getElementById("age-select").addEventListener("change",
        function() { setAgeGroup(parseInt(this.value)); });
    render();
});
"""
)
COPYRIGHT = """
<div style="
max-width:1150px;
margin:0 auto 20px auto;
text-align:center;
font-family:'Segoe UI',Arial,sans-serif;
font-size:12px;
color:#7f8c8d;
padding-top:8px;
border-top:1px solid #dddddd;">
© 2026 PSG Institute of Medical Sciences & Research
</div>
"""
# Body HTML — plain string, dropdowns injected via concatenation
BODY_HTML = (
    """
  <div class="title-bar">
    <h1>AMR Resistance Burden vs Health Expenditure &mdash;
        Country vs High Income Countries (2004&ndash;2024)</h1>
    <p>AMR Data Challenge 2026 &middot; Phase 3 Geographic Analysis &middot;
       Nosocomial Neuroinvasive Bacterial AMR &middot;
       Resistance: Pfizer ATLAS &middot; Expenditure: WHO GHED via UNdata</p>
  </div>

  <div class="main-row">
    <div class="chart-col">
      <div id="dual-chart"></div>
    </div>
    <div class="ctrl-panel">

      <div>
        <div class="section-label">Organism</div>
        <select id="org-select" class="ctrl-select">
""" + org_opts + """
        </select>
      </div>

      <div>
        <div class="section-label">Country</div>
        <select id="cty-select" class="ctrl-select blue-sel">
""" + cty_opts + """
        </select>
      </div>

      <div>
        <div class="section-label">Resistance Type</div>
        <select id="res-select" class="ctrl-select green-sel">
""" + res_opts + """
        </select>
      </div>

      <div>
        <div class="section-label">Age Group</div>
        <select id="age-select" class="ctrl-select purple-sel">
""" + age_opts + """
        </select>
      </div>

      <div class="legend-block">
        <div class="leg-title">Legend</div>

        <div class="leg-grp">Left Axis &mdash; Resistance</div>
        <div class="leg-line">
          <div class="leg-sw">
            <div style="background:#c0392b;height:3px;margin-top:5px;"></div>
          </div>
          <span>Country resistance rate (%)</span>
        </div>
        <div class="leg-line">
          <div class="leg-sw">
            <div style="border-top:3px dashed #2980b9;height:0;margin-top:5px;"></div>
          </div>
          <span>HIC weighted mean rate (%)</span>
        </div>
        <div class="leg-line">
          <div class="leg-fill-r"></div>
          <span>Excess resistance burden</span>
        </div>
        <div class="leg-line">
          <div class="leg-sw">
            <div style="border-top:2px dotted #aaa;height:0;margin-top:5px;"></div>
          </div>
          <span>50% crisis threshold</span>
        </div>

        <div class="leg-grp">Right Axis &mdash; Expenditure</div>
        <div class="leg-line">
          <div class="leg-sw">
            <div style="background:#1e8449;height:3px;margin-top:5px;"></div>
          </div>
          <span>Country health exp. (% GDP)</span>
        </div>
        <div class="leg-line">
          <div class="leg-sw">
            <div style="border-top:3px dashed #7d3c98;height:0;margin-top:5px;"></div>
          </div>
          <span>HIC mean expenditure (% GDP)</span>
        </div>
        <div class="leg-line">
          <div style="flex-shrink:0;width:26px;height:11px;
               border:1px solid rgba(30,132,73,0.4);
               background:rgba(30,132,73,0.15);"></div>
          <span>Funding gap area</span>
        </div>
      </div>

    </div>
  </div>

  <div style="
      background:#f8f9fa;
      border-left:7px solid #2980b9;
      border-bottom:1px solid #d5dde5;
      border-right:1px solid #d5dde5;
      border-radius:0 0 10px 10px;
      padding:14px 20px 14px 18px;
      max-height:200px;
      overflow-y:auto;
      font-size:0.84rem;
      line-height:1.72;
      color:#1a252f;
      box-shadow:0 4px 12px rgba(0,0,0,0.07);
  ">
<strong>How to read this chart.</strong>
This is a dual Y-axis chart plotting two independent variables on the same time axis
(2004&ndash;2024) for direct visual comparison of resistance burden and healthcare investment.
<br><br>
<strong>Left Y-axis &mdash; Resistance Rate (%):</strong><br>
&bull;&nbsp;<strong style="color:#c0392b;">Red solid line</strong> &mdash;
selected country's MDR/XDR/PDR rate per year. Higher = more isolates
resistant to multiple antibiotic classes that year.<br>
&bull;&nbsp;<strong style="color:#2980b9;">Blue dashed line</strong> &mdash;
weighted mean resistance rate across all HIC countries &mdash; the benchmark
achievable in well-resourced healthcare systems.<br>
&bull;&nbsp;<span style="background:rgba(192,57,43,0.18);padding:1px 5px;border-radius:3px;">
Pink shaded area</span> &mdash; excess resistance burden (country minus HIC). Wider = greater disparity.<br>
&bull;&nbsp;<em>Grey dotted line</em> &mdash; 50% crisis threshold (ECDC AMR Threat Report 2022).
<br><br>
<strong>Right Y-axis &mdash; Health Expenditure (% of GDP):</strong><br>
&bull;&nbsp;<strong style="color:#1e8449;">Green solid line (diamond markers)</strong> &mdash;
selected country's health expenditure as % of GDP.
WHO GHED via UNdata SYB68 (May 2025) &mdash; snapshots 2005, 2010, 2015, 2020, 2021, 2022 only.<br>
&bull;&nbsp;<strong style="color:#7d3c98;">Purple dashed line</strong> &mdash;
mean expenditure % of GDP across all HIC countries for those same years.<br>
&bull;&nbsp;<span style="background:rgba(30,132,73,0.15);padding:1px 5px;border-radius:3px;">
Green shaded area</span> &mdash; funding gap: how much less the country spends vs HIC mean.
<br><br>
<strong>What to look for:</strong>
When the green line (spending) goes down while the red line (resistance) goes up,
this is visual evidence of the inverse relationship between healthcare investment and AMR burden.
Countries with wide green shading AND wide pink shading face a
<em>structural double disadvantage</em> &mdash; underfunded systems sustaining higher resistance.
<br><br>
<strong>Data sources:</strong>
Resistance &mdash; Pfizer ATLAS 2004&ndash;2024.
Expenditure &mdash; WHO GHED via UNdata SYB68 (May 2025).
MDR/XDR/PDR &mdash; Magiorakos <em>et al.</em> CMI 2012;18:268&ndash;281.
HIC &mdash; World Bank FY2026.
Equity gap &mdash; Laxminarayan <em>et al.</em> Lancet Infect Dis 2013.
Crisis threshold &mdash; ECDC AMR Threat Report 2022.
Min. isolates &mdash; &ge;10 per cell (WHO GLASS 2023).
<br><br>
<strong>Definitions:</strong>
<strong>MDR</strong> &mdash; &ge;3 antibiotic classes non-susceptible &nbsp;|&nbsp;
<strong>XDR</strong> &mdash; &le;2 classes susceptible &nbsp;|&nbsp;
<strong>PDR</strong> &mdash; 0 classes susceptible.
<em>Magiorakos et al. CMI 2012.</em>
<strong>HIC</strong> &mdash; World Bank FY2026. &nbsp;
<strong>Min. isolates</strong> &mdash; 10 (WHO GLASS 2023). &nbsp;
Expenditure data: WHO GHED snapshots 2005, 2010, 2015, 2020&ndash;2022.
No interpolation applied.
  </div>
"""
)

# Assemble full HTML — pure string concatenation
HTML = (
    "<!DOCTYPE html>\n"
    "<html lang='en'>\n"
    "<head>\n"
    "<meta charset='UTF-8'/>\n"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>\n"
    "<title>AMR Resistance Burden vs Health Expenditure - Dual Axis (2004-2024)</title>\n"
    "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>\n"
    "<style>\n"
    + PAGE_CSS +
    "\n</style>\n"
    "</head>\n"
    "<body>\n"
    "<div class='page-wrap'>\n"
    + BODY_HTML +
    "\n</div>\n"
    "<script>\n"
    + JS +
    "\n</script>\n"
    + COPYRIGHT +
    "</body>\n"
    "</html>"
)

out_path = os.path.join(LAY_DIR, "equity_gap_dual_axis.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"  Saved: {out_path}")

print("\n" + "=" * 65)
print("PHASE 3 EQUITY GAP DUAL AXIS — COMPLETE")
print("=" * 65)
print("""
  Output:
    HTML: phase3_outputs/layout/equity_gap_dual_axis.html

  FIX APPLIED:
    HTML built via string concatenation — no nested f-strings.
    JS curly braces are now safe and will not be misread by Python.
    Auto-skip added: if default country has no data, first country
    with data is selected automatically on load.

  Colours:
    Resistance  — red (#c0392b) + blue (#2980b9) + pink fill
    Expenditure — dark green (#1e8449) + purple (#7d3c98) + green fill

  Margin: b=95, annotation y=-0.20 (no X-axis label overlap)
""")

import pandas as pd
import numpy as np
import os
import warnings
import urllib.request
warnings.filterwarnings("ignore")

import geopandas as gpd
import plotly.graph_objects as go

OUT_DIR   = "phase3_outputs"
CSV_DIR   = os.path.join(OUT_DIR, "csv_outputs")
WHO_DIR   = os.path.join(OUT_DIR, "who_region")
LAY_DIR   = os.path.join(OUT_DIR, "layout")
os.makedirs(LAY_DIR, exist_ok=True)

TARGET_ORGANISMS = [
    "Klebsiella pneumoniae", "Acinetobacter baumannii",
    "Escherichia coli", "Staphylococcus aureus",
    "Pseudomonas aeruginosa", "Enterobacter cloacae",
    "Enterococcus faecium", "Staphylococcus epidermidis",
    "Serratia marcescens", "Citrobacter koseri",
    "Streptococcus pneumoniae", "Haemophilus influenzae",
    "Salmonella spp"
]

WHO_REGIONS = ["AFRO", "AMRO", "EMRO", "EURO", "SEARO", "WPRO"]
WHO_REGION_COLORS = {
    "AFRO": "#e41a1c", "AMRO": "#377eb8", "EMRO": "#ff7f00",
    "EURO": "#4daf4a", "SEARO": "#984ea3", "WPRO": "#a65628"
}

# Yellow-Orange-Red colour scale (same as originals)
COLORSCALE = [
    [0.0, "#ffffcc"], [0.25, "#fecc5c"],
    [0.5,  "#fd8d3c"], [0.75, "#f03b20"],
    [1.0,  "#bd0026"]
]

NAME_MAP = {
    "United States of America": "United States",
    "Republic of Korea":        "Korea, South",
    "Czechia":                  "Czech Republic",
    "Russian Federation":       "Russia",
    "Slovakia":                 "Slovak Republic",
    "Dominican Rep.":           "Dominican Republic",
    "Côte d'Ivoire":            "Ivory Coast",
}

# ============================================================
# LOAD SHAPEFILE
# ============================================================
print("=" * 60)
print("LOADING SHAPEFILE")
print("=" * 60)
SHP = "ne_110m_admin_0_countries.zip"
if not os.path.exists(SHP):
    urllib.request.urlretrieve(
        "https://naciscdn.org/naturalearth/110m/cultural/"
        "ne_110m_admin_0_countries.zip", SHP)
world = gpd.read_file(f"zip://{SHP}")
world["Country_Std"] = world["NAME"].apply(lambda x: NAME_MAP.get(x, x))
iso3 = dict(zip(world["Country_Std"], world["ADM0_A3"]))
world["rep"] = world.geometry.representative_point()
centroids = {r["Country_Std"]: (r["rep"].x, r["rep"].y)
             for _, r in world.iterrows()}
print(f"  {len(world)} polygons loaded")


OUT_DIR   = "phase3_outputs"
CSV_DIR   = os.path.join(OUT_DIR, "csv_outputs")

# ============================================================
# WHO region year trend — from temporal dataset
# ============================================================


# ============================================================
# LOAD REQUIRED DATA FROM DISK
# (df_temporal and heatmap are not in memory — load from CSV)
# ============================================================

OUT_DIR = "phase3_outputs"          # correct output directory
CSV_DIR = os.path.join(OUT_DIR, "csv_outputs")
WHO_DIR = os.path.join(OUT_DIR, "who_region")
os.makedirs(WHO_DIR, exist_ok=True)

df_temporal = pd.read_csv(
    os.path.join(CSV_DIR, "temporal_mxp_age_country_year.csv"))
print(f"  Temporal CSV loaded: {len(df_temporal):,} rows")

heatmap = pd.read_csv(
    "PHASE2_outputs/FULL_country_organism_heatmap_data.csv")
print(f"  Heatmap loaded: {len(heatmap):,} rows")

# Since Age_Group column now exists (3x rows), filter to All only
# so WHO region aggregation doesn't triple-count isolates
if "Age_Group" in df_temporal.columns:
    df_temporal = df_temporal[df_temporal["Age_Group"] == "All"].copy()
    print(f"  Filtered to Age_Group='All': {len(df_temporal):,} rows")

def save_csv(df, name, subdir="csv_outputs"):
    path = os.path.join(OUT_DIR, subdir, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df):,} rows)")

def write_with_footer(fig, path, footnote_html):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    box = f"""
<div style="max-width:1150px;margin:18px auto 32px auto;
  padding:18px 26px;background:#f8f9fa;
  border-left:7px solid #2980b9;border-radius:7px;
  font-family:'Segoe UI',Arial,sans-serif;font-size:14px;
  line-height:1.7;color:#1a252f;
  box-shadow:2px 2px 8px rgba(0,0,0,0.10);">
{footnote_html}
</div>
<div style="max-width:1150px;margin:0 auto 24px auto;
  text-align:center;font-family:'Segoe UI',Arial,sans-serif;
  font-size:12px;color:#7f8c8d;padding:8px 0;">
  &copy; 2026 PSG Institute of Medical Sciences &amp; Research
</div>"""
    html = html.replace("</body>", box + "\n</body>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓  {path}")
who_yr_trend_rows = []
for org in TARGET_ORGANISMS:
    sub_t = df_temporal[df_temporal["Organism"] == org]
    if len(sub_t) == 0:
        continue
    # Add WHO region from heatmap country lookup
    region_lookup = dict(zip(heatmap["Country"], heatmap["WHO_Region"]))
    sub_t = sub_t.copy()
    sub_t["WHO_Region"] = sub_t["Country"].map(region_lookup)
    for yr in sorted(sub_t["Year"].unique()):
        yr_d = sub_t[sub_t["Year"] == yr]
        for region in ["AFRO", "AMRO", "EMRO", "EURO", "SEARO", "WPRO"]:
            r_d = yr_d[yr_d["WHO_Region"] == region]
            if len(r_d) == 0:
                continue
            n   = int(r_d["N_Isolates"].sum())
            mdr = int(r_d["MDR_Count"].sum())
            who_yr_trend_rows.append({
                "Organism":      org,
                "WHO_Region":    region,
                "Year":          yr,
                "N_Isolates":    n,
                "MDR_Count":     mdr,
                "Weighted_MDR_%": round(mdr / n * 100, 2) if n > 0 else None
            })

df_who_yr = pd.DataFrame(who_yr_trend_rows)
save_csv(df_who_yr, "who_region_organism_mdr_by_year")


# ============================================================
#  WHO REGION YEAR TREND — highlighted, colourful, footnote
# ============================================================

df_who_yr  = pd.read_csv(os.path.join(CSV_DIR,
    "who_region_organism_mdr_by_year.csv"))


print("\n" + "="*60)
print("4. WHO REGION YEAR TREND")
print("="*60)

orgs_who = [o for o in TARGET_ORGANISMS
            if o in df_who_yr["Organism"].unique()]

fig_who = go.Figure()

for o_idx, org in enumerate(orgs_who):
    sub_o = df_who_yr[df_who_yr["Organism"]==org]
    visible = (o_idx==0)
    for region in WHO_REGIONS:
        sub_r = sub_o[sub_o["WHO_Region"]==region]
        if len(sub_r)==0:
            continue
        color = WHO_REGION_COLORS.get(region,"#aaaaaa")
        fig_who.add_trace(go.Scatter(
            x=sub_r["Year"],
            y=sub_r["Weighted_MDR_%"],
            mode="lines+markers",
            name=f"<b>{region}</b>",
            line=dict(color=color, width=2.8),
            marker=dict(size=9, color=color,
                        line=dict(color="white",width=1.5)),
            visible=visible,
            legendgroup=f"{org}_{region}",
            hovertemplate=(
                f"<b>{region}</b><br>"
                f"Organism: {org}<br>"
                "Year: %{x}<br>"
                "Weighted MDR: %{y:.1f}%<extra></extra>")
        ))
        # label at last data point
        if len(sub_r)>0:
            last = sub_r.sort_values("Year").iloc[-1]
            fig_who.add_trace(go.Scatter(
                x=[last["Year"]+0.3],
                y=[last["Weighted_MDR_%"]],
                mode="text",
                text=[f"<b>{region}</b>"],
                textfont=dict(size=10, color=color,
                              family="Arial Black"),
                visible=visible,
                showlegend=False,
                hoverinfo="skip",
                legendgroup=f"{org}_{region}_lbl"
            ))

# count traces per organism (varies because not all regions present)
traces_per_org=[]
for org in orgs_who:
    sub_o = df_who_yr[df_who_yr["Organism"]==org]
    n=0
    for region in WHO_REGIONS:
        sub_r = sub_o[sub_o["WHO_Region"]==region]
        if len(sub_r)>0:
            n+=2  # line + label
    traces_per_org.append(n)

cumulative=[0]
for n in traces_per_org:
    cumulative.append(cumulative[-1]+n)

who_buttons=[]
for o_idx, org in enumerate(orgs_who):
    vis=[False]*len(fig_who.data)
    start=cumulative[o_idx]
    end  =cumulative[o_idx+1]
    for k in range(start,end):
        vis[k]=True
    who_buttons.append(dict(
        label=org, method="update",
        args=[{"visible":vis},
              "title",{"text":(
                  f"<b>WHO Region MDR Trends — {org} — 2004–2024</b><br>"
                  "<span style='font-size:12px;font-weight:normal'>"
                  "Weighted MDR rate per WHO region per year"
                  "</span><br>"
                  "<span style='font-size:11px;font-weight:normal;color:#888'>"
                  "How to use: Select an organism from the amber dropdown at the top-left "
                  "to switch between all 13 target organisms.</span>"),
                  # Plotly's update() REPLACES the whole title object,
                  # not just 'text' — without restating x/xanchor/font
                  # here, every click after the first snaps the title
                  # back to Plotly's default left-aligned position
                  # (x=0.05), which is why only the initial organism
                  # (Klebsiella) looked centered and every other
                  # organism's title collided with the dropdown.
                  "x":0.5, "xanchor":"center",
                  "font":{"size":19,"color":"#1a252f",
                          "family":"Arial Black"}}]))

# make first organism visible
for k in range(cumulative[0],cumulative[1]):
    fig_who.data[k].visible=True

first_who_org = orgs_who[0] if orgs_who else ""

fig_who.update_layout(
    title=dict(
        text=(f"<b>WHO Region MDR Trends — "
              f"{first_who_org} — 2004–2024</b><br>"
              "<span style='font-size:13px;font-weight:normal;color:#444'>"
              "Weighted MDR rate per region per year | "
              "Select organism from amber dropdown</span><br>"
              "<span style='font-size:11px;font-weight:normal;color:#888'>"
              "How to use: Select an organism from the amber dropdown at the top-left "
              "to switch between all 13 target organisms.</span>"),
        x=0.5, xanchor="center",
        font=dict(size=19, color="#1a252f",
                  family="Arial Black")),
    xaxis=dict(
        title=dict(text="<b>Year</b>",
                   font=dict(size=14,color="#1a252f",
                             family="Arial Black")),
        tickfont=dict(size=12,color="#1a252f",family="Arial"),
        dtick=2, gridcolor="#dddddd",
        linecolor="#444",linewidth=1.5,
        showline=True, mirror=True),
    yaxis=dict(
        title=dict(text="<b>Weighted MDR Rate (%)</b>",
                   font=dict(size=14,color="#1a252f",
                             family="Arial Black")),
        tickfont=dict(size=12,color="#1a252f",family="Arial"),
        range=[0,108], gridcolor="#dddddd",
        linecolor="#444",linewidth=1.5,
        showline=True, mirror=True),
    shapes=[dict(
        type="line",x0=2003,x1=2025,y0=50,y1=50,
        line=dict(color="#555555",width=1.2,dash="dot"),
        xref="x",yref="y")],
    legend=dict(
        title=dict(text="<b>WHO Region</b>",
                   font=dict(size=13,color="#1a252f",
                             family="Arial Black")),
        font=dict(size=12,color="#1a252f",family="Arial"),
        bgcolor="white",bordercolor="#555",borderwidth=1.5,
        x=1.01,y=1,
        itemsizing="constant"),
    plot_bgcolor="#fdfefe",
    paper_bgcolor="white",
    height=580,
    margin=dict(l=70,r=180,t=160,b=80),
    updatemenus=[dict(
        buttons=who_buttons,
        direction="down",showactive=True,
        x=0.01, y=1.23, xanchor="left", yanchor="top",
        bgcolor="#fff3cd",bordercolor="#e6a817",
        borderwidth=2.5,
        font=dict(size=12,color="#5a3e00",
                  family="Arial Black"))],
    annotations=[
        dict(text="<b>Organism:</b>",
             x=0.01,y=1.28,xref="paper",yref="paper",
             showarrow=False,
             font=dict(size=12,color="#5a3e00",
                       family="Arial Black")),
        dict(text="<i>Dotted line = 50% MDR threshold (ECDC 2022)</i>",
             x=0.5,y=-0.10,xref="paper",yref="paper",
             showarrow=False,
             font=dict(size=11,color="#555"))
    ]
)

WHO_FN = """
<b>About this chart.</b>
For the organism selected in the amber dropdown, each coloured line
shows the <b>weighted MDR rate (%)</b> for one WHO region from 2004 to
2024. "Weighted" means the total MDR isolates across all countries in
that region are divided by the total isolates in that region for that
year — so larger countries (by isolate count) contribute
proportionally more.
<br><br>
<b>Regions and colours:</b>
<span style="color:#e41a1c"><b>AFRO</b></span> (Africa) |
<span style="color:#377eb8"><b>AMRO</b></span> (Americas) |
<span style="color:#ff7f00"><b>EMRO</b></span> (Eastern Mediterranean) |
<span style="color:#4daf4a"><b>EURO</b></span> (Europe) |
<span style="color:#984ea3"><b>SEARO</b></span> (South-East Asia) |
<span style="color:#a65628"><b>WPRO</b></span> (Western Pacific).
<br><br>
Only country-year cells with ≥10 isolates are included in each
regional weighted average (WHO GLASS 2023 threshold). Region-year
combinations with zero qualifying countries are not plotted.
The grey dotted horizontal line marks the 50% MDR threshold
(ECDC AMR Threat Report, 2022).
<br><br>
"""

write_with_footer(
    fig_who,
    os.path.join(WHO_DIR, "who_region_year_trend.html"),
    WHO_FN)


