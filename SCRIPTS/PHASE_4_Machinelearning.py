# ============================================================
# AMR DATA CHALLENGE 2026 — PHASE 4: ML CLASSIFICATION (FINAL, CONSOLIDATED)
# Task 1: Predict MDR_v2 (binary) from surveillance metadata
# 5-model benchmark: LR, RF, GBDT, XGBoost, CatBoost
# 
# ============================================================
# REFERENCES:
#   MDR definition: Magiorakos et al. CMI 2012;18(3):268-281
#   Model shortlist justified via literature review:
#     LR: baseline (14/25 studies historically), weak on real-world AST metadata (mean AUROC 0.68)
#     RF: real-world AST review mean AUROC 0.75; WGS review mean AUC 0.89
#     GBDT: highest mean AUROC (0.80) in real-world AST systematic review (PMC11856330)
#     XGBoost: best performer in ATLAS-dataset study (Nature Sci Rep 2025)
#     CatBoost: top performer in N. gonorrhoeae AMR study (PMC12295555), MALDI-TOF study (PMC11817502)
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
ATLAS_PATH  = "phase1_outputs/atlas_filtered_mdr_flagged.csv"   # confirmed path
OUT_DIR     = "phase4_outputs"
FIG_DIR     = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_SEED = 42
TRAIN_END_YEAR = 2019
THRESHOLD = 0.5
N_BOOTSTRAP = 1000
SEEDS_FOR_VARIANCE = [42, 7, 123, 2024, 99]
CALIB_START_YEAR = 2018

MODEL_COLORS = {
    "Logistic_Regression": "#7f7f7f", "Random_Forest": "#1f77b4",
    "GBDT": "#2ca02c", "XGBoost": "#ff7f0e", "CatBoost": "#d62728"
}

def save_csv(df, name, outdir=OUT_DIR):
    path = os.path.join(outdir, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df):,} rows)")

# ============================================================
# STEP 0 — LOAD DATA
# ============================================================
print("="*65); print("STEP 0 — LOADING PHASE 1 OUTPUT"); print("="*65)
atlas = pd.read_csv(ATLAS_PATH, low_memory=False)
print(f"  Loaded: {len(atlas):,} rows x {atlas.shape[1]} cols")

# ============================================================
# STEP 1 — Organism_Label 
# ============================================================
TARGET_ORGANISMS = [
    "Klebsiella pneumoniae", "Acinetobacter baumannii", "Escherichia coli",
    "Staphylococcus aureus", "Pseudomonas aeruginosa", "Enterobacter cloacae",
    "Enterococcus faecium", "Staphylococcus epidermidis", "Serratia marcescens",
    "Citrobacter koseri", "Streptococcus pneumoniae", "Haemophilus influenzae",
    "Salmonella spp"
]
def get_organism_label(species_val):
    for org in TARGET_ORGANISMS:
        if org.lower() in str(species_val).lower():
            return org
    return str(species_val)
atlas["Organism_Label"] = atlas["Species"].apply(get_organism_label)
print(f"  Organism_Label created. Unique organisms: {atlas['Organism_Label'].nunique()}")

# ============================================================
# STEP 2 — Recompute MDR_v2 
# ============================================================
print("\n" + "="*65); print("STEP 2 — RECOMPUTING MDR_v2"); print("="*65)

MDR_CLASSES = {
    "Penicillins": ["Ampicillin_I","Amoxycillin clavulanate_I","Ampicillin sulbactam_I",
                    "Piperacillin tazobactam_I","Penicillin_I"],
    "Cephalosporins": ["Cefepime_I","Ceftazidime_I","Ceftriaxone_I","Ceftaroline_I",
                       "Ceftazidime avibactam_I","Ceftolozane tazobactam_I"],
    "Monobactams": ["Aztreonam_I"],
    "Carbapenems": ["Meropenem_I","Imipenem_I","Doripenem_I"],
    "Fluoroquinolones": ["Levofloxacin_I","Ciprofloxacin_I"],
    "Aminoglycosides": ["Amikacin_I","Gentamicin_I"],
    "Glycopeptides": ["Vancomycin_I","Teicoplanin_I"],
    "Polymyxins": ["Colistin_I"],
    "Tetracyclines": ["Minocycline_I","Tigecycline_I"],
    "Trimethoprim_Sulfonamides": ["Trimethoprim sulfa_I"],
    "Oxazolidinones": ["Linezolid_I"],
    "Lipopeptides": ["Daptomycin_I"],
    "Penicillinase_Resistant_Penicillins": ["Oxacillin_I"],
    "Lincosamides": ["Clindamycin_I"],
    "Macrolides": ["Erythromycin_I"]
}
TOTAL_CLASSES = len(MDR_CLASSES)
atlas_cols = set(atlas.columns)

def compute_mxp(row, df_cols):
    ns_classes, skipped = [], []
    for cls, drugs in MDR_CLASSES.items():
        available = [d for d in drugs if d in df_cols]
        if not available:
            skipped.append(cls); continue
        vals = [row[d] for d in available if pd.notna(row[d])]
        if not vals:
            skipped.append(cls); continue
        if any(v in ["Resistant","Intermediate"] for v in vals):
            ns_classes.append(cls)
    return 1 if len(ns_classes) >= 3 else 0

print("  Processing...")
atlas["MDR_v2"] = atlas.apply(lambda row: compute_mxp(row, atlas_cols), axis=1)
print(f"  MDR_v2 positive: {atlas['MDR_v2'].sum():,} ({atlas['MDR_v2'].mean()*100:.2f}%)")

# ============================================================
# STEP 3 — FEATURES + STEP 4 — TEMPORAL SPLIT
# ============================================================
print("\n" + "="*65); print("STEP 3-4 — FEATURES & TEMPORAL SPLIT"); print("="*65)

FEATURES_CATEGORICAL = ["Country", "Organism_Label", "Age_Category", "Source", "Gender"]
FEATURE_YEAR = "Year"
TARGET = "MDR_v2"

model_df = atlas[FEATURES_CATEGORICAL + [FEATURE_YEAR, TARGET]].copy()

train_df = model_df[model_df[FEATURE_YEAR] <= TRAIN_END_YEAR].reset_index(drop=True)
test_df  = model_df[model_df[FEATURE_YEAR] >  TRAIN_END_YEAR].reset_index(drop=True)
y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

print(f"  Train: {len(train_df):,} rows (<= {TRAIN_END_YEAR})  MDR rate: {y_train.mean()*100:.2f}%")
print(f"  Test : {len(test_df):,} rows (>  {TRAIN_END_YEAR})   MDR rate: {y_test.mean()*100:.2f}%")

save_csv(train_df, "training_dataset")
save_csv(test_df, "test_dataset")

# ============================================================
# STEP 5 — ENCODING
# ============================================================
print("\n" + "="*65); print("STEP 5 — ENCODING"); print("="*65)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[FEATURES_CATEGORICAL])
X_train_ohe = encoder.transform(train_df[FEATURES_CATEGORICAL])
X_test_ohe  = encoder.transform(test_df[FEATURES_CATEGORICAL])
X_train_final = np.hstack([X_train_ohe, train_df[[FEATURE_YEAR]].values])
X_test_final  = np.hstack([X_test_ohe,  test_df[[FEATURE_YEAR]].values])
print(f"  One-hot feature count (incl. Year): {X_train_final.shape[1]}")

ohe_feature_names = list(encoder.get_feature_names_out(FEATURES_CATEGORICAL)) + [FEATURE_YEAR]
np.save(os.path.join(OUT_DIR, "X_train_encoded.npy"), X_train_final)
np.save(os.path.join(OUT_DIR, "X_test_encoded.npy"), X_test_final)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
pd.Series(ohe_feature_names).to_csv(os.path.join(OUT_DIR, "encoded_feature_names.csv"), index=False, header=["feature_name"])
print(f"  Saved: X_train_encoded.npy, X_test_encoded.npy, y_train.npy, y_test.npy, encoded_feature_names.csv")

X_train_cb = train_df[FEATURES_CATEGORICAL + [FEATURE_YEAR]].copy()
X_test_cb  = test_df[FEATURES_CATEGORICAL + [FEATURE_YEAR]].copy()
cat_feature_idx = list(range(len(FEATURES_CATEGORICAL)))

# ============================================================
# STEP 6 — CLASS WEIGHTING
# ============================================================
n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos
sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)
print(f"\n  scale_pos_weight = {scale_pos_weight:.3f}")

# ============================================================
# STEP 7 — TRAIN ALL FIVE MODELS
# ============================================================
print("\n" + "="*65); print("STEP 7 — TRAINING MODELS"); print("="*65)

models = {}

print("  Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_final, y_train)
models["Logistic_Regression"] = ("ohe", lr)

print("  Random Forest...")
rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train_final, y_train)
models["Random_Forest"] = ("ohe", rf)

print("  GBDT...")
gbdt = GradientBoostingClassifier(random_state=RANDOM_SEED)
gbdt.fit(X_train_final, y_train, sample_weight=sample_weights_train)
models["GBDT"] = ("ohe", gbdt)

print("  XGBoost...")
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED,
                               eval_metric="logloss", n_jobs=-1)
xgb_model.fit(X_train_final, y_train)
models["XGBoost"] = ("ohe", xgb_model)

print("  CatBoost...")
cb_model = CatBoostClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED,
                               cat_features=cat_feature_idx, verbose=False)
cb_model.fit(X_train_cb, y_train)
models["CatBoost"] = ("native", cb_model)

print("  All 5 models trained.")

# ============================================================
# STEP 8 — EVALUATE + VISUALIZE
# ============================================================
print("\n" + "="*65); print("STEP 8 — EVALUATION"); print("="*65)

results = []
y_proba_store = {}
y_pred_store = {}

for name, (enc_type, model) in models.items():
    X_eval = X_test_final if enc_type == "ohe" else X_test_cb
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    y_proba_store[name] = y_proba
    y_pred_store[name] = y_pred

    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_proba)

    results.append({"Model": name, "AUROC": round(auroc,4), "AUPRC": round(auprc,4),
                     "Precision": round(prec,4), "Recall": round(rec,4),
                     "F1": round(f1,4), "Brier_Score": round(brier,4)})
    print(f"  {name:22s} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  Brier={brier:.4f}")

results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False).reset_index(drop=True)
save_csv(results_df, "PHASE4_model_comparison_table")

print("\n  Generating figures...")
metrics_to_plot = ["AUROC","AUPRC","Precision","Recall","F1","Brier_Score"]
fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(metrics_to_plot)); width = 0.15
for i, row in results_df.iterrows():
    ax.bar(x+i*width, [row[m] for m in metrics_to_plot], width, label=row["Model"],
           color=MODEL_COLORS.get(row["Model"]))
ax.set_xticks(x+width*2); ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score"); ax.set_title("Phase 4 — Model Comparison (Test Set 2020-2024)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"01_metric_comparison_bar.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,7))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, y_proba_store[name])
    av = results_df.loc[results_df["Model"]==name,"AUROC"].values[0]
    ax.plot(fpr, tpr, label=f"{name} (AUROC={av:.3f})", color=MODEL_COLORS.get(name))
ax.plot([0,1],[0,1],"--",color="black",alpha=0.5,label="Chance")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"02_roc_curves.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,7))
for name in models:
    p, r, _ = precision_recall_curve(y_test, y_proba_store[name])
    av = results_df.loc[results_df["Model"]==name,"AUPRC"].values[0]
    ax.plot(r, p, label=f"{name} (AUPRC={av:.3f})", color=MODEL_COLORS.get(name))
ax.axhline(y_test.mean(), linestyle="--", color="black", alpha=0.5, label=f"Baseline={y_test.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curves")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"03_precision_recall_curves.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,7))
for name in models:
    fp, mp = calibration_curve(y_test, y_proba_store[name], n_bins=10, strategy="quantile")
    bv = results_df.loc[results_df["Model"]==name,"Brier_Score"].values[0]
    ax.plot(mp, fp, marker="o", label=f"{name} (Brier={bv:.3f})", color=MODEL_COLORS.get(name))
ax.plot([0,1],[0,1],"--",color="black",alpha=0.5,label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Observed Fraction Positive")
ax.set_title("Calibration Curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"04_calibration_curves.png"), dpi=150); plt.close()

fig, axes = plt.subplots(1,5, figsize=(22,4.5))
for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, y_pred_store[name])
    ax.imshow(cm, cmap="Blues"); ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Non-MDR","MDR"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Non-MDR","MDR"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",
                     color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=9)
plt.suptitle(f"Confusion Matrices (Threshold={THRESHOLD})")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"05_confusion_matrices.png"), dpi=150); plt.close()
print("  5 figures saved to figures/")

# ============================================================
# STEP 9 — SAVE MODELS
# ============================================================
for name, (enc_type, model) in models.items():
    joblib.dump(model, os.path.join(OUT_DIR, f"model_{name}.joblib"))
joblib.dump(encoder, os.path.join(OUT_DIR, "onehot_encoder.joblib"))
print("\n  Models + encoder saved.")

# ============================================================
#  UNSEEN-COUNTRY QUANTIFICATION
# ============================================================
print("\n" + "="*65); print("UNSEEN COUNTRIES IN TEST SET"); print("="*65)

train_countries = set(train_df["Country"].unique())
test_countries  = set(test_df["Country"].unique())
unseen_countries = test_countries - train_countries
unseen_mask = test_df["Country"].isin(unseen_countries).values
n_unseen = unseen_mask.sum()
print(f"  Unseen countries: {sorted(unseen_countries)}")
print(f"  Affected test isolates: {n_unseen:,} ({n_unseen/len(test_df)*100:.2f}%)")

gap3_results = []
for name in models:
    y_proba = y_proba_store[name]; y_pred = y_pred_store[name]
    full_auroc = roc_auc_score(y_test, y_proba)
    seen_mask = ~unseen_mask
    excl_auroc = roc_auc_score(y_test[seen_mask], y_proba[seen_mask]) if len(set(y_test[seen_mask]))>1 else np.nan
    gap3_results.append({"Model": name, "AUROC_full_test": round(full_auroc,4),
                          "AUROC_excl_unseen": round(excl_auroc,4),
                          "Delta": round(excl_auroc-full_auroc,4),
                          "N_unseen_isolates": int(n_unseen)})
gap3_df = pd.DataFrame(gap3_results)
save_csv(gap3_df, "Unseen_country_impact")

# ============================================================
#  BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("\n" + "="*65); print(f"BOOTSTRAP CI ({N_BOOTSTRAP} resamples)"); print("="*65)

rng = np.random.RandomState(RANDOM_SEED)
n_test = len(y_test)
boot_metrics = {name: {"AUROC":[], "AUPRC":[], "F1":[], "Brier":[]} for name in models}
boot_diff = []

for b in range(N_BOOTSTRAP):
    idx = rng.randint(0, n_test, n_test)
    y_boot = y_test[idx]
    if len(set(y_boot)) < 2:
        continue
    for name in models:
        pb = y_proba_store[name][idx]
        preb = (pb >= THRESHOLD).astype(int)
        boot_metrics[name]["AUROC"].append(roc_auc_score(y_boot, pb))
        boot_metrics[name]["AUPRC"].append(average_precision_score(y_boot, pb))
        boot_metrics[name]["F1"].append(f1_score(y_boot, preb, zero_division=0))
        boot_metrics[name]["Brier"].append(brier_score_loss(y_boot, pb))
    boot_diff.append(roc_auc_score(y_boot, y_proba_store["CatBoost"][idx]) -
                      roc_auc_score(y_boot, y_proba_store["XGBoost"][idx]))

gap1_results = []
for name in models:
    row = {"Model": name}
    for metric in ["AUROC","AUPRC","F1","Brier"]:
        vals = np.array(boot_metrics[name][metric])
        row[f"{metric}_mean"] = round(vals.mean(),4)
        row[f"{metric}_CI_lower"] = round(np.percentile(vals,2.5),4)
        row[f"{metric}_CI_upper"] = round(np.percentile(vals,97.5),4)
    gap1_results.append(row)
gap1_df = pd.DataFrame(gap1_results)
save_csv(gap1_df, "Bootstrap_confidence_intervals")

diff_arr = np.array(boot_diff)
ci_l, ci_u = np.percentile(diff_arr,2.5), np.percentile(diff_arr,97.5)
pct_fav = (diff_arr>0).mean()*100
print(f"  CatBoost-XGBoost AUROC diff: mean={diff_arr.mean():+.4f}  95% CI=[{ci_l:+.4f}, {ci_u:+.4f}]  "
      f"favoring CatBoost: {pct_fav:.1f}%")
print(f"  {'CI includes zero -> not classically significant' if ci_l<=0<=ci_u else 'CI excludes zero -> significant'}")

# ============================================================
#  RETRAINING VARIANCE (5 seeds)
# ============================================================
print("\n" + "="*65); print("RETRAINING VARIANCE"); print("="*65)

gap2_results = []
for seed in SEEDS_FOR_VARIANCE:
    print(f"  Seed {seed}...")
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    lr_s = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed).fit(X_train_final, y_train)
    rf_s = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1).fit(X_train_final, y_train)
    gbdt_s = GradientBoostingClassifier(random_state=seed).fit(X_train_final, y_train, sample_weight=sw)
    xgb_s = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=seed,
                               eval_metric="logloss", n_jobs=-1).fit(X_train_final, y_train)
    cb_s = CatBoostClassifier(scale_pos_weight=scale_pos_weight, random_state=seed,
                               cat_features=cat_feature_idx, verbose=False).fit(X_train_cb, y_train)
    seed_models = {"Logistic_Regression":("ohe",lr_s), "Random_Forest":("ohe",rf_s),
                   "GBDT":("ohe",gbdt_s), "XGBoost":("ohe",xgb_s), "CatBoost":("native",cb_s)}
    for name, (enc_type, model) in seed_models.items():
        X_eval = X_test_final if enc_type=="ohe" else X_test_cb
        proba = model.predict_proba(X_eval)[:,1]
        gap2_results.append({"Model": name, "Seed": seed,
                              "AUROC": roc_auc_score(y_test, proba),
                              "AUPRC": average_precision_score(y_test, proba)})

gap2_raw_df = pd.DataFrame(gap2_results)
save_csv(gap2_raw_df, "Retraining_variance_raw")
gap2_summary = gap2_raw_df.groupby("Model").agg(
    AUROC_mean=("AUROC","mean"), AUROC_std=("AUROC","std"),
    AUPRC_mean=("AUPRC","mean"), AUPRC_std=("AUPRC","std")).round(4).reset_index()
save_csv(gap2_summary, "Rretraining_variance_summary")
print(gap2_summary.to_string(index=False))

# ============================================================
#  RECALIBRATION (held-out calibration set, CatBoost & XGBoost)
# ============================================================
print("\n" + "="*65); print("RECALIBRATION"); print("="*65)

recal_train_df = model_df[model_df[FEATURE_YEAR] < CALIB_START_YEAR].reset_index(drop=True)
recal_calib_df = model_df[(model_df[FEATURE_YEAR] >= CALIB_START_YEAR) & (model_df[FEATURE_YEAR] <= TRAIN_END_YEAR)].reset_index(drop=True)
y_recal_train = recal_train_df[TARGET].values
y_calib = recal_calib_df[TARGET].values

recal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
recal_encoder.fit(recal_train_df[FEATURES_CATEGORICAL])
X_recal_train_ohe = recal_encoder.transform(recal_train_df[FEATURES_CATEGORICAL])
X_calib_ohe = recal_encoder.transform(recal_calib_df[FEATURES_CATEGORICAL])
X_test_ohe_recal = recal_encoder.transform(test_df[FEATURES_CATEGORICAL])
X_recal_train_final = np.hstack([X_recal_train_ohe, recal_train_df[[FEATURE_YEAR]].values])
X_calib_final = np.hstack([X_calib_ohe, recal_calib_df[[FEATURE_YEAR]].values])
X_test_final_recal = np.hstack([X_test_ohe_recal, test_df[[FEATURE_YEAR]].values])

recal_spw = (len(y_recal_train)-y_recal_train.sum()) / y_recal_train.sum()

print("  Retraining XGBoost/CatBoost on <2018 data...")
xgb_recal_base = xgb.XGBClassifier(scale_pos_weight=recal_spw, random_state=RANDOM_SEED,
                                    eval_metric="logloss", n_jobs=-1).fit(X_recal_train_final, y_recal_train)
X_recal_train_cb = recal_train_df[FEATURES_CATEGORICAL+[FEATURE_YEAR]].copy()
X_calib_cb = recal_calib_df[FEATURES_CATEGORICAL+[FEATURE_YEAR]].copy()
X_test_cb_recal = test_df[FEATURES_CATEGORICAL+[FEATURE_YEAR]].copy()
cb_recal_base = CatBoostClassifier(scale_pos_weight=recal_spw, random_state=RANDOM_SEED,
                                    cat_features=cat_feature_idx, verbose=False).fit(X_recal_train_cb, y_recal_train)

print("  Fitting isotonic calibration on held-out 2018-2019 set...")
try:
    from sklearn.frozen import FrozenEstimator
    xgb_calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_recal_base), method="isotonic")
    xgb_calibrated.fit(X_calib_final, y_calib)
    cb_calibrated = CalibratedClassifierCV(FrozenEstimator(cb_recal_base), method="isotonic")
    cb_calibrated.fit(X_calib_cb, y_calib)
except ImportError:
    xgb_calibrated = CalibratedClassifierCV(xgb_recal_base, method="isotonic", cv="prefit")
    xgb_calibrated.fit(X_calib_final, y_calib)
    cb_calibrated = CalibratedClassifierCV(cb_recal_base, method="isotonic", cv="prefit")
    cb_calibrated.fit(X_calib_cb, y_calib)

gap5_results = []
for name, base_m, calib_m, X_eb, X_ec in [
    ("XGBoost", xgb_recal_base, xgb_calibrated, X_test_final_recal, X_test_final_recal),
    ("CatBoost", cb_recal_base, cb_calibrated, X_test_cb_recal, X_test_cb_recal)]:
    pb = base_m.predict_proba(X_eb)[:,1]; pa = calib_m.predict_proba(X_ec)[:,1]
    gap5_results.append({"Model": name,
                          "Brier_before": round(brier_score_loss(y_test,pb),4),
                          "Brier_after": round(brier_score_loss(y_test,pa),4),
                          "AUROC_before": round(roc_auc_score(y_test,pb),4),
                          "AUROC_after": round(roc_auc_score(y_test,pa),4)})
    print(f"  {name}: Brier {brier_score_loss(y_test,pb):.4f} -> {brier_score_loss(y_test,pa):.4f}")

gap5_df = pd.DataFrame(gap5_results)
save_csv(gap5_df, "Recalibration_results")
joblib.dump(xgb_calibrated, os.path.join(OUT_DIR, "model_XGBoost_calibrated.joblib"))
joblib.dump(cb_calibrated, os.path.join(OUT_DIR, "model_CatBoost_calibrated.joblib"))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*65); print("PHASE 4 (FINAL, CONSOLIDATED) COMPLETE"); print("="*65)
print(f"""
  Best model by AUROC: {results_df.iloc[0]['Model']} (AUROC={results_df.iloc[0]['AUROC']})
  All outputs saved to: {OUT_DIR}/
""")
