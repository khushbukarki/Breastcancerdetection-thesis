# Generated from: hybridmodelres.ipynb
# Converted at: 2026-01-29T11:26:02.606Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================================================
# CBIS-DDSM HYBRID (Diagram): ResNet50 + ViT-B/16 + CLIP(B/32)
# -> CONCAT -> StandardScaler -> PCA -> RandomForest -> Threshold
# RUN CELLS IN ORDER (1 .. 6)
# ============================================================

# -----------------------------
# CELL 1) PATHS + LOAD META CSV
# -----------------------------
import os, glob
import numpy as np
import pandas as pd

CSV_DIR   = r"C:\Users\khush\OneDrive\Desktop\thesisnew\dataset\cbis\csv"
CBIS_ROOT = r"C:\Users\khush\OneDrive\Desktop\thesisnew\dataset\cbis"
META_PATH = os.path.join(CSV_DIR, "meta_with_roi.csv")

print("META_PATH:", META_PATH, "exists?", os.path.exists(META_PATH))
print("CSV files:", glob.glob(os.path.join(CSV_DIR, "*.csv"))[:10])

df = pd.read_csv(META_PATH)
print("Loaded df:", df.shape)
print("Columns:", df.columns.tolist())
df.head(2)


# ---------------------------------------------------
# CELL 2) STANDARDIZE COLUMNS: patient_id, label, img_full
# ---------------------------------------------------
# auto-detect columns
patient_col = next((c for c in ["patient_id","Patient ID","PatientID","patient","subject_id"] if c in df.columns), None)
label_col   = next((c for c in ["label","pathology","Pathology","target"] if c in df.columns), None)
path_col    = next((c for c in ["img_full","image_path","png_path","full_path","img_path","image_file_path"] if c in df.columns), None)

assert patient_col is not None, "No patient_id column found. Paste df.columns."
assert label_col   is not None, "No label/pathology column found. Paste df.columns."
assert path_col    is not None, "No image path column found. Paste df.columns."

df = df.rename(columns={patient_col:"patient_id", label_col:"label", path_col:"img_full"})

# label -> 0/1
if df["label"].dtype == "object":
    s = df["label"].astype(str).str.lower()
    df["label"] = s.map({"benign":0, "malignant":1})
    df.loc[df["label"].isna() & s.str.contains("malig"), "label"] = 1
    df.loc[df["label"].isna() & s.str.contains("benign"), "label"] = 0

df["label"] = df["label"].astype(int)

print("Using columns: patient_id, label, img_full")
print(df[["patient_id","label","img_full"]].head())
print("Label counts:\n", df["label"].value_counts())


# ---------------------------------------
# CELL 3) FIX IMAGE PATHS + CHECK EXISTENCE
# ---------------------------------------
def fix_path(p):
    p = str(p)

    # already absolute and exists
    if os.path.isabs(p) and os.path.exists(p):
        return p

    # try common subfolders
    for sub in ["png", "jpeg", "JPEG", "roi_crops"]:
        cand = os.path.join(CBIS_ROOT, sub, p)
        if os.path.exists(cand):
            return cand

    # try directly under root
    cand = os.path.join(CBIS_ROOT, p)
    if os.path.exists(cand):
        return cand

    return p

df["img_full"] = df["img_full"].apply(fix_path)

exists_rate = df["img_full"].map(os.path.exists).mean()
print("Image exists %:", exists_rate)
print("Example path:", df["img_full"].iloc[0], "exists?", os.path.exists(df["img_full"].iloc[0]))

assert exists_rate > 0.95, "Too many missing image paths. Show df['img_full'].head(10)"


# ---------------------------------------
# CELL 4) PATIENT-WISE SPLIT (NO LEAKAGE)
# ---------------------------------------
from sklearn.model_selection import GroupShuffleSplit

SEED = 42

# train vs temp
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=SEED)
tr_idx, tmp_idx = next(gss1.split(df, df["label"], groups=df["patient_id"]))
df_tr = df.iloc[tr_idx].reset_index(drop=True)
df_tmp = df.iloc[tmp_idx].reset_index(drop=True)

# val vs test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
va_idx, te_idx = next(gss2.split(df_tmp, df_tmp["label"], groups=df_tmp["patient_id"]))
df_va = df_tmp.iloc[va_idx].reset_index(drop=True)
df_te = df_tmp.iloc[te_idx].reset_index(drop=True)

print("Sizes:", len(df_tr), len(df_va), len(df_te))
print("\nLabel counts:")
print("Train:\n", df_tr["label"].value_counts())
print("Val:\n", df_va["label"].value_counts())
print("Test:\n", df_te["label"].value_counts())

print("\nLeakage check (must be 0):")
print("tr∩va:", len(set(df_tr.patient_id) & set(df_va.patient_id)))
print("tr∩te:", len(set(df_tr.patient_id) & set(df_te.patient_id)))
print("va∩te:", len(set(df_va.patient_id) & set(df_te.patient_id)))


# ----------------------------------------------------------
# CELL 5) FEATURE EXTRACTORS (ResNet50 + ViT-B/16 + CLIP-B/32)
# ----------------------------------------------------------
import torch
import timm
import open_clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
BATCH = 8 if device == "cuda" else 1

def load_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ResNet50 (2048-d)
cnn = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()
cnn_cfg = timm.data.resolve_model_data_config(cnn)
cnn_tfm = timm.data.create_transform(**cnn_cfg, is_training=False)

# ViT-B/16 (768-d)
vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device).eval()
vit_cfg = timm.data.resolve_model_data_config(vit)
vit_tfm = timm.data.create_transform(**vit_cfg, is_training=False)

# CLIP ViT-B/32 (512-d)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()

@torch.no_grad()
def extract_timm_feats(df_part, model, tfm, batch=BATCH, desc="timm"):
    paths = df_part["img_full"].tolist()
    feats = []
    for i in tqdm(range(0, len(paths), batch), desc=desc):
        b = paths[i:i+batch]
        x = torch.stack([tfm(load_rgb(p)) for p in b]).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.vstack(feats)

@torch.no_grad()
def extract_clip_feats(df_part, model, preprocess, batch=BATCH, desc="clip"):
    paths = df_part["img_full"].tolist()
    feats = []
    for i in tqdm(range(0, len(paths), batch), desc=desc):
        b = paths[i:i+batch]
        x = torch.stack([preprocess(load_rgb(p)) for p in b]).to(device)
        f = model.encode_image(x)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.detach().cpu().numpy())
    return np.vstack(feats)

# Extract
Xtr_cnn  = extract_timm_feats(df_tr, cnn, cnn_tfm, desc="cnn-tr")
Xva_cnn  = extract_timm_feats(df_va, cnn, cnn_tfm, desc="cnn-va")
Xte_cnn  = extract_timm_feats(df_te, cnn, cnn_tfm, desc="cnn-te")

Xtr_vit  = extract_timm_feats(df_tr, vit, vit_tfm, desc="vit-tr")
Xva_vit  = extract_timm_feats(df_va, vit, vit_tfm, desc="vit-va")
Xte_vit  = extract_timm_feats(df_te, vit, vit_tfm, desc="vit-te")

Xtr_clip = extract_clip_feats(df_tr, clip_model, clip_preprocess, desc="clip-tr")
Xva_clip = extract_clip_feats(df_va, clip_model, clip_preprocess, desc="clip-va")
Xte_clip = extract_clip_feats(df_te, clip_model, clip_preprocess, desc="clip-te")

print("Shapes:",
      "CNN", Xtr_cnn.shape,
      "ViT", Xtr_vit.shape,
      "CLIP", Xtr_clip.shape)


# ----------------------------------------------------------
# CELL 6) CONCAT -> SCALE -> PCA -> RF -> THRESHOLD -> EVAL
# ----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

ytr = df_tr["label"].astype(int).values
yva = df_va["label"].astype(int).values
yte = df_te["label"].astype(int).values

# CONCAT (3328 dims)
Xtr = np.concatenate([Xtr_cnn, Xtr_vit, Xtr_clip], axis=1)
Xva = np.concatenate([Xva_cnn, Xva_vit, Xva_clip], axis=1)
Xte = np.concatenate([Xte_cnn, Xte_vit, Xte_clip], axis=1)
print("Concat shapes:", Xtr.shape, Xva.shape, Xte.shape)

# SCALE -> PCA (fit on train only)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xva_s = scaler.transform(Xva)
Xte_s = scaler.transform(Xte)

PCA_DIM = 256
pca = PCA(n_components=PCA_DIM, random_state=SEED)
Xtr_p = pca.fit_transform(Xtr_s)
Xva_p = pca.transform(Xva_s)
Xte_p = pca.transform(Xte_s)
print("PCA shapes:", Xtr_p.shape, Xva_p.shape, Xte_p.shape)

# RF
rf = RandomForestClassifier(
    n_estimators=2500,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED
)
rf.fit(Xtr_p, ytr)

def pick_threshold_for_target_sens(y_true, prob_pos, target_sens=0.80):
    ths = np.linspace(0.01, 0.99, 991)
    best = None
    for t in ths:
        pred = (prob_pos >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn+1e-12)
        spec = tn/(tn+fp+1e-12)
        if sens >= target_sens:
            if best is None or spec > best["spec"]:
                best = {"t": float(t), "sens": float(sens), "spec": float(spec), "fp": int(fp), "fn": int(fn)}
    return best

pva = rf.predict_proba(Xva_p)[:, 1]
best = pick_threshold_for_target_sens(yva, pva, target_sens=0.80)
thr = 0.5 if best is None else best["t"]
print("Chosen threshold:", thr, "details:", best)

def eval_split(name, X, y, thr):
    p = rf.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    auc  = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    sens = recall_score(y, pred, zero_division=0)
    spec = tn/(tn+fp+1e-12)
    f1   = f1_score(y, pred, zero_division=0)

    print("\n" + "="*62)
    print(f"{name} @ thr={thr:.3f}")
    print("="*62)
    print(f"AUC        : {auc:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1         : {f1:.4f}")
    print("Confusion Matrix [[TN FP] [FN TP]]")
    print(np.array([[tn, fp],[fn, tp]]))
    print("\nClassification Report")
    print(classification_report(y, pred, digits=4, zero_division=0))

eval_split("VAL",  Xva_p, yva, thr)
eval_split("TEST", Xte_p, yte, thr)


# ----------------------------------------------------------
# CELL 7) SIDE-BY-SIDE VISUALIZATION (VAL | TEST)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals(Xva, yva, Xte, yte, thr):

    pva = rf.predict_proba(Xva)[:,1]
    pte = rf.predict_proba(Xte)[:,1]

    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)

    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    ths = np.linspace(0.01, 0.99, 100)

    def sweep(y, prob):
        sens, spec = [], []
        for t in ths:
            pred = (prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens.append(tp/(tp+fn+1e-12))
            spec.append(tn/(tn+fp+1e-12))
        return sens, spec

    sens_va, spec_va = sweep(yva, pva)
    sens_te, spec_te = sweep(yte, pte)

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    # =========================
    # Confusion matrices
    # =========================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL Confusion Matrix"),
        (ax[0,1], cm_te, "TEST Confusion Matrix")
    ]:
        im = a.imshow(cm)
        a.set_title(title)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["Benign","Malignant"])
        a.set_yticklabels(["Benign","Malignant"])
        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center")

    # =========================
    # ROC curves
    # =========================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0],"VAL ROC"), (ax[1,1],"TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR")
        a.set_ylabel("TPR")
        a.set_title(title)
        a.legend()

    # =========================
    # Threshold sweep
    # =========================
    ax[2,0].plot(ths, sens_va, label="Sensitivity")
    ax[2,0].plot(ths, spec_va, label="Specificity")
    ax[2,0].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,0].set_title("VAL Threshold Sweep")
    ax[2,0].legend()

    ax[2,1].plot(ths, sens_te, label="Sensitivity")
    ax[2,1].plot(ths, spec_te, label="Specificity")
    ax[2,1].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,1].set_title("TEST Threshold Sweep")
    ax[2,1].legend()

    plt.tight_layout()
    plt.show()


# Run
side_by_side_visuals(Xva_p, yva, Xte_p, yte, thr)


# ----------------------------------------------------------
# COMPACT SIDE-BY-SIDE VISUALS (small + thesis friendly)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals_small(Xva, yva, Xte, yte, thr):

    pva = rf.predict_proba(Xva)[:,1]
    pte = rf.predict_proba(Xte)[:,1]

    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)

    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    # SMALL FIGURE HERE 👇
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))

    # ======================
    # Confusion matrices
    # ======================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL CM"),
        (ax[0,1], cm_te, "TEST CM")
    ]:
        a.imshow(cm)
        a.set_title(title, fontsize=10)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["B","M"], fontsize=8)
        a.set_yticklabels(["B","M"], fontsize=8)

        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)

    # ======================
    # ROC curves
    # ======================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0],"VAL ROC"), (ax[1,1],"TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR", fontsize=9)
        a.set_ylabel("TPR", fontsize=9)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()


# Run
side_by_side_visuals_small(Xva_p, yva, Xte_p, yte, thr)


# ----------------------------------------------------------
# COMPACT SIDE-BY-SIDE VISUALS + THRESHOLD SWEEPS (thesis friendly)
# 3 rows x 2 cols:
# Row1: Confusion Matrices (VAL/TEST)
# Row2: ROC curves (VAL/TEST)
# Row3: Threshold sweep (Sensitivity/Specificity vs threshold) (VAL/TEST)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals_with_sweeps(Xva, yva, Xte, yte, thr, n_thr=101):
    # Probabilities
    pva = rf.predict_proba(Xva)[:, 1]
    pte = rf.predict_proba(Xte)[:, 1]

    # Predictions at chosen threshold
    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    # Confusion matrices
    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    # ROC
    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)
    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    # Threshold sweeps
    ths = np.linspace(0.0, 1.0, n_thr)

    def sweep(y, p):
        sens_list, spec_list = [], []
        for t in ths:
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            sens_list.append(sens)
            spec_list.append(spec)
        return np.array(sens_list), np.array(spec_list)

    sens_va, spec_va = sweep(yva, pva)
    sens_te, spec_te = sweep(yte, pte)

    # Plot: compact 3x2
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 8))

    # ======================
    # Row 1: Confusion matrices
    # ======================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL Confusion Matrix"),
        (ax[0,1], cm_te, "TEST Confusion Matrix")
    ]:
        a.imshow(cm)
        a.set_title(title, fontsize=10)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["Benign","Malignant"], fontsize=8)
        a.set_yticklabels(["Benign","Malignant"], fontsize=8)

        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)

    # ======================
    # Row 2: ROC
    # ======================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0], "VAL ROC"), (ax[1,1], "TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR", fontsize=9)
        a.set_ylabel("TPR", fontsize=9)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)

    # ======================
    # Row 3: Threshold sweeps
    # ======================
    ax[2,0].plot(ths, sens_va, label="Sensitivity")
    ax[2,0].plot(ths, spec_va, label="Specificity")
    ax[2,0].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,0].set_title("VAL Threshold Sweep", fontsize=10)
    ax[2,0].set_xlabel("Threshold", fontsize=9)
    ax[2,0].set_ylabel("Score", fontsize=9)
    ax[2,0].legend(fontsize=8)

    ax[2,1].plot(ths, sens_te, label="Sensitivity")
    ax[2,1].plot(ths, spec_te, label="Specificity")
    ax[2,1].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,1].set_title("TEST Threshold Sweep", fontsize=10)
    ax[2,1].set_xlabel("Threshold", fontsize=9)
    ax[2,1].set_ylabel("Score", fontsize=9)
    ax[2,1].legend(fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()


# Run (use your PCA features)
side_by_side_visuals_with_sweeps(Xva_p, yva, Xte_p, yte, thr, n_thr=201)


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Use your CONCAT features BEFORE PCA: Xtr, Xva, Xte must exist
assert "Xtr" in globals() and "Xva" in globals() and "Xte" in globals(), "Need Xtr/Xva/Xte (concat features) first."
assert "ytr" in globals() and "yva" in globals() and "yte" in globals(), "Need ytr/yva/yte first."

def train_eval_pca_rf(pca_dim, n_estimators):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    pca = PCA(n_components=pca_dim, random_state=SEED)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xva_p = pca.transform(Xva_s)
    Xte_p = pca.transform(Xte_s)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(Xtr_p, ytr)

    pva = rf.predict_proba(Xva_p)[:, 1]
    pte = rf.predict_proba(Xte_p)[:, 1]

    auc_va = roc_auc_score(yva, pva)
    auc_te = roc_auc_score(yte, pte)
    return auc_va, auc_te, scaler, pca, rf, Xtr_p, Xva_p, Xte_p

dims = [128, 256, 512]
trees = [2500, 4000]

best = None
results = []
for d in dims:
    for n in trees:
        auc_va, auc_te, scaler, pca, rf, Xtr_p2, Xva_p2, Xte_p2 = train_eval_pca_rf(d, n)
        results.append((d, n, auc_va, auc_te))
        print(f"PCA={d:3d} | trees={n:4d} | VAL AUC={auc_va:.4f} | TEST AUC={auc_te:.4f}")

        if best is None or auc_va > best["auc_va"]:
            best = {
                "pca_dim": d, "trees": n,
                "auc_va": auc_va, "auc_te": auc_te,
                "scaler": scaler, "pca": pca, "rf": rf,
                "Xtr_p": Xtr_p2, "Xva_p": Xva_p2, "Xte_p": Xte_p2
            }

print("\n✅ BEST by VAL AUC:", {k: best[k] for k in ["pca_dim","trees","auc_va","auc_te"]})

# overwrite current model/features with best so next cells use them
scaler = best["scaler"]
pca    = best["pca"]
rf     = best["rf"]
Xtr_p  = best["Xtr_p"]
Xva_p  = best["Xva_p"]
Xte_p  = best["Xte_p"]


from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def pick_threshold_target_sens(y_true, prob_pos, target_sens=0.80):
    ths = np.linspace(0.01, 0.99, 991)
    best = None
    for t in ths:
        pred = (prob_pos >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn+1e-12)
        spec = tn/(tn+fp+1e-12)
        if sens >= target_sens:
            # maximize specificity among feasible thresholds
            if best is None or spec > best["spec"]:
                best = {"t": float(t), "sens": float(sens), "spec": float(spec), "fp": int(fp), "fn": int(fn)}
    return best

def eval_at_thr(X, y, thr):
    p = rf.predict_proba(X)[:,1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "AUC": roc_auc_score(y, p),
        "ACC": accuracy_score(y, pred),
        "PREC": precision_score(y, pred, zero_division=0),
        "SENS": recall_score(y, pred, zero_division=0),
        "SPEC": tn/(tn+fp+1e-12),
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

pva = rf.predict_proba(Xva_p)[:,1]

for target in [0.75, 0.80, 0.90]:
    best = pick_threshold_target_sens(yva, pva, target_sens=target)
    if best is None:
        print(f"\nTarget sens={target}: ❌ no threshold met this target")
        continue
    thr = best["t"]
    val = eval_at_thr(Xva_p, yva, thr)
    tes = eval_at_thr(Xte_p, yte, thr)

    print("\n" + "="*70)
    print(f"Target sensitivity={target} | chosen thr={thr:.3f} (VAL sens={best['sens']:.3f}, spec={best['spec']:.3f})")
    print("- VAL :", val)
    print("- TEST:", tes)


# ==================================================
# FINAL MODEL (from your sweep)
# PCA = 128
# RF trees = 2500
# ==================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# scale
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xva_s = scaler.transform(Xva)
Xte_s = scaler.transform(Xte)

# PCA 128
pca = PCA(n_components=128, random_state=SEED)
Xtr_p = pca.fit_transform(Xtr_s)
Xva_p = pca.transform(Xva_s)
Xte_p = pca.transform(Xte_s)

# RF 2500
rf = RandomForestClassifier(
    n_estimators=2500,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED
)

rf.fit(Xtr_p, ytr)
print("✅ Final model trained")


import numpy as np
from sklearn.metrics import confusion_matrix

def pick_threshold(y_true, probs, target_sens=0.80):
    ths = np.linspace(0.01, 0.99, 991)
    best = None

    for t in ths:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

        sens = tp/(tp+fn)
        spec = tn/(tn+fp)

        if sens >= target_sens:
            if best is None or spec > best["spec"]:
                best = {"t": t, "sens": sens, "spec": spec}

    return best

pva = rf.predict_proba(Xva_p)[:,1]
best = pick_threshold(yva, pva, 0.80)

thr = best["t"]
print("✅ Final threshold:", best)


from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

def evaluate(name, X, y, thr):
    p = rf.predict_proba(X)[:,1]
    pred = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print("\n" + "="*60)
    print(name)
    print("="*60)

    print("AUC       :", roc_auc_score(y, p))
    print("Accuracy  :", accuracy_score(y, pred))
    print("Sensitivity:", recall_score(y, pred))
    print("Specificity:", tn/(tn+fp))
    print("Precision :", precision_score(y, pred))
    print("F1        :", f1_score(y, pred))
    print("CM [[TN FP][FN TP]]")
    print([[tn, fp],[fn, tp]])

    print("\nReport:\n", classification_report(y, pred))


evaluate("VAL",  Xva_p, yva, thr)
evaluate("TEST", Xte_p, yte, thr)



# COMPACT SIDE-BY-SIDE VISUALS + THRESHOLD SWEEPS (thesis friendly)
# 3 rows x 2 cols:
# Row1: Confusion Matrices (VAL/TEST)
# Row2: ROC curves (VAL/TEST)
# Row3: Threshold sweep (Sensitivity/Specificity vs threshold) (VAL/TEST)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals_with_sweeps(Xva, yva, Xte, yte, thr, n_thr=101):
    # Probabilities
    pva = rf.predict_proba(Xva)[:, 1]
    pte = rf.predict_proba(Xte)[:, 1]

    # Predictions at chosen threshold
    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    # Confusion matrices
    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    # ROC
    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)
    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    # Threshold sweeps
    ths = np.linspace(0.0, 1.0, n_thr)

    def sweep(y, p):
        sens_list, spec_list = [], []
        for t in ths:
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            sens_list.append(sens)
            spec_list.append(spec)
        return np.array(sens_list), np.array(spec_list)

    sens_va, spec_va = sweep(yva, pva)
    sens_te, spec_te = sweep(yte, pte)

    # Plot: compact 3x2
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 8))

    # ======================
    # Row 1: Confusion matrices
    # ======================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL Confusion Matrix"),
        (ax[0,1], cm_te, "TEST Confusion Matrix")
    ]:
        a.imshow(cm)
        a.set_title(title, fontsize=10)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["Benign","Malignant"], fontsize=8)
        a.set_yticklabels(["Benign","Malignant"], fontsize=8)

        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)

    # ======================
    # Row 2: ROC
    # ======================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0], "VAL ROC"), (ax[1,1], "TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR", fontsize=9)
        a.set_ylabel("TPR", fontsize=9)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)

    # ======================
    # Row 3: Threshold sweeps
    # ======================
    ax[2,0].plot(ths, sens_va, label="Sensitivity")
    ax[2,0].plot(ths, spec_va, label="Specificity")
    ax[2,0].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,0].set_title("VAL Threshold Sweep", fontsize=10)
    ax[2,0].set_xlabel("Threshold", fontsize=9)
    ax[2,0].set_ylabel("Score", fontsize=9)
    ax[2,0].legend(fontsize=8)

    ax[2,1].plot(ths, sens_te, label="Sensitivity")
    ax[2,1].plot(ths, spec_te, label="Specificity")
    ax[2,1].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,1].set_title("TEST Threshold Sweep", fontsize=10)
    ax[2,1].set_xlabel("Threshold", fontsize=9)
    ax[2,1].set_ylabel("Score", fontsize=9)
    ax[2,1].legend(fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()


# Run (use your PCA features)
side_by_side_visuals_with_sweeps(Xva_p, yva, Xte_p, yte, thr, n_thr=201)

thr_final = 0.346


eval_split("VAL",  Xva_p, yva, thr_final)
eval_split("TEST", Xte_p, yte, thr_final)



# COMPACT SIDE-BY-SIDE VISUALS + THRESHOLD SWEEPS (thesis friendly)
# 3 rows x 2 cols:
# Row1: Confusion Matrices (VAL/TEST)
# Row2: ROC curves (VAL/TEST)
# Row3: Threshold sweep (Sensitivity/Specificity vs threshold) (VAL/TEST)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals_with_sweeps(Xva, yva, Xte, yte, thr, n_thr=101):
    # Probabilities
    pva = rf.predict_proba(Xva)[:, 1]
    pte = rf.predict_proba(Xte)[:, 1]

    # Predictions at chosen threshold
    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    # Confusion matrices
    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    # ROC
    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)
    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    # Threshold sweeps
    ths = np.linspace(0.0, 1.0, n_thr)

    def sweep(y, p):
        sens_list, spec_list = [], []
        for t in ths:
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            sens_list.append(sens)
            spec_list.append(spec)
        return np.array(sens_list), np.array(spec_list)

    sens_va, spec_va = sweep(yva, pva)
    sens_te, spec_te = sweep(yte, pte)

    # Plot: compact 3x2
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 8))

    # ======================
    # Row 1: Confusion matrices
    # ======================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL Confusion Matrix"),
        (ax[0,1], cm_te, "TEST Confusion Matrix")
    ]:
        a.imshow(cm)
        a.set_title(title, fontsize=10)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["Benign","Malignant"], fontsize=8)
        a.set_yticklabels(["Benign","Malignant"], fontsize=8)

        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)

    # ======================
    # Row 2: ROC
    # ======================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0], "VAL ROC"), (ax[1,1], "TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR", fontsize=9)
        a.set_ylabel("TPR", fontsize=9)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)

    # ======================
    # Row 3: Threshold sweeps
    # ======================
    ax[2,0].plot(ths, sens_va, label="Sensitivity")
    ax[2,0].plot(ths, spec_va, label="Specificity")
    ax[2,0].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,0].set_title("VAL Threshold Sweep", fontsize=10)
    ax[2,0].set_xlabel("Threshold", fontsize=9)
    ax[2,0].set_ylabel("Score", fontsize=9)
    ax[2,0].legend(fontsize=8)

    ax[2,1].plot(ths, sens_te, label="Sensitivity")
    ax[2,1].plot(ths, spec_te, label="Specificity")
    ax[2,1].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,1].set_title("TEST Threshold Sweep", fontsize=10)
    ax[2,1].set_xlabel("Threshold", fontsize=9)
    ax[2,1].set_ylabel("Score", fontsize=9)
    ax[2,1].legend(fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()


# Run (use your PCA features)
side_by_side_visuals_with_sweeps(Xva_p, yva, Xte_p, yte, thr, n_thr=201)





import importlib.util, sys
print("imblearn installed?", importlib.util.find_spec("imblearn") is not None)


from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

brf = BalancedRandomForestClassifier(
    n_estimators=2000,
    random_state=SEED,
    n_jobs=-1
)
brf.fit(Xtr_p, ytr)

pva = brf.predict_proba(Xva_p)[:,1]
pte = brf.predict_proba(Xte_p)[:,1]

print("BRF VAL AUC:", roc_auc_score(yva, pva))
print("BRF TEST AUC:", roc_auc_score(yte, pte))

# set rf = brf so your plotting/eval functions still work
rf = brf


import numpy as np
from sklearn.metrics import confusion_matrix

def pick_threshold_target_sens(y_true, prob_pos, target_sens=0.80):
    ths = np.linspace(0.01, 0.99, 991)
    best = None

    for t in ths:
        pred = (prob_pos >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()

        sens = tp/(tp+fn+1e-12)
        spec = tn/(tn+fp+1e-12)

        if sens >= target_sens:
            if best is None or spec > best["spec"]:
                best = {"t": float(t), "sens": float(sens), "spec": float(spec),
                        "fp": int(fp), "fn": int(fn)}

    return best


# VAL probabilities
pva = rf.predict_proba(Xva_p)[:,1]

best = pick_threshold_target_sens(yva, pva, target_sens=0.80)
thr = best["t"]

print("Chosen threshold:", best)


from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

def evaluate_split(name, X, y, thr):
    p = rf.predict_proba(X)[:,1]
    pred = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()

    auc  = roc_auc_score(y, p)
    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    sens = recall_score(y, pred)
    spec = tn/(tn+fp)
    f1   = f1_score(y, pred)

    print("\n" + "="*60)
    print(name)
    print("="*60)
    print(f"AUC        : {auc:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"F1         : {f1:.4f}")
    print("Confusion Matrix [[TN FP] [FN TP]]")
    print([[tn, fp],[fn, tp]])

    print("\nClassification Report")
    print(classification_report(y, pred))


evaluate_split("VAL",  Xva_p, yva, thr)
evaluate_split("TEST", Xte_p, yte, thr)



# COMPACT SIDE-BY-SIDE VISUALS + THRESHOLD SWEEPS (thesis friendly)
# 3 rows x 2 cols:
# Row1: Confusion Matrices (VAL/TEST)
# Row2: ROC curves (VAL/TEST)
# Row3: Threshold sweep (Sensitivity/Specificity vs threshold) (VAL/TEST)
# ----------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def side_by_side_visuals_with_sweeps(Xva, yva, Xte, yte, thr, n_thr=101):
    # Probabilities
    pva = rf.predict_proba(Xva)[:, 1]
    pte = rf.predict_proba(Xte)[:, 1]

    # Predictions at chosen threshold
    pred_va = (pva >= thr).astype(int)
    pred_te = (pte >= thr).astype(int)

    # Confusion matrices
    cm_va = confusion_matrix(yva, pred_va, labels=[0,1])
    cm_te = confusion_matrix(yte, pred_te, labels=[0,1])

    # ROC
    fpr_va, tpr_va, _ = roc_curve(yva, pva)
    fpr_te, tpr_te, _ = roc_curve(yte, pte)
    auc_va = auc(fpr_va, tpr_va)
    auc_te = auc(fpr_te, tpr_te)

    # Threshold sweeps
    ths = np.linspace(0.0, 1.0, n_thr)

    def sweep(y, p):
        sens_list, spec_list = [], []
        for t in ths:
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            sens_list.append(sens)
            spec_list.append(spec)
        return np.array(sens_list), np.array(spec_list)

    sens_va, spec_va = sweep(yva, pva)
    sens_te, spec_te = sweep(yte, pte)

    # Plot: compact 3x2
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 8))

    # ======================
    # Row 1: Confusion matrices
    # ======================
    for a, cm, title in [
        (ax[0,0], cm_va, "VAL Confusion Matrix"),
        (ax[0,1], cm_te, "TEST Confusion Matrix")
    ]:
        a.imshow(cm)
        a.set_title(title, fontsize=10)
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(["Benign","Malignant"], fontsize=8)
        a.set_yticklabels(["Benign","Malignant"], fontsize=8)

        for (i,j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)

    # ======================
    # Row 2: ROC
    # ======================
    ax[1,0].plot(fpr_va, tpr_va, label=f"AUC={auc_va:.3f}")
    ax[1,1].plot(fpr_te, tpr_te, label=f"AUC={auc_te:.3f}")

    for a, title in [(ax[1,0], "VAL ROC"), (ax[1,1], "TEST ROC")]:
        a.plot([0,1],[0,1],'--')
        a.set_xlabel("FPR", fontsize=9)
        a.set_ylabel("TPR", fontsize=9)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)

    # ======================
    # Row 3: Threshold sweeps
    # ======================
    ax[2,0].plot(ths, sens_va, label="Sensitivity")
    ax[2,0].plot(ths, spec_va, label="Specificity")
    ax[2,0].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,0].set_title("VAL Threshold Sweep", fontsize=10)
    ax[2,0].set_xlabel("Threshold", fontsize=9)
    ax[2,0].set_ylabel("Score", fontsize=9)
    ax[2,0].legend(fontsize=8)

    ax[2,1].plot(ths, sens_te, label="Sensitivity")
    ax[2,1].plot(ths, spec_te, label="Specificity")
    ax[2,1].axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
    ax[2,1].set_title("TEST Threshold Sweep", fontsize=10)
    ax[2,1].set_xlabel("Threshold", fontsize=9)
    ax[2,1].set_ylabel("Score", fontsize=9)
    ax[2,1].legend(fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()


# Run (use your PCA features)
side_by_side_visuals_with_sweeps(Xva_p, yva, Xte_p, yte, thr, n_thr=201)