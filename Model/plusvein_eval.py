"""
============================================================================
PLUSVein RL850 — Cross-Dataset Verification Evaluation  +  Grad-CAM
============================================================================
Purpose
-------
Produce a DEFENSIBLE biometric result on data the models have never seen.

The 11 models were trained on FYODB (150 classes). PLUSVein subjects are NOT
in those classes, so "identification accuracy" is undefined. The correct
protocol is VERIFICATION via embeddings:

    1. Strip each model's classifier head -> use as a feature extractor.
    2. Embed every PLUSVein image.
    3. Genuine pairs  = same hand (same subject+L/R), different photo.
       Impostor pairs = different hands.
    4. Cosine similarity -> EER, FAR/FRR, ROC/DET, AUC.

This is the methodologically correct way to report cross-dataset
generalisation, and it runs INFERENCE-ONLY on existing checkpoints
(no retraining). Also generates Grad-CAM overlays for qualitative analysis.

How to run
----------
Place this file in your project root (same folder that contains
'Saved Models/' and 'Outputs/'), then:

    python plusvein_eval.py

Outputs land in:  Outputs/PalmVein_V2/cross_dataset_eer/
Send that whole folder back and the results chapters get written from it.

Requirements (you already have these from the notebook):
    torch torchvision timm opencv-python pillow numpy pandas scikit-learn
    matplotlib tqdm
    grad-cam   ->  pip install grad-cam     (only needed for the Grad-CAM step)
============================================================================
"""

import os, re, json, gc, itertools, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. CONFIG  — edit ONLY if your paths differ from the notebook
# ----------------------------------------------------------------------------
PLUSVEIN_PREPROCESSED = r'F:\Palm Vein Multimodel V2\External\HandVein PLUSVein\HandVein_RL850_Preprocessed'
MODELS_DIR = r'F:\Palm Vein Multimodel V2\Model\Saved Models\PalmVein_V2'
OUT_DIR    = r'F:\Palm Vein Multimodel V2\Model\Outputs\PalmVein_V2\cross_dataset_eer'

NUM_CLASSES = 150          # FYODB classes the heads were trained for (for head rebuild only)
SEED        = 42
MAX_IMPOSTOR_PAIRS = 200_000   # cap impostor pairs so RAM/time stay sane
GRADCAM_MODELS = ['EfficientNetV2_S', 'ResNet50', 'ConvNeXt_Tiny', 'PVCNN']  # top set + custom
GRADCAM_SAMPLES = 8        # how many palms to visualise per model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ----------------------------------------------------------------------------
# 1. PVCNN definition  (must match training exactly — copied from notebook)
# ----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.GELU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 2
        self.branch3 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(mid), nn.GELU())
        self.branch5 = nn.Sequential(nn.Conv2d(in_ch, mid, 5, padding=2, bias=False),
                                        nn.BatchNorm2d(mid), nn.GELU())
        self.se = SEBlock(out_ch)
        self.proj = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x):
        x = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        x = self.se(x)
        return self.proj(x)

class PVCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(32), nn.GELU())
        self.stage1 = nn.Sequential(MultiScaleBlock(32, 64),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(MultiScaleBlock(64, 128), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(MultiScaleBlock(128, 256), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(MultiScaleBlock(256, 512), nn.MaxPool2d(2))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.Flatten(),
                                    nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.4),
                                    nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x); x = self.pool(x)
        return self.head(x)

# ----------------------------------------------------------------------------
# 2. Model builders  (copied from notebook cell 36)
# ----------------------------------------------------------------------------
import timm
def build_mobilenet():
    m = models.mobilenet_v3_large(); m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES); return m
def build_vgg16():
    m = models.vgg16(); m.classifier[6] = nn.Linear(4096, NUM_CLASSES); return m
def build_vgg19():
    m = models.vgg19(); m.classifier[6] = nn.Linear(4096, NUM_CLASSES); return m
def build_convnext():
    m = models.convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES); return m
def build_inception(aux_logits=False):
    m = models.inception_v3(aux_logits=aux_logits, transform_input=False)
    if not aux_logits and getattr(m, 'AuxLogits', None) is not None:
        m.aux_logits = False; m.AuxLogits = None
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m
def build_resnet50():
    m = models.resnet50(); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m
def build_resnet101():
    m = models.resnet101(); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m
def build_densenet():
    m = models.densenet121(); m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES); return m
def build_regnet():
    m = models.regnet_y_8gf(); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m

MODEL_REGISTRY = {
    'EfficientNetV2_S':  lambda: timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=NUM_CLASSES),
    'ResNet50':  build_resnet50, 'ResNet101': build_resnet101, 'DenseNet121': build_densenet,
    'MobileNetV3_Large': build_mobilenet, 'VGG16': build_vgg16, 'VGG19': build_vgg19,
    'InceptionV3': lambda: build_inception(False), 'RegNetY_8GF': build_regnet,
    'ConvNeXt_Tiny': build_convnext, 'PVCNN': lambda: PVCNN(NUM_CLASSES),
}

def extract_state_dict(raw):
    if isinstance(raw, dict):
        sd = raw.get('state_dict', raw.get('model_state_dict', raw))
    else:
        sd = raw
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

def load_model(name):
    ckpt = os.path.join(MODELS_DIR, f'{name}_best.pth')
    if not os.path.exists(ckpt):
        print(f'  MISSING checkpoint: {name}'); return None
    raw = torch.load(ckpt, map_location=DEVICE)
    sd  = extract_state_dict(raw)
    m   = MODEL_REGISTRY[name]()
    try:
        m.load_state_dict(sd, strict=True)
    except Exception:
        m.load_state_dict(sd, strict=False)
        print(f'  (non-strict load) {name}')
    return m.to(DEVICE).eval()

# ----------------------------------------------------------------------------
# 3. Feature extraction — strip the head, return penultimate embedding
# ----------------------------------------------------------------------------
def to_feature_extractor(name, model):
    """Return a callable that maps an input batch -> (B, D) embedding."""
    if name.startswith('ResNet') or name == 'RegNetY_8GF':
        modules = list(model.children())[:-1]               # drop fc
        net = nn.Sequential(*modules)
        return lambda x: torch.flatten(net(x), 1)
    if name == 'DenseNet121':
        feats = model.features
        return lambda x: torch.flatten(
            nn.functional.adaptive_avg_pool2d(nn.functional.relu(feats(x)), 1), 1)
    if name in ('VGG16', 'VGG19'):
        feats, avg = model.features, model.avgpool
        clf = nn.Sequential(*list(model.classifier.children())[:-1])  # keep through 4096
        return lambda x: clf(torch.flatten(avg(feats(x)), 1))
    if name == 'ConvNeXt_Tiny':
        feats, avg = model.features, model.avgpool
        return lambda x: torch.flatten(avg(feats(x)), 1)
    if name == 'MobileNetV3_Large':
        feats, avg = model.features, model.avgpool
        return lambda x: torch.flatten(avg(feats(x)), 1)
    if name == 'InceptionV3':
        # forward through everything except final fc
        def fwd(x):
            m = model
            x = m.Conv2d_1a_3x3(x); x = m.Conv2d_2a_3x3(x); x = m.Conv2d_2b_3x3(x)
            x = m.maxpool1(x); x = m.Conv2d_3b_1x1(x); x = m.Conv2d_4a_3x3(x); x = m.maxpool2(x)
            x = m.Mixed_5b(x); x = m.Mixed_5c(x); x = m.Mixed_5d(x)
            x = m.Mixed_6a(x); x = m.Mixed_6b(x); x = m.Mixed_6c(x); x = m.Mixed_6d(x); x = m.Mixed_6e(x)
            x = m.Mixed_7a(x); x = m.Mixed_7b(x); x = m.Mixed_7c(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            return torch.flatten(x, 1)
        return fwd
    if name == 'EfficientNetV2_S':
        return lambda x: model.forward_features(x).mean(dim=(2, 3))  # timm global feat
    if name == 'PVCNN':
        def fwd(x):
            x = model.stem(x); x = model.stage1(x); x = model.stage2(x)
            x = model.stage3(x); x = model.stage4(x); x = model.pool(x)
            return torch.flatten(x, 1)
        return fwd
    raise ValueError(name)

# ----------------------------------------------------------------------------
# 4. Build identity list from filenames:  01_001_L-PALMAR_01_RL850.png
#    identity = subject + hand   (per-hand classes, the locked decision)
# ----------------------------------------------------------------------------
def parse_identity(fname):
    """Return (identity_key, ) from filename. Robust to minor format drift."""
    stem = Path(fname).stem
    # subject = second underscore field; hand = the L or R token
    parts = stem.split('_')
    subj = parts[1] if len(parts) > 1 else parts[0]
    hand_match = re.search(r'_([LR])[-_]', stem) or re.search(r'\b([LR])-PALMAR', stem)
    hand = hand_match.group(1) if hand_match else ('L' if '_L' in stem else 'R' if '_R' in stem else '?')
    return f'{subj}_{hand}'

def transform_for(name):
    size = 299 if name == 'InceptionV3' else 224
    return T.Compose([T.Resize((size, size)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# ----------------------------------------------------------------------------
# 5. EER / metric helpers
# ----------------------------------------------------------------------------
def compute_eer(genuine, impostor):
    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    roc_auc = auc(fpr, tpr)
    return eer * 100, roc_auc, fpr, tpr, thr[idx]

def cosine_matrix(emb):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    return emb @ emb.T

# ----------------------------------------------------------------------------
# 6. Main loop
# ----------------------------------------------------------------------------
def main():
    img_paths = sorted(Path(PLUSVEIN_PREPROCESSED).glob('*.png'))
    if not img_paths:
        print(f'No images at {PLUSVEIN_PREPROCESSED} — check the path.'); return
    identities = [parse_identity(p.name) for p in img_paths]
    uniq = sorted(set(identities))
    print(f'PLUSVein: {len(img_paths)} images | {len(uniq)} hand-identities')

    id_to_idx = {k: i for i, k in enumerate(uniq)}
    labels = np.array([id_to_idx[i] for i in identities])

    # precompute genuine/impostor pair index lists (shared across models)
    idx_by_id = {}
    for i, lab in enumerate(labels):
        idx_by_id.setdefault(lab, []).append(i)
    genuine_pairs = []
    for lab, idxs in idx_by_id.items():
        genuine_pairs += list(itertools.combinations(idxs, 2))
    all_pairs = list(itertools.combinations(range(len(img_paths)), 2))
    genuine_set = set(genuine_pairs)
    impostor_pairs = [p for p in all_pairs if p not in genuine_set]
    if len(impostor_pairs) > MAX_IMPOSTOR_PAIRS:
        impostor_pairs = random.sample(impostor_pairs, MAX_IMPOSTOR_PAIRS)
    print(f'Genuine pairs: {len(genuine_pairs)} | Impostor pairs (capped): {len(impostor_pairs)}')

    summary = []
    roc_curves = {}   # name -> (fpr, tpr, eer, auc) for a correct combined ROC

    for name in MODEL_REGISTRY:
        model = load_model(name)
        if model is None:
            continue
        feat_fn = to_feature_extractor(name, model)
        tfm = transform_for(name)

        embs = []
        with torch.no_grad():
            for p in tqdm(img_paths, desc=f'{name} embed', leave=False):
                img = Image.open(p).convert('RGB')
                x = tfm(img).unsqueeze(0).to(DEVICE)
                with torch.autocast('cuda', enabled=torch.cuda.is_available()):
                    e = feat_fn(x)
                embs.append(e.float().squeeze(0).cpu().numpy())
        embs = np.stack(embs)

        sim = cosine_matrix(embs)
        gen_scores = np.array([sim[i, j] for i, j in genuine_pairs])
        imp_scores = np.array([sim[i, j] for i, j in impostor_pairs])
        eer, roc_auc, fpr, tpr, thr = compute_eer(gen_scores, imp_scores)
        roc_curves[name] = (fpr, tpr, eer, roc_auc)

        summary.append({'Model': name, 'EER_%': round(eer, 3),
                        'ROC_AUC': round(roc_auc, 4),
                        'Genuine_mean': round(float(gen_scores.mean()), 4),
                        'Impostor_mean': round(float(imp_scores.mean()), 4),
                        'Threshold': round(float(thr), 4)})
        print(f'  {name:20s} | EER {eer:6.2f}% | AUC {roc_auc:.4f}')

        # per-model score distribution
        plt.figure(figsize=(7, 4))
        plt.hist(imp_scores, bins=60, alpha=0.6, label='Impostor', color='#c0392b', density=True)
        plt.hist(gen_scores, bins=60, alpha=0.6, label='Genuine', color='#27ae60', density=True)
        plt.axvline(thr, color='k', linestyle='--', linewidth=1, label=f'EER thr={thr:.3f}')
        plt.title(f'{name} — Genuine vs Impostor (PLUSVein RL850)')
        plt.xlabel('Cosine similarity'); plt.ylabel('Density'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'dist_{name}.png'), dpi=150); plt.close()

        del model, embs; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- combined ROC overlay (built from stored curves — correct & cheap) ----
    plt.figure(figsize=(8, 7))
    for name, (fpr, tpr, eer, roc_auc) in sorted(roc_curves.items(), key=lambda kv: kv[1][2]):
        plt.plot(fpr, tpr, linewidth=1.6, label=f'{name} (EER {eer:.1f}%, AUC {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Acceptance Rate'); plt.ylabel('True Acceptance Rate')
    plt.title('PLUSVein RL850 — Cross-Dataset Verification ROC')
    plt.legend(fontsize=8, loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'combined_roc.png'), dpi=150); plt.close()

    # Save summary table
    df = pd.DataFrame(summary).sort_values('EER_%')
    df.to_csv(os.path.join(OUT_DIR, 'plusvein_eer_summary.csv'), index=False)
    print('\n==== PLUSVein cross-dataset verification (sorted best EER) ====')
    print(df.to_string(index=False))
    print(f'\nSaved summary + ROC + per-model distributions to: {OUT_DIR}')

    # ---- Grad-CAM ----
    try:
        run_gradcam(img_paths)
    except Exception as e:
        print(f'[Grad-CAM skipped] {e}\n  (Install with: pip install grad-cam)')

# ----------------------------------------------------------------------------
# 7. Grad-CAM  (qualitative — does the model look at veins?)
# ----------------------------------------------------------------------------
def gradcam_target_layer(name, model):
    if name.startswith('ResNet'):      return [model.layer4[-1]]
    if name == 'RegNetY_8GF':          return [model.trunk_output[-1]]
    if name == 'DenseNet121':          return [model.features[-1]]
    if name in ('VGG16', 'VGG19'):     return [model.features[-1]]
    if name == 'ConvNeXt_Tiny':        return [model.features[-1][-1]]
    if name == 'MobileNetV3_Large':    return [model.features[-1]]
    if name == 'PVCNN':                return [model.stage4[0].proj[0]]  # Conv2d in proj block
    if name == 'EfficientNetV2_S':     return [model.conv_head]   # timm
    if name == 'InceptionV3':          return [model.Mixed_7c]
    return None

def run_gradcam(img_paths):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam_dir = os.path.join(OUT_DIR, 'gradcam'); os.makedirs(cam_dir, exist_ok=True)
    samples = random.sample(img_paths, min(GRADCAM_SAMPLES, len(img_paths)))

    for name in GRADCAM_MODELS:
        model = load_model(name)
        if model is None: continue
        layers = gradcam_target_layer(name, model)
        if layers is None: continue
        tfm = transform_for(name)
        cam = GradCAM(model=model, target_layers=layers)

        fig, axes = plt.subplots(2, len(samples), figsize=(2.2*len(samples), 4.6))
        for j, p in enumerate(samples):
            img = Image.open(p).convert('RGB')
            x = tfm(img).unsqueeze(0).to(DEVICE)
            grayscale = cam(input_tensor=x)[0]
            size = 299 if name == 'InceptionV3' else 224
            base = np.array(img.resize((size, size)), dtype=np.float32) / 255.0
            vis = show_cam_on_image(base, grayscale, use_rgb=True)
            axes[0][j].imshow(base); axes[0][j].axis('off')
            axes[1][j].imshow(vis);  axes[1][j].axis('off')
            if j == 0:
                axes[0][j].set_ylabel('input', fontsize=9)
                axes[1][j].set_ylabel('Grad-CAM', fontsize=9)
        fig.suptitle(f'Grad-CAM — {name} (PLUSVein RL850)', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(cam_dir, f'gradcam_{name}.png'), dpi=150)
        plt.close()
        print(f'  Grad-CAM saved: {name}')
        del model, cam; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
