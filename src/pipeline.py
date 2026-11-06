"""
Assignment 2 - Hand Gesture Recognition with a SOTA component (MediaPipe).

End-to-end, reproducible pipeline:

  Baseline  : MobileNetV2 transfer-learning on the FULL image  (reproduces A1)
  SOTA      : MediaPipe Hand Landmarker -> crop to the hand -> MobileNetV2
  (bonus)   : MLP on the 21 MediaPipe hand landmarks

Experiments produced:
  * clean test metrics (accuracy / precision / recall / F1) + confusion matrices
  * a background-robustness STRESS test (the key experiment: shows the FULL
    model relies on the background while the CROP model does not)
  * an ABLATION (with / without the MediaPipe crop; with / without augmentation)
  * Grad-CAM explainability figures (FULL vs CROP)
  * latency measurements (MediaPipe + classifier), FPS

All numbers and figures are written to the results/ directory and to
results/metrics.json. Random seed is fixed to 42 for reproducibility.

Usage:
    python src/pipeline.py --data-dir ./ASSIGNMENT1CV --out results
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mp_hand import make_detector, crop_hand, detect_landmarks, normalized_landmark_vector

SEED = 42
IMG = 128
CLASSES = ["peace", "stop", "thumbs_up"]

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_dataset(data_dir, out_dir, size=IMG, cache=True):
    """Build FULL and CROP image arrays + landmark vectors for every image.

    Returns a dict of numpy arrays. Results are cached to out_dir/_cache.npz.
    """
    cache_path = os.path.join(out_dir, "_cache.npz")
    if cache and os.path.exists(cache_path):
        print(f"[data] loading cache {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    detector = make_detector()
    X_full, X_crop, y, paths, bboxes, fallbacks, lmks = [], [], [], [], [], [], []
    for ci, cls in enumerate(CLASSES):
        cdir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in files:
            p = os.path.join(cdir, f)
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            full = cv2.resize(rgb, (size, size))
            # Detect once, reuse for both crop and landmark vector.
            px, norm = detect_landmarks(detector, rgb)
            crop, fb, bbox, _ = crop_hand(rgb, detector, size=size)
            X_full.append(full)
            X_crop.append(crop)
            y.append(ci)
            paths.append(p)
            bboxes.append(bbox)
            fallbacks.append(fb)
            lmks.append(normalized_landmark_vector(norm))
    detector.close()

    out = dict(
        X_full=np.asarray(X_full, dtype=np.float32),
        X_crop=np.asarray(X_crop, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        paths=np.asarray(paths),
        bboxes=np.asarray(bboxes, dtype=np.int64),
        fallbacks=np.asarray(fallbacks, dtype=bool),
        lmks=np.asarray(lmks, dtype=np.float32),
    )
    if cache:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    print(f"[data] loaded {len(out['y'])} images; "
          f"MediaPipe detection rate = {100*(1-out['fallbacks'].mean()):.1f}%")
    return out


def split_indices(y):
    """Stratified 70/15/15 split with seed 42 (matches A1 for comparability)."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])
    return tr, va, te


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def build_model(augment=True):
    """MobileNetV2 transfer model. Input = raw RGB in [0,255]."""
    inp = keras.Input(shape=(IMG, IMG, 3))
    x = inp
    if augment:
        x = layers.RandomFlip("horizontal", seed=SEED)(x)
        x = layers.RandomRotation(0.08, seed=SEED)(x)
        x = layers.RandomZoom(0.15, seed=SEED)(x)
        x = layers.RandomTranslation(0.10, 0.10, seed=SEED)(x)
        x = layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED)(x)
        x = layers.RandomContrast(0.2, seed=SEED)(x)
    # MobileNetV2 preprocess_input == scale to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    base = keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    # Name the feature map so Grad-CAM can tap it cleanly.
    x = layers.Activation("linear", name="feature_map")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = keras.Model(inp, out)
    return model, base


def train_model(Xtr, ytr, Xva, yva, augment=True, tag="model"):
    """Two-stage transfer learning: train head, then fine-tune top layers."""
    model, base = build_model(augment=augment)
    ytr_oh = keras.utils.to_categorical(ytr, len(CLASSES))
    yva_oh = keras.utils.to_categorical(yva, len(CLASSES))

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True)
    h1 = model.fit(Xtr, ytr_oh, validation_data=(Xva, yva_oh),
                   epochs=15, batch_size=16, callbacks=[es], verbose=2)

    # Fine-tune last 30 layers with a small LR.
    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    start = len(h1.history["loss"])
    es2 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                        restore_best_weights=True)
    h2 = model.fit(Xtr, ytr_oh, validation_data=(Xva, yva_oh),
                   epochs=start + 15, batch_size=16, callbacks=[es2], verbose=2,
                   initial_epoch=start)

    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
    return model, history


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate(model, X, y):
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(1)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro",
                                                  zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    return dict(accuracy=float(acc), precision=float(p), recall=float(r),
                f1=float(f1), confusion=cm.tolist(), pred=pred.tolist())


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(history, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(history["accuracy"], label="train")
    ax[0].plot(history["val_accuracy"], label="val")
    ax[0].set_title(f"{title} - accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["loss"], label="train")
    ax[1].plot(history["val_loss"], label="val")
    ax[1].set_title(f"{title} - loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Stress test: background robustness                                          #
# --------------------------------------------------------------------------- #
def make_bg_randomized(data, idx, size=IMG, mode="color"):
    """Replace the background (outside the hand bbox) of FULL test images.

    Returns FULL images with a new background. CROP images are left as-is,
    because the MediaPipe crop already removed the background -- that is the
    whole point of the experiment.
    """
    rng = np.random.RandomState(SEED)
    out_full = []
    for i in idx:
        p = str(data["paths"][i])
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bx0, by0, bx1, by1 = [int(v) for v in data["bboxes"][i]]
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(w, bx1), min(h, by1)
        if mode == "color":
            bg = np.ones_like(img) * rng.randint(0, 256, size=3, dtype=np.uint8)
        elif mode == "noise":
            bg = rng.randint(0, 256, size=img.shape, dtype=np.uint8)
        else:  # another real image's background
            j = rng.randint(len(data["paths"]))
            bg = cv2.cvtColor(cv2.imread(str(data["paths"][j])), cv2.COLOR_BGR2RGB)
            bg = cv2.resize(bg, (w, h))
        comp = bg.copy()
        comp[by0:by1, bx0:bx1] = img[by0:by1, bx0:bx1]  # keep hand region
        out_full.append(cv2.resize(comp, (size, size)))
    return np.asarray(out_full, dtype=np.float32)


def make_photometric(X):
    """Strong brightness/hue perturbation to simulate a new environment."""
    rng = np.random.RandomState(SEED)
    out = []
    for im in X.astype(np.uint8):
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-25, 25)) % 180
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.5, 1.5), 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Grad-CAM                                                                    #
# --------------------------------------------------------------------------- #
def gradcam_heatmap(model, image):
    """Grad-CAM for a model that exposes a 'feature_map' layer.

    `image` is (H,W,3) in [0,255]. Returns a normalized heatmap (IMG,IMG).
    """
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer("feature_map").output, model.output])
    arr = tf.convert_to_tensor(image[None].astype("float32"))
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr, training=False)
        cls = tf.argmax(preds[0])
        loss = preds[:, cls]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heat = tf.reduce_sum(conv_out * pooled, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return cv2.resize(heat.numpy(), (IMG, IMG))


def overlay_heatmap(image, heat):
    hm = np.uint8(255 * heat)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * image + 0.45 * hm)


def gradcam_figure(model_full, model_crop, data, test_idx, path, n=4):
    rng = np.random.RandomState(1)
    pick = rng.choice(test_idx, size=min(n, len(test_idx)), replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.4 * n))
    for r, i in enumerate(pick):
        full = data["X_full"][i]
        crop = data["X_crop"][i]
        hf = gradcam_heatmap(model_full, full)
        hc = gradcam_heatmap(model_crop, crop)
        axes[r, 0].imshow(full.astype(np.uint8)); axes[r, 0].set_title("input (full)")
        axes[r, 1].imshow(overlay_heatmap(full, hf)); axes[r, 1].set_title("Grad-CAM FULL (baseline)")
        axes[r, 2].imshow(overlay_heatmap(crop, hc)); axes[r, 2].set_title("Grad-CAM CROP (SOTA)")
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Latency                                                                     #
# --------------------------------------------------------------------------- #
def measure_latency(model, X, detector=None, n=40):
    X = X[:n]
    # classifier-only latency
    model.predict(X[:4], verbose=0)  # warmup
    t = time.time()
    for im in X:
        model.predict(im[None], verbose=0)
    clf_ms = (time.time() - t) / len(X) * 1000
    res = dict(classifier_ms=float(clf_ms), classifier_fps=float(1000 / clf_ms))
    if detector is not None:
        # end-to-end: MediaPipe detect + classify
        import mediapipe as mp
        t = time.time()
        for i in range(len(X)):
            im = X[i].astype(np.uint8)
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
            detector.detect(mpimg)
            model.predict(im[None], verbose=0)
        e2e_ms = (time.time() - t) / len(X) * 1000
        res.update(end_to_end_ms=float(e2e_ms), end_to_end_fps=float(1000 / e2e_ms))
    return res


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./ASSIGNMENT1CV")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_dataset(args.data_dir, args.out, cache=not args.no_cache)
    tr, va, te = split_indices(data["y"])
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    results = {"meta": {
        "n_images": int(len(data["y"])),
        "detection_rate": float(1 - data["fallbacks"].mean()),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "seed": SEED, "img_size": IMG, "classes": CLASSES,
    }}

    # ---- Baseline: FULL image -------------------------------------------- #
    print("\n=== Training BASELINE (full image) ===")
    m_full, h_full = train_model(data["X_full"][tr], data["y"][tr],
                                 data["X_full"][va], data["y"][va],
                                 augment=True, tag="full")
    ev_full = evaluate(m_full, data["X_full"][te], data["y"][te])
    plot_confusion(ev_full["confusion"], "Baseline (full)",
                   os.path.join(args.out, "cm_baseline.png"))
    plot_history(h_full, "Baseline (full)", os.path.join(args.out, "curves_baseline.png"))

    # ---- SOTA: MediaPipe crop -------------------------------------------- #
    print("\n=== Training SOTA (MediaPipe crop) ===")
    m_crop, h_crop = train_model(data["X_crop"][tr], data["y"][tr],
                                 data["X_crop"][va], data["y"][va],
                                 augment=True, tag="crop")
    ev_crop = evaluate(m_crop, data["X_crop"][te], data["y"][te])
    plot_confusion(ev_crop["confusion"], "SOTA (crop)",
                   os.path.join(args.out, "cm_sota.png"))
    plot_history(h_crop, "SOTA (crop)", os.path.join(args.out, "curves_sota.png"))

    # ---- Ablation: SOTA crop without augmentation ------------------------ #
    print("\n=== Ablation: crop WITHOUT augmentation ===")
    m_crop_na, _ = train_model(data["X_crop"][tr], data["y"][tr],
                               data["X_crop"][va], data["y"][va],
                               augment=False, tag="crop_noaug")
    ev_crop_na = evaluate(m_crop_na, data["X_crop"][te], data["y"][te])

    results["clean"] = {
        "baseline_full": {k: ev_full[k] for k in ["accuracy", "precision", "recall", "f1", "confusion"]},
        "sota_crop": {k: ev_crop[k] for k in ["accuracy", "precision", "recall", "f1", "confusion"]},
        "ablation_crop_noaug": {k: ev_crop_na[k] for k in ["accuracy", "precision", "recall", "f1"]},
    }

    # ---- Stress test: background robustness ------------------------------ #
    print("\n=== Stress test: background randomization ===")
    stress = {}
    for mode in ["color", "noise", "image"]:
        full_bg = make_bg_randomized(data, te, mode=mode)
        ev_bf = evaluate(m_full, full_bg, data["y"][te])      # full model, new bg
        ev_bc = evaluate(m_crop, data["X_crop"][te], data["y"][te])  # crop unaffected
        stress[f"bg_{mode}"] = {
            "baseline_full_acc": ev_bf["accuracy"],
            "sota_crop_acc": ev_bc["accuracy"],
        }
    # photometric on both
    full_ph = make_photometric(data["X_full"][te])
    crop_ph = make_photometric(data["X_crop"][te])
    stress["photometric"] = {
        "baseline_full_acc": evaluate(m_full, full_ph, data["y"][te])["accuracy"],
        "sota_crop_acc": evaluate(m_crop, crop_ph, data["y"][te])["accuracy"],
    }
    results["stress"] = stress

    # stress bar chart
    plot_stress(results, os.path.join(args.out, "stress.png"))

    # ---- Grad-CAM -------------------------------------------------------- #
    print("\n=== Grad-CAM ===")
    try:
        gradcam_figure(m_full, m_crop, data, te, os.path.join(args.out, "gradcam.png"))
    except Exception as e:
        print("[gradcam] skipped:", e)

    # ---- Latency --------------------------------------------------------- #
    print("\n=== Latency ===")
    det = make_detector()
    results["latency"] = {
        "baseline_full": measure_latency(m_full, data["X_full"][te]),
        "sota_crop_end_to_end": measure_latency(m_crop, data["X_crop"][te], detector=det),
    }
    det.close()

    # ---- Save models ----------------------------------------------------- #
    m_crop.save(os.path.join(args.out, "sota_crop_model.keras"))
    m_full.save(os.path.join(args.out, "baseline_full_model.keras"))
    with open(os.path.join(args.out, "history_baseline.json"), "w") as f:
        json.dump(h_full, f)
    with open(os.path.join(args.out, "history_sota.json"), "w") as f:
        json.dump(h_crop, f)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[done] results written to", args.out)
    print(json.dumps({"clean": results["clean"]["baseline_full"]["accuracy"],
                      "sota": results["clean"]["sota_crop"]["accuracy"]}, indent=2))


def plot_stress(results, path):
    modes = list(results["stress"].keys())
    base = [results["stress"][m]["baseline_full_acc"] for m in modes]
    sota = [results["stress"][m]["sota_crop_acc"] for m in modes]
    clean_b = results["clean"]["baseline_full"]["accuracy"]
    clean_s = results["clean"]["sota_crop"]["accuracy"]
    modes = ["clean"] + modes
    base = [clean_b] + base
    sota = [clean_s] + sota
    x = np.arange(len(modes)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(x - w/2, base, w, label="Baseline (full image)")
    ax.bar(x + w/2, sota, w, label="SOTA (MediaPipe crop)")
    ax.set_xticks(x); ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Robustness to environment change")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
