"""Post-hoc probe (no retraining): does the model rely on the hand or the
background? Adds a 'hand_hidden' condition to results/metrics.json and
regenerates stress.png, using the already-trained models.

hand_hidden: gray out the detected hand region, keep the background. If
accuracy collapses toward chance (~0.33), the model relies on the hand.
"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from tensorflow import keras

RES = "results"
data = {k: v for k, v in np.load(os.path.join(RES, "_cache.npz"), allow_pickle=True).items()}
tr, va, te = P.split_indices(data["y"])
m_full = keras.models.load_model(os.path.join(RES, "baseline_full_model.keras"))
m_crop = keras.models.load_model(os.path.join(RES, "sota_crop_model.keras"))
metrics = json.load(open(os.path.join(RES, "metrics.json")))


def hand_hidden_full(idx, size=P.IMG):
    out = []
    for i in idx:
        img = cv2.cvtColor(cv2.imread(str(data["paths"][i])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bx0, by0, bx1, by1 = [int(v) for v in data["bboxes"][i]]
        img = img.copy()
        img[max(0, by0):min(h, by1), max(0, bx0):min(w, bx1)] = 128  # gray the hand
        out.append(cv2.resize(img, (size, size)))
    return np.asarray(out, dtype=np.float32)


yte = data["y"][te]
# Baseline on full image with the hand grayed out (background kept).
acc_base_hidden = P.evaluate(m_full, hand_hidden_full(te), yte)["accuracy"]
# SOTA crop with the hand (i.e. the whole crop) grayed out -> pure chance ref.
gray_crops = np.full((len(te), P.IMG, P.IMG, 3), 128.0, dtype=np.float32)
acc_sota_hidden = P.evaluate(m_crop, gray_crops, yte)["accuracy"]

metrics["stress"]["hand_hidden"] = {
    "baseline_full_acc": acc_base_hidden,
    "sota_crop_acc": acc_sota_hidden,
}
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=2)
P.plot_stress(metrics, os.path.join(RES, "stress.png"))
print("hand_hidden  baseline=%.3f  sota=%.3f" % (acc_base_hidden, acc_sota_hidden))
print("updated metrics.json + stress.png")
