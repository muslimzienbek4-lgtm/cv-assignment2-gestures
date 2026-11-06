# Hand Gesture Recognition — Assignment 2 (SOTA + Demo)

**Author:** Zhiyenbek Muslim · Astana IT University — Computer Vision
**Topic (from A1):** hand gesture recognition — `peace` / `stop` / `thumbs_up`

Assignment 2 upgrades the A1 topic with a **state-of-the-art** component and ships
an interactive demo with an explanation of *why* each prediction is made.

## What's new vs Assignment 1

| | A1 baseline | A2 (this repo) |
|---|---|---|
| Input | full image | **MediaPipe Hand Landmarker → crop to the hand** |
| Classifier | MobileNetV2 (full frame) | MobileNetV2 (hand crop) |
| Weakness | can cheat on the background (one session per class) | forced to use hand shape |
| Evidence | accuracy on a leaky random split | **background-robustness stress test** + Grad-CAM |
| Demo | notebook only | **web app**: camera → prediction → Grad-CAM + latency |

**SOTA component:** [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(Google, `mediapipe.tasks` API, `hand_landmarker.task`). It localizes the hand so the
classifier sees only the hand region — this directly fixes the background-leakage
weakness identified in A1.

## Repository layout

```
CV_Assignment2/
├── src/
│   ├── mp_hand.py       # MediaPipe hand detection + cropping (SOTA component)
│   └── pipeline.py      # data → train baseline & SOTA → eval → stress → Grad-CAM → latency
├── webapp/
│   └── app.py           # Gradio demo (camera/upload → prediction → Grad-CAM + latency)
├── notebook/
│   └── CV_A2_Gestures.ipynb   # one-click Colab version of the whole pipeline
├── results/             # metrics.json + figures + saved models (produced by the pipeline)
├── report/              # 6–8 page report (PDF)
├── requirements.txt
└── README.md
```

## How to run

### 1. Dataset
Place the A1 dataset so the classes are sub-folders:
```
ASSIGNMENT1CV/peace/*.jpg
ASSIGNMENT1CV/stop/*.jpg
ASSIGNMENT1CV/thumbs_up/*.jpg
```

### 2. Train + evaluate (reproduces every number and figure)
```bash
pip install -r requirements.txt
python src/pipeline.py --data-dir ./ASSIGNMENT1CV --out results
```
Outputs: `results/metrics.json`, confusion matrices, training curves,
`stress.png`, `gradcam.png`, and the saved models `sota_crop_model.keras` /
`baseline_full_model.keras`. Seed is fixed to **42**.

**Colab:** open `notebook/CV_A2_Gestures.ipynb`, upload the dataset zip, Run all
(uses a free GPU).

### 3. Run the demo app
```bash
python webapp/app.py            # local URL at http://127.0.0.1:7860
SHARE=1 python webapp/app.py    # also prints a public share link
```

## Ethics & responsible use
The dataset shows only the author's own hand; no third-party personal data is used.
The demo processes images locally and stores nothing. The model is trained on a
small, single-person dataset and should **not** be deployed for real users without
collecting a larger, multi-person dataset (see the report, §Ethics & limitations).
No synthetic/generated media is presented as real.

## Reproducibility
- Random seed: 42 · Image size: 128×128 · Split: stratified 70/15/15
- Environment: `requirements.txt`; MediaPipe model `hand_landmarker.task` is
  auto-downloaded on first run.
