# Demo video script (3–5 min) — Assignment 2

**Gesture recognition with MediaPipe + MobileNetV2**
Author: Zhiyenbek Muslim. Record screen + webcam. Target length **4:00**.

> Tip: record the app running (`python webapp/app.py`), and have the notebook
> results open in another tab to show the plots.

---

### 0:00–0:25 — Intro (talking head)
- "Hi, I'm Muslim. In Assignment 1 I classified three hand gestures — peace, stop,
  thumbs up — with a self-collected dataset and a MobileNetV2."
- "A1 had a hidden problem: every class was shot in one session, so the model could
  cheat on the background. In Assignment 2 I fix that with a SOTA component and prove
  the fix. Let me show you."

### 0:25–1:05 — The weakness (screen: A1 report + dataset figure)
- Show the dataset samples and the file timestamps (one burst per class).
- "A random split then leaks near-duplicate backgrounds into the test set — that's why
  my A1 CNN hit an implausible 100%."

### 1:05–1:55 — The SOTA fix (screen: pipeline diagram + sample_crop figure)
- "My SOTA component is Google's MediaPipe Hand Landmarker. It finds the hand, I crop to
  it, and only then classify — so the model sees the hand, not the room."
- Show Figure 2 (pipeline) and Figure 3 (bbox → crop). "21 keypoints, real-time, runs in
  the browser."

### 1:55–2:55 — Live demo (screen: the Gradio app + your webcam)
- Open the app. Do each gesture to the webcam: **peace → stop → thumbs up**.
- For each: point to (1) the predicted label + confidence, (2) the **Grad-CAM** heatmap
  on the hand, (3) the **latency** in ms and the device.
- "Notice the Grad-CAM lights up on the hand, and inference is ~X ms."

### 2:55–3:35 — Results & the stress test (screen: stress.png + confusion matrices)
- "On the clean split both models look near-perfect — expected, because the split is leaky."
- "The real test: I randomize the background. The full-image baseline collapses from
  ~1.00 to ~Y, while the MediaPipe-crop model barely moves. That's the proof the crop
  removed the background shortcut."
- Show Grad-CAM side-by-side (baseline on background vs crop on hand).

### 3:35–4:00 — Ethics & wrap-up (talking head)
- "Caveats: one person, one setting — not deployable as-is; a real system needs a
  multi-person dataset and on-device consent. No synthetic media is faked as real."
- "So A2 turns a leaky 100% into an honest, explainable, more robust system. Code, report
  and this demo are in the repo. Thanks!"

---
**Checklist before recording**
- [ ] `python webapp/app.py` runs and the webcam works
- [ ] Notebook run once so `results/` has all figures
- [ ] Good lighting; say the numbers you actually measured (fill X/Y from `metrics.json`)
- [ ] Keep under 5:00
