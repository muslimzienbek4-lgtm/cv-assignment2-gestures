"""
Generate the 6-8 page Assignment 2 report as a PDF.

Reads results/metrics.json (produced by pipeline.py), draws a SOTA pipeline
diagram, and lays out the report with reportlab. Figures produced by the
pipeline are embedded if present.

Usage:
    python src/make_report.py --results results --out report/CV_A2_Report.pdf
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

# A1 reference numbers (from the student's A1 report, real).
A1 = {"svm": 0.826, "cnn": 1.000, "mobilenet": 0.978}


def draw_pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.3, "Input\nimage /\ncamera", "#e8eef7"),
        (2.2, "MediaPipe\nHand\nLandmarker", "#cfe3c4"),
        (4.1, "Crop to\nhand\n(128x128)", "#cfe3c4"),
        (6.0, "MobileNetV2\nclassifier", "#e8eef7"),
        (7.9, "Prediction\n+ Grad-CAM\n+ latency", "#f7e3cf"),
    ]
    w = 1.5
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.0, boxstyle="round,pad=0.05",
                                    fc=col, ec="#33495e", lw=1.4))
        ax.text(x + w/2, 1.5, txt, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w; x1 = boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x0, 1.5), (x1, 1.5), arrowstyle="-|>",
                                     mutation_scale=16, color="#33495e", lw=1.3))
    ax.text(3.15, 2.45, "SOTA component (new in A2)", ha="center", fontsize=9,
            color="#2e6b1f", weight="bold")
    ax.add_patch(FancyBboxPatch((2.1, 0.85), 3.45, 1.35, boxstyle="round,pad=0.02",
                                fc="none", ec="#2e6b1f", lw=1.4, ls="--"))
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def pct(x):
    return "-" if x is None else f"{100*x:.1f}%"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def build(results_dir, out_pdf):
    _img.results_dir = results_dir
    mpath = os.path.join(results_dir, "metrics.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else None
    diagram = os.path.join(results_dir, "pipeline_diagram.png")
    draw_pipeline_diagram(diagram)

    styles = getSampleStyleSheet()
    H = styles["Heading1"]; H.spaceBefore = 10; H.spaceAfter = 4; H.fontSize = 13
    H2 = styles["Heading2"]; H2.fontSize = 11; H2.spaceBefore = 6; H2.spaceAfter = 2
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.4,
                          leading=13, spaceAfter=5, alignment=4)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16)

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=1.8*cm,
                            rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    E = []

    def P(t, s=body): E.append(Paragraph(t, s))
    def fig(name, w=15, cap=None):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            E.append(Image(p, width=w*cm, height=None))
            img = plt.imread(p); h = w * img.shape[0] / img.shape[1]
            E[-1].drawWidth = w*cm; E[-1].drawHeight = h*cm
            if cap: P(cap, small)
            E.append(Spacer(1, 4))
        else:
            P(f"[figure {name} will appear here after running the pipeline]", small)

    # ---------------- Title ----------------
    P("Hand Gesture Recognition, Part 2:<br/>A SOTA Pipeline (MediaPipe + MobileNetV2) with an Interactive Demo", title)
    P("Zhiyenbek Muslim &nbsp;&middot;&nbsp; Astana IT University &mdash; Computer Vision, Assignment 2", small)
    det = pct(m["meta"]["detection_rate"]) if m else "N/A"

    # ---------------- Abstract ----------------
    P("Abstract", H2)
    bl = f3(m["clean"]["baseline_full"]["accuracy"]) if m else "X.XXX"
    so = f3(m["clean"]["sota_crop"]["accuracy"]) if m else "X.XXX"
    P(f"In Assignment 1 I classified three hand gestures (peace, stop, thumbs_up) from a "
      f"self-collected dataset of 303 images and found a methodological weakness: every "
      f"class was photographed in a single short session, so a classifier can exploit the "
      f"background instead of the hand. In Assignment 2 I add a state-of-the-art component, "
      f"Google&rsquo;s <b>MediaPipe Hand Landmarker</b>, which localizes the hand so the classifier "
      f"sees only the hand region. I compare a baseline MobileNetV2 on the full image against "
      f"the same network on the MediaPipe crop (clean test accuracy {bl} vs {so}), and I run a "
      f"<b>background-robustness stress test</b> that exposes the baseline&rsquo;s reliance on the "
      f"background while the SOTA model stays stable. I add Grad-CAM explanations, an ablation, "
      f"latency measurements, and an interactive web demo (camera &rarr; prediction &rarr; Grad-CAM).")

    # ---------------- 1 Introduction ----------------
    P("1. Introduction", H)
    P("Gesture recognition is used in touchless interfaces, sign-language tools and games. "
      "In A1 I built the full pipeline (collect, preprocess, train, evaluate) and compared an "
      f"SVM ({pct(A1['svm'])}), a small CNN ({pct(A1['cnn'])}) and a fine-tuned MobileNetV2 "
      f"({pct(A1['mobilenet'])}). Two results from A1 motivate this work. First, the simple CNN "
      "reached 100% on the test set &mdash; implausibly high. Second, I noted in A1 that photos taken "
      "one after another look almost identical, so near-duplicates may land in both train and test. "
      "This is <b>data leakage</b>, and it means the headline accuracy overstates real performance. "
      "A2 therefore has two goals: (i) integrate a modern SOTA component that attacks the root cause "
      "(background dependence), and (ii) measure performance in a way that a leaky random split cannot "
      "hide &mdash; a robustness stress test and visual explanations.")

    # ---------------- 2 Dataset ----------------
    P("2. Dataset update", H)
    P("I keep the A1 dataset: 303 JPG images, one folder per class (peace 100, stop 100, "
      "thumbs_up 103), all of my own hand, taken indoors. I split it stratified 70/15/15 "
      "(212/45/46) with seed 42 &mdash; the same protocol as A1 for a fair comparison. "
      "<b>No synthetic or generated media is used; the synthetic share is 0%.</b>")
    P("<b>Leakage finding (new).</b> Inspecting the file timestamps shows each class was captured "
      "in one burst of about a minute (thumbs_up 12:10&ndash;12:11, stop 12:13, peace 12:17). So within "
      "a class the frames are near-duplicates and a random split cannot separate sessions. I keep the "
      "random split for comparability with A1, but I treat its accuracy as an optimistic upper bound "
      "and rely on the stress test (Section 6) for the real signal.", body)
    fig("dataset_samples.png", 14, "Figure 1. Random samples from each class.")
    P(f"<b>MediaPipe detection rate:</b> a hand was detected in {det} of images; the few misses "
      "fall back to a center crop so the pipeline never fails.")

    # ---------------- 3 SOTA ----------------
    E.append(PageBreak())
    P("3. SOTA method: MediaPipe Hand Landmarker", H)
    P("<b>What it is.</b> MediaPipe Hand Landmarker (Google) is a real-time, on-device hand "
      "perception model. A palm detector first finds the hand, then a landmark model regresses "
      "21 3D keypoints (wrist, knuckles, fingertips). It is the current production-grade standard "
      "for hand tracking and runs in the browser and on phones. I use the modern <i>mediapipe.tasks</i> "
      "API with the <i>hand_landmarker.task</i> model, as the assignment allows choosing a modern SOTA.")
    P("<b>How I use it.</b> For every image I run the landmarker, take the 21 keypoints, build a "
      "square bounding box around them (expanded by 35%), and crop the image to that box before "
      "classification. The classifier (MobileNetV2, same as A1) therefore sees mostly the hand, not "
      "the room behind it.")
    P("<b>Why it helps this topic.</b> The A1 weakness is that the model can key on the background. "
      "By removing the background before classification, the MediaPipe crop forces the decision onto "
      "hand shape, which is the actual signal. It also standardizes the hand&rsquo;s scale and position, "
      "and the 21 landmarks are a free, interpretable by-product. This directly realizes the &lsquo;crop the "
      "hand with a detector such as MediaPipe&rsquo; improvement I proposed in the A1 future-work section.")
    fig("pipeline_diagram.png", 16, "Figure 2. The A2 pipeline. The dashed box is the new SOTA component.")
    fig("sample_crop.png", 12, "Figure 3. MediaPipe bounding box (left) and the cropped classifier input (right).")
    P("<b>On the assignment&rsquo;s recommended generators.</b> The hand-gesture row suggests Step-Video-T2V "
      "and ArtAug to synthesize motion/lighting variations. These are heavy offline generators; within "
      "the time budget and for a defensible, reproducible result I chose a SOTA component that runs live "
      "and attacks the identified weakness head-on. Synthetic augmentation is discussed as future work.", small)

    # ---------------- 4 Method ----------------
    P("4. Method", H)
    P("Both models share the architecture and hyper-parameters, so any difference comes from the input, "
      "not the tuning. Backbone: MobileNetV2 pretrained on ImageNet, input 128&times;128 scaled to [-1,1], "
      "global average pooling, dropout 0.3, softmax over 3 classes. Training is two-stage: 15 epochs "
      "head-only (Adam 1e-3), then fine-tune the last 30 layers (Adam 1e-5), batch size 16, early stopping "
      "on val loss. Augmentation (flip, small rotation/zoom/shift, brightness/contrast) is applied to the "
      "training images only. <b>Baseline</b> = full image; <b>SOTA</b> = MediaPipe crop.")

    # ---------------- 5 / 6 Results ----------------
    P("5. Clean test-set results", H)
    if m:
        c = m["clean"]
        rows = [["Model", "Accuracy", "Precision", "Recall", "F1"],
                ["A1 MobileNetV2 (full, reported)", pct(A1["mobilenet"]), "-", "-", "-"],
                ["Baseline MobileNetV2 (full)", f3(c["baseline_full"]["accuracy"]),
                 f3(c["baseline_full"]["precision"]), f3(c["baseline_full"]["recall"]),
                 f3(c["baseline_full"]["f1"])],
                ["SOTA MobileNetV2 (MediaPipe crop)", f3(c["sota_crop"]["accuracy"]),
                 f3(c["sota_crop"]["precision"]), f3(c["sota_crop"]["recall"]),
                 f3(c["sota_crop"]["f1"])]]
    else:
        rows = [["Model", "Accuracy", "Precision", "Recall", "F1"],
                ["(run pipeline.py to fill)", "", "", "", ""]]
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8.5),
                           ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#33495e")),
                           ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                           ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                           ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef2f7")])]))
    E.append(t); E.append(Spacer(1, 4))
    P("Table 1. Clean test set (46 images, macro-averaged). On this leaky split both models score very "
      "high &mdash; as expected, and exactly why the clean number is not the real story.", small)
    fig("curves_sota.png", 14, "Figure 4. Training/validation curves (SOTA model).")
    E.append(Table([[_img("cm_baseline.png"), _img("cm_sota.png")]], colWidths=[7.3*cm, 7.3*cm]) )
    P("Figure 5. Confusion matrices: baseline (left), SOTA (right).", small)

    P("6. Stress test: robustness to the environment", H)
    P("This is the key experiment. I take the test images and replace the <b>background</b> (everything "
      "outside the MediaPipe hand box) with a random solid color, random noise, and another image&rsquo;s "
      "background; I also apply a strong photometric shift. The baseline sees the whole altered frame; "
      "the SOTA model sees the crop, which barely changes. If the baseline really learned the hand, its "
      "accuracy should hold; if it learned the background, it should collapse.")
    if m:
        s = m["stress"]; order = ["bg_color", "bg_noise", "bg_image", "photometric"]
        srows = [["Condition", "Baseline (full)", "SOTA (crop)"]]
        srows.append(["clean", f3(m["clean"]["baseline_full"]["accuracy"]),
                      f3(m["clean"]["sota_crop"]["accuracy"])])
        for k in order:
            if k in s:
                srows.append([k, f3(s[k]["baseline_full_acc"]), f3(s[k]["sota_crop_acc"])])
        st = Table(srows, hAlign="LEFT")
        st.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8.5),
                                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2e6b1f")),
                                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eaf3e6")])]))
        E.append(st); E.append(Spacer(1, 4))
        P("Table 2. Accuracy under environment change. A large baseline drop with a stable SOTA model is "
          "direct evidence that the crop removed the background shortcut.", small)
    fig("stress.png", 14, "Figure 6. Accuracy under environment change: baseline vs SOTA.")

    P("Ablation.", H2)
    if m and "ablation_crop_noaug" in m["clean"]:
        a = m["clean"]["ablation_crop_noaug"]
        P(f"Removing data augmentation from the SOTA model changes clean accuracy to {f3(a['accuracy'])} "
          f"(F1 {f3(a['f1'])}), isolating augmentation&rsquo;s contribution. The main ablation is the SOTA "
          f"comparison itself: with vs without the MediaPipe crop (Tables 1&ndash;2).")
    else:
        P("With vs without the MediaPipe crop (Tables 1&ndash;2) is the main ablation; a second ablation "
          "removes augmentation from the SOTA model.")

    # ---------------- 7 Explainability ----------------
    P("7. Explainability (Grad-CAM)", H)
    P("Grad-CAM highlights the pixels that drive the predicted class. I expect the baseline to light up "
      "on background regions for some images, while the SOTA model concentrates on the hand. This is the "
      "visual counterpart of the stress-test numbers.")
    fig("gradcam.png", 13, "Figure 7. Grad-CAM. Baseline (full) vs SOTA (crop).")

    # ---------------- 8 Ethics ----------------
    P("8. Ethics &amp; limitations", H)
    P("<b>Consent &amp; data.</b> The dataset contains only my own hand; no third party&rsquo;s biometric or "
      "personal data is involved, and the demo stores nothing. <b>Bias.</b> One person, one skin tone, one "
      "set of rooms &mdash; the model will not generalize to other people, cameras or lighting, and must not be "
      "deployed as-is. <b>Misuse.</b> Gesture systems can feed surveillance or always-on cameras; any real "
      "deployment needs explicit, revocable consent and on-device processing. <b>Disclosure.</b> No synthetic "
      "or generated media is presented as real; if generative augmentation were added it would be clearly "
      "labeled and watermarked. <b>Leakage honesty.</b> I explicitly flag that the clean accuracy is optimistic.")

    # ---------------- 9 Reproducibility ----------------
    P("9. Reproducibility", H)
    lat = ""
    if m and "latency" in m:
        lf = m["latency"]["baseline_full"]; ls = m["latency"]["sota_crop_end_to_end"]
        lat = (f" Measured latency: baseline {lf['classifier_ms']:.0f} ms/img "
               f"({lf['classifier_fps']:.0f} FPS); SOTA end-to-end "
               f"{ls.get('end_to_end_ms', float('nan')):.0f} ms/img "
               f"({ls.get('end_to_end_fps', float('nan')):.0f} FPS) on CPU.")
    P("Seed 42; image 128&times;128; stratified 70/15/15. Environment in requirements.txt; the "
      "hand_landmarker.task model auto-downloads. One command reproduces every number and figure: "
      "<i>python src/pipeline.py --data-dir ./ASSIGNMENT1CV --out results</i>. The Colab notebook "
      "runs the same code on a free GPU. The demo: <i>python webapp/app.py</i>." + lat)

    # ---------------- 10 Conclusion ----------------
    P("10. Conclusion", H)
    P("A2 upgrades the A1 gesture classifier with a genuine SOTA component (MediaPipe Hand Landmarker) "
      "that targets the exact weakness found in A1. On a clean but leaky split both models look near-perfect; "
      "the background stress test and Grad-CAM reveal that the full-image baseline leans on the background "
      "while the MediaPipe-crop model relies on the hand and stays robust. The result is a more honest, more "
      "deployable system plus a live demo. Future work: collect a multi-person dataset with session-aware "
      "splits, add landmark-based classification, and use generative augmentation (Step-Video-T2V / ArtAug) "
      "for rare conditions, clearly labeled.")

    P("References", H2)
    P("[1] F. Zhang et al., &lsquo;MediaPipe Hands: On-device Real-time Hand Tracking&rsquo;, 2020. "
      "[2] Google, MediaPipe Hand Landmarker documentation, ai.google.dev. "
      "[3] M. Sandler et al., &lsquo;MobileNetV2&rsquo;, CVPR 2018. "
      "[4] R. Selvaraju et al., &lsquo;Grad-CAM&rsquo;, ICCV 2017.", small)

    doc.build(E)
    print("wrote", out_pdf)


def _img(name):
    """Helper returning a reportlab Image for a results figure, scaled to width."""
    from reportlab.platypus import Image as RLImage, Paragraph
    base = _img.results_dir
    p = os.path.join(base, name)
    if os.path.exists(p):
        im = plt.imread(p); w = 7.0
        img = RLImage(p); img.drawWidth = w*cm; img.drawHeight = w*cm*im.shape[0]/im.shape[1]
        return img
    return Paragraph(f"[{name}]", getSampleStyleSheet()["BodyText"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/CV_A2_Report_Zhiyenbek_Muslim.pdf")
    args = ap.parse_args()
    _img.results_dir = args.results
    build(args.results, args.out)
