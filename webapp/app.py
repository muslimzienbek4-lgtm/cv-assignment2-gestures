"""
Interactive demo app (Gradio) for Assignment 2.

Flow required by the assignment:
    upload / camera  ->  prediction  ->  visual explanation (Grad-CAM)
    + latency (avg inference time) + environment (device)

The app runs the SOTA pipeline: MediaPipe Hand Landmarker finds and crops the
hand, then a MobileNetV2 classifier predicts the gesture, and Grad-CAM shows
where the model looked.

Run locally:
    pip install -r requirements.txt
    python webapp/app.py
Gradio prints a local URL (http://127.0.0.1:7860). Add share=True for a public
link you can submit.
"""

import os
import sys
import time

import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mp_hand import make_detector, crop_hand  # noqa: E402

IMG = 128
CLASSES = ["peace", "stop", "thumbs_up"]
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "results", "sota_crop_model.keras"))

print(f"[app] loading model {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
detector = make_detector()
DEVICE = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def gradcam(image_128):
    grad_model = keras.Model(model.inputs,
                             [model.get_layer("feature_map").output, model.output])
    arr = tf.convert_to_tensor(image_128[None].astype("float32"))
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr, training=False)
        cls = tf.argmax(preds[0])
        loss = preds[:, cls]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    heat = cv2.resize(heat.numpy(), (IMG, IMG))
    hm = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * image_128 + 0.45 * hm)


def predict(image):
    if image is None:
        return None, None, "Upload an image or use the camera."
    rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    t0 = time.time()
    crop, fallback, bbox, _ = crop_hand(rgb, detector, size=IMG)
    probs = model.predict(crop[None].astype("float32"), verbose=0)[0]
    dt = (time.time() - t0) * 1000

    conf = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    cam = gradcam(crop.astype("float32"))
    note = "hand detected" if not fallback else "NO hand detected (used full frame)"
    info = (f"Prediction: **{CLASSES[int(probs.argmax())]}**  \n"
            f"Latency: {dt:.1f} ms  ({1000/dt:.1f} FPS)  \n"
            f"Device: {DEVICE}  ·  MediaPipe: {note}")
    return conf, cam, info


with gr.Blocks(title="Gesture Recognition - A2 (SOTA: MediaPipe + MobileNetV2)") as demo:
    gr.Markdown("# Hand Gesture Recognition - Assignment 2\n"
                "SOTA pipeline: **MediaPipe Hand Landmarker -> crop -> MobileNetV2**, "
                "with Grad-CAM explanation. Classes: peace / stop / thumbs_up.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Prediction")
            out_cam = gr.Image(label="Grad-CAM (where the model looked)")
            out_info = gr.Markdown()
    btn.click(predict, inputs=inp, outputs=[out_label, out_cam, out_info])
    inp.change(predict, inputs=inp, outputs=[out_label, out_cam, out_info])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE", "0") == "1")
