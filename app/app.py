import json, numpy as np, tensorflow as tf
from tensorflow import keras
from PIL import Image
import gradio as gr
from pathlib import Path

IMG_SIZE = (224, 224)
MODEL_PATH  = Path(__file__).resolve().parents[1] / "models" / "plant_mobilenetv2.keras"
LABELS_PATH = Path(__file__).resolve().parents[1] / "models" / "labels.json"

INFO = {
    "Healthy": "Leaf appears healthy; maintain good cultural practices.",
    "Powdery": "Powdery mildew; improve airflow, avoid overhead watering; consider fungicide.",
    "Rust":    "Rust fungus; remove infected leaves and apply appropriate controls.",
}

mdl = keras.models.load_model(MODEL_PATH)
ID2LBL = {int(k): v for k,v in json.load(open(LABELS_PATH)).items()}
class_names = [ID2LBL[i] for i in range(len(ID2LBL))]

def predict_with_tta(img, tta=4):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    xs = [x, np.fliplr(x), np.flipud(x), np.rot90(x)][:tta]
    X = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(xs))
    proba = mdl.predict(X, verbose=0).mean(axis=0)
    idx = int(np.argmax(proba))
    lines = "\n".join([f"{class_names[i]}: {p*100:.2f}%" for i,p in enumerate(proba)])
    extra = INFO.get(class_names[idx], "")
    return f"Prediction: {class_names[idx]}\n\nProbabilities:\n{lines}\n\nInfo: {extra}"

gr.Interface(
    fn=lambda im: predict_with_tta(im, 4),
    inputs=gr.Image(type="pil", label="Upload leaf image"),
    outputs="text",
    title="Plant Disease Detector (MobileNetV2)",
    description="Upload a leaf image. Uses test-time augmentation for robustness."
).launch()
