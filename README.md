# plant-disease-detector
Computer Vision Individual project
# Plant Disease Detection (CPU • Colab + Gradio)

**Course:** Computer Vision — Individual Project  
**Student:** M.D.S.G.K Gunawardhana  
**Topic:** Plant Disease Detection Using Leaf Images (Healthy / Powdery / Rust)

---

## Overview
This project implements an end-to-end computer vision pipeline on **CPU only**: data prep, two deep-learning models, evaluation, and a small web UI (Gradio) to upload a leaf image and get the predicted disease with short guidance.

**Models**
1. **Small CNN (from scratch)** — simple 3-block conv net to demonstrate baseline learning.  
2. **MobileNetV2 Transfer Learning (best)** — ImageNet-pretrained backbone, feature extraction + fine-tuning of the last layers, class weights and test-time augmentation (TTA) for robustness.

---

## Dataset
- **Kaggle:** `rashikrahmanpritom/plant-disease-recognition-dataset`  
- **Classes:** `Healthy`, `Powdery` (powdery mildew), `Rust`  
- **Splits used:**  
  - Train: **1322** images (≈458 Healthy / 430 Powdery / 434 Rust)  
  - Validation: **60** images  
  - Test: **150** images  
- License shown on Kaggle page (CC0 / Public Domain) — cite below.

Downloaded in Colab via `kagglehub`, then loaded directly from its cache path.

---

## Repository Structure



---

## How to Reproduce (Colab, CPU)
1. Open `notebooks/PlantDisease_Colab.ipynb` in Google Colab.  
2. **Runtime → Change runtime type → Hardware accelerator = None** (CPU).  
3. Run all cells. The notebook:
   - downloads the dataset via `kagglehub`
   - builds the input pipelines with preprocessing/augmentation
   - trains **Small CNN** and **MobileNetV2**
   - evaluates on the **test** set (Accuracy, Precision, Recall, F1)
   - saves artifacts to `/content/models/`:
     - `plant_mobilenetv2.keras` (best model)
     - `plant_smallcnn.keras`
     - `labels.json`
     - `*_cm.png` confusion matrices

---

## Run the Demo App Locally (VS Code)
1. Download the trained model and place it in `models/`:
   - **Model (MobileNetV2):**  
     👉 [Download `plant_mobilenetv2.keras`](https://github.com/samadhigunawardhana/plant-disease-detector/releases/download/v1.0.0/plant_mobilenetv2.keras)
   - `labels.json` is already included in this repo.
2. Create a virtual env and install deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # macOS/Linux
   pip install -r requirements.txt

## DEMO VIDEO LINK
youtube link(Unlisted)- https://youtu.be/6uR40DeM8Ag 