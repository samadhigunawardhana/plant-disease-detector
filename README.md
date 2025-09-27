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
