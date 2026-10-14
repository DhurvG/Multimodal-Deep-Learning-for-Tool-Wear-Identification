Multimodal Deep Learning for Tool Wear Identification

This repository contains a PyTorch-based multimodal deep learning framework designed to monitor and predict cutting tool wear condition. By fusing multiple visual modalities—including images of the cutting tool, produced chips, workpiece surfaces, and frequency-domain sensor representations (spectrograms and scalograms)—the model performs joint coarse-grained wear classification and fine-grained regression of tool wear dimensions.

## 📊 Dataset: Nonastreda Multimodal Dataset
The model utilizes **9 distinct visual modalities** per sample to capture different aspects of the machining process:
* **Physical Images:**
  * `tool`: Directly captures physical wear on the cutting edge.
  * `chip`: Images of the metal chips produced during machining.
  * `work`: Surface finish of the machined workpiece.
* **Sensor-to-Image Representations:**
  * `spec_x`, `spec_y`, `spec_z`: Spectrograms of high-frequency vibration signals along 3 axes.
  * `scal_x`, `scal_y`, `scal_z`: Wavelet-transform scalograms of high-frequency vibration signals along 3 axes.

---

## 🏗️ Model Architecture (`MultimodalToolWearNet`)

                              ┌─► Classification Head ──► Class (Sharp / Used / Dulled)
9 Modalities ──► [MobileNetV2] ─┼─► GIRAPE Fusion (MLP) (Inputs) (Encoder) └─► Regression Head ──► Metrics (Gaps / Flank Wear / Overhang)


1. **Modality Encoder:** Uses a shared, ImageNet-pretrained **MobileNetV2** backbone. To maximize efficiency, all 9 inputs are stacked and processed simultaneously in a single GPU-accelerated batch pass.
2. **GIRAPE Fusion Module:** Flattens and concatenates features from all 9 modalities (total of 4,608 dimensions) and projects them into a unified 256-dimensional multimodal embedding space using multi-layer perceptrons (MLP) with Batch Normalization, ReLU activations, and Dropout.
3. **Dual Prediction Heads:**
   * **Classification Head:** Categorizes tool state into `sharp` (0), `used` (1), or `dulled` (2).
   * **Regression Head:** Jointly estimates physical wear measurements in micrometers ($\\mu$m) for: `gaps`, `flank_wear`, and `overhang`.

---

## 🧪 Training & Evaluation Methodology
* **10-Fold Leave-One-Tool-Out (LOTO) Cross-Validation:** Since cutting tools exhibit unique individual signatures, evaluating performance by holding out entire tools (e.g., training on tools T2–T10, testing on T1) ensures the model generalizes robustly to unseen hardware under real shop-floor conditions.
* **Multitask Loss:** Combined loss optimized via:  
  $$\\mathcal{L}_{total} = \\mathcal{L}_{CrossEntropy} + 0.5 \\times \\mathcal{L}_{MSE}$$
* **Optimization:** Adam Optimizer ($lr=3e-4$, $wd=1e-4$) utilizing a Cosine Annealing learning rate schedule.
* **Performance Optimization:** Implemented **RAM preloading caching** for all 4,608 images to bypass Google Drive IO bottlenecks, accelerating training times from hours to minutes.

---

## 📈 Performance Results

### Leave-One-Tool-Out Cross-Validation Metrics
Our 10-fold cross-validation results demonstrate excellent generalizability across distinct cutting tools:

* **Classification Accuracy:** **95.0% \\pm 4.1%**
* **Mean Absolute Error (MAE):**
  * **Flank Wear:** $16.180 \\pm 7.519 \\mu$m
  * **Gaps:** $6.851 \\pm 3.395 \\mu$m
  * **Overhang:** $8.172 \\pm 1.782 \\mu$m

---

## 🚀 Quick Start

### Prerequisites
Ensure you have PyTorch and standard scientific computing packages installed:
```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow
Running Inference
import torch
# Load the pretrained overall best model
checkpoint = torch.load('best_model_overall.pth')
model.load_state_dict(checkpoint['model_state'])
model.eval()
