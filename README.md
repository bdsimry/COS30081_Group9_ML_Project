<!-- PROJECT HEADER -->
# 🌿 Cross-Domain Plant Species Identification

<!-- BADGES -->
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)
![DINOv2](https://img.shields.io/badge/Backbone-DINOv2-blue)
![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA-orange)
![Course](https://img.shields.io/badge/Course-COS30082%20Applied%20ML-green)

A deep learning framework designed to bridge the "Domain Gap" between dried herbarium specimens and fresh field photographs. This project evaluates state-of-the-art Vision Transformers and proposes a robust Multi-Task Learning solution for identifying species even when field training data is missing.

🔗 **[Live Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/shirctt/plant-species-classifier)**

---

<!-- THE CHALLENGE -->
## ⚠️ The Domain Shift Challenge
Identifying plants in the wild using models trained on museum (herbarium) data is difficult due to:
1.  **Visual Discrepancy:** Dried, pressed brown plants vs. 3D, green, living plants.
2.  **The Missing Pairs Problem:** 40% of the species in our dataset have abundant herbarium images but **zero** field photos for training.
3.  **Class Imbalance:** Highly skewed distribution typical of botanical datasets.

---

<!-- METHODOLOGY SECTION -->
## 🧠 Proposed Architectures

### 🔹 Baseline Approaches
*   **Mix-Stream CNNs:** Direct training on a combined stream of herbarium and field images using **ResNet50** and **EfficientNetB3**.
*   **Plant-Pretrained DINOv2:** Using frozen self-supervised features paired with classical classifiers (SVM/Random Forest).

### 🚀 Our Innovations (New Approaches)
*   **Approach A: Metric Learning (Geometric):** Uses **Triplet Margin Loss** with Semi-Hard Mining to force species into clusters based on structural geometry rather than color/texture.
*   **Approach B: Multi-Task Learning (Semantic) [Winner]:** 
    *   **Shared Backbone:** DINOv2 (ViT-B/14) with **LoRA (Low-Rank Adaptation)** adapters.
    *   **Auxiliary Heads:** Simultaneously predicts **Leaf Shape** and **Leaf Arrangement** to enforce botanical structural awareness.
    *   **Inference:** Enhanced with **TenCrop Test-Time Adaptation (TTA)** for maximum robustness.

---

<!-- QUANTITATIVE RESULTS -->
## 📊 Results & Performance

| Approach | Architecture / Method | Overall (%) | With Pairs (%) | Herbarium-Only (%) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline 1 | ResNet50 | 48.79% | 65.36% | 1.85% |
| Baseline 2 | DINOv2 + SVM | 72.95% | 97.39% | 3.70% |
| Approach A | Metric Learning (LoRA+SVM+TTA) | 61.35% | 75.82% | 20.37% |
| **Approach B** | **Multi-Task (LoRA+TTA)** | **80.69%** | **94.77%** | **40.74%** |

**Key Takeaway:** Our Multi-Task LoRA model increased accuracy on "unseen" species by **over 10x** compared to standard CNN baselines.

---

<!-- TECH STACK -->
## 🛠️ Technical Stack
*   **Core:** Python 3.x, PyTorch
*   **Models:** Timm (DINOv2, ResNet, EfficientNet)
*   **Fine-Tuning:** PEFT (LoRA Injection)
*   **Processing:** OpenCV, Scikit-Learn (SVM/KNN/RF)
*   **GUI:** Developed for real-time plant identification

---

<!-- TEAM SECTION -->
## 👥 Group 9 Members
* **Brenda Ru Yi SIM** (102778817) - Pipeline Design & Multi-Task LoRA Implementation.
* **Elaine Yung Hui HO** (102776251) - Baseline 1 (ResNet50) & Metric Learning Implementation.
* **Li En CHAI** (104381372) - Baseline 2 (DINOv2 + SVM) Implementation.
* **Shirleen Tsze Ting CHUO** (102776497) - Baseline 1 (EfficientNet) & GUI Development.

---

<!-- REPOSITORY STRUCTURE -->
## 📂 Repository Structure
```text
├── Baseline1/
│   ├── Baseline1_EfficientNet.ipynb
│   └── Baseline1_Resnet50.ipynb
├── Baseline2/
│   ├── DINOv2+SVM.ipynb
│   └── DinoV2_RF.ipynb
├── NewAppproachA_Metric_Learning/
│   ├── NewApproachA_KNN.ipynb
│   ├── NewApproachA_KNN_LoRA.ipynb
│   └── NewApproachA_SVM.ipynb
├── NewApproachB_MultiTask_Learning/
│   ├── Data_Aligner_For_New_Approach_B.ipynb
│   ├── NewApproachB_LoRA_TTA.ipynb
│   ├── NewApproachB_Visual_Descriptors_Only.ipynb
│   ├── prepare_visual_metadata.py
│   ├── species_to_leaf_arrangement.json
│   ├── species_to_leaf_shape.json
│   ├── aligned_train_embeddings.npy  # Pre-computed features
│   └── aligned_train_metadata.pkl    # Processed metadata
├── plant-species-classifier/         # App & Deployment Files
└── README.md
```
---

*Developed for the Final Year Project at Swinburne University of Technology.*
