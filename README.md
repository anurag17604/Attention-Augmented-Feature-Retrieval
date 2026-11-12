# Attention-Enhanced Deep Visual Representation for Fine-Grained Fashion Image Retrieval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Springer LNNS](https://img.shields.io/badge/Springer-LNNS_Series-003366.svg)](https://www.springer.com/series/15179)

Official PyTorch implementation and evaluation suite for the ICICC 2027 (Springer Lecture Notes in Networks and Systems) research paper:  
**"Attention-Enhanced Deep Visual Representation for Fine-Grained Image Retrieval"** *(Submission ID: 344)*.

---

## 📌 Abstract & Overview
Content-based image retrieval (CBIR) for fashion requires distinguishing subtle design, texture, and silhouette variations among visually similar items. This repository provides a rigorous, leak-free visual retrieval benchmark evaluating ImageNet-pretrained convolutional residual backbones (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50, ResNeXt101) augmented with cross-dimensional **Triplet Attention** (Misra et al., WACV 2021) across chromatic (RGB) and structural (grayscale) visual domains.

### Key Architectural Highlights
- **Cross-Dimensional Interaction:** Triplet Attention captures simultaneous dependencies across $(C, H)$, $(C, W)$, and $(H, W)$ without learnable parameters.
- **Leak-Free Deduplication Protocol:** Enforces cryptographic MD5 hashing to purge exact image duplicates between query and gallery sets.
- **Independent Ground-Truth Relevance:** Rigorous evaluation against 143 fine-grained retail `articleType` categories.
- **Real-Time Efficiency:** Sub-10 ms query latency across 35,000+ catalog items on commodity hardware.

---

## 📂 Repository Structure

```
├── models/
│   └── triplet_attention.py     # Pure PyTorch Triplet Attention implementation
├── run_benchmark.py             # Full experimental evaluation & metrics suite
├── demo_retrieval.py            # Visual Top-K query recommendation demo
├── requirements.txt             # Dependency specification
└── README.md                    # Documentation & reproducibility guide
```

---

## ⚙️ Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/FineGrained-Fashion-Image-Retrieval.git
   cd FineGrained-Fashion-Image-Retrieval
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Preparation
The benchmark uses the **Fashion Product Images (Small)** dataset from Kaggle:
- Download dataset from Kaggle: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- Extract the archive to structure:
  ```
  dataset fashion/
  ├── images/          # 44,441 verified .jpg product images
  └── styles.csv       # Product metadata (id, articleType, masterCategory, etc.)
  ```

---

## 🚀 Running Experiments

### 1. Run the Full Retrieval Benchmark & Metric Evaluation
Executes feature extraction, query-gallery ranking, and computes Precision@K, Recall@K, F1@K, mAP, MRR, latency, and gallery scaling:
```bash
python run_benchmark.py \
    --dataset_dir "dataset fashion/images" \
    --styles_csv "dataset fashion/styles.csv" \
    --output_dir "retrieval_results" \
    --sample_queries 1000 \
    --batch_size 128
```

### 2. Interactive Visual Top-K Recommendation Demo
Test any query image and generate visual retrieval cards:
```bash
python demo_retrieval.py \
    --query "dataset fashion/images/9959.jpg" \
    --top_k 5 \
    --output "retrieval_demo.png"
```

---

## 📈 Evaluation Protocol & Metrics

To avoid common methodological pitfalls (such as circular ground-truth definitions or converting similarity scores to percentage accuracy):
- **Fine-Grained Relevance:** Ground truth match defined by exact catalog `articleType` (143 distinct classes: Casual Shoes, Handbags, T-Shirts, Watches, etc.).
- **Deduplication:** Queries and gallery candidates sharing identical MD5 hashes are purged during search.
- **Reported Metrics:**
  - **Precision@K ($P@K$):** $\frac{\text{Relevant Items in Top-}K}{K}$
  - **Recall@K ($R@K$):** $\frac{\text{Relevant Items in Top-}K}{\text{Total Relevant Gallery Items}}$
  - **Mean Average Precision (mAP):** Standard area under the precision-recall curve across all queries.
  - **Mean Reciprocal Rank (MRR):** Rank position of first relevant retrieved item.
  - **Similarity Metric ($S$):** $S(p,q) = \frac{1}{1 + D(p,q)}$ where $D(p,q)$ is $L_2$-normalized Euclidean distance.

---

## 📝 Citation
If you use this codebase or the audited experimental protocol in your research, please cite:

```bibtex
@inproceedings{arora2027attention,
  title={Attention-Enhanced Deep Visual Representation for Fine-Grained Image Retrieval},
  author={Arora, Divya and Chhabra, Anusha and Jain, Deepti and Sharma, Mugdha},
  booktitle={Proceedings of the International Conference on Innovative Computing and Communication (ICICC 2027)},
  series={Lecture Notes in Networks and Systems},
  publisher={Springer},
  year={2027}
}
```

---
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
