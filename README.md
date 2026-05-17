# PCA Face Recognition System — Eigenfaces Method

> **Final-Year Project**  
> Based on: *"Novel Image Recognition Techniques Employing Principal Component Analysis"*  
> — Moataz M. Abdelwahab, University of Central Florida

---

## Overview

A fully-featured, dark-mode desktop application for face recognition using the classic
**Eigenfaces / PCA** algorithm (Turk & Pentland, 1991), implemented in Python with a
professional PyQt5 GUI.

---

## Project Structure

```
face_recognition_system/
│
├── main.py            ← Entry point; launches the GUI
├── dataset_loader.py  ← Scans & loads the ORL dataset dynamically
├── pca_model.py       ← PCA / Eigenfaces training, projection, recognition
├── recognition.py     ← High-level engine; history, confusion matrix, PDF export
├── gui.py             ← Full PyQt5 dark-mode interface
│
├── requirements.txt
│
├── dataset/           ← Place the ORL/AT&T dataset here
│   ├── s1/  1.pgm … 10.pgm
│   ├── s2/  1.pgm … 10.pgm
│   └── s40/ 1.pgm … 10.pgm
│
├── pca_model/         ← Auto-created; stores trained .npy files
├── logs/              ← Auto-created; session logs + recognition CSV
└── exports/           ← Auto-created; PDF reports
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the ORL/AT&T dataset

Download from: https://cam-orl.co.uk/facedatabase.html  
Extract so the structure is `dataset/s1/`, `dataset/s2/`, … `dataset/s40/`

### 3. Run

```bash
python main.py
```

---

## Algorithm (Preserved Exactly)

| Step | Operation |
|------|-----------|
| 1 | Load & flatten all images into matrix **A** (M × pixels) |
| 2 | Compute mean face **μ** |
| 3 | Subtract mean: **A_diff = A − μ** |
| 4 | Compact covariance: **S = A_diff · A_diffᵀ** (M×M, not pixels×pixels) |
| 5 | Eigendecomposition of **S** via `numpy.linalg.eigh` |
| 6 | Convert small eigenvectors → full image-space eigenfaces (normalised) |
| 7 | Project training images: **features = eigenfaces · A_diffᵀ** |
| 8 | Recognition: nearest neighbour by Euclidean distance in eigenface space |

---

## GUI Features

| Panel | Features |
|-------|----------|
| **Left** | Upload Image, Train, Recognize, Save, Clear, Export PDF, Confusion Matrix, Exit |
| **Center — Images tab** | Uploaded image + best-match image side by side |
| **Center — Similarity tab** | Horizontal bar chart of top-5 similarity scores |
| **Center — Eigenfaces tab** | Average face + top-15 eigenfaces grid |
| **Center — Confusion Matrix tab** | Leave-one-out confusion matrix heatmap |
| **Right** | Predicted Person, Distance, Confidence, Status, Execution Time |
| **Right** | Top-5 nearest matches table |
| **Right** | Recognition history table (auto-scrolls) |
| **Bottom** | Training progress bar + live status messages |

---

## Output Files

| File | Content |
|------|---------|
| `pca_model/*.npy` | Saved model (eigenfaces, avg face, features, labels, paths, meta) |
| `logs/session_*.log` | Full session log |
| `logs/recognition_log.csv` | CSV record of every recognition |
| `exports/recognition_report_*.pdf` | Per-recognition PDF report |

---

## Configuration

All tunable parameters are in `main.py`:

```python
DATASET_PATH       = "dataset"   # Root of ORL dataset
N_COMPONENTS       = 40          # Top eigenfaces to keep
DISTANCE_THRESHOLD = 5000.0      # Euclidean accept/reject boundary
IMAGE_SIZE         = (112, 92)   # Height × Width (ORL standard)
```

---

## Dependencies

- `numpy` — matrix operations, eigendecomposition  
- `opencv-python` — image I/O and resizing  
- `PyQt5` — desktop GUI  
- `matplotlib` — embedded charts (similarity, eigenfaces, confusion matrix)  
- `reportlab` — PDF report generation  
