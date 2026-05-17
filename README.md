<div align="center">

# 🧠 PCA Face Recognition System using Eigenfaces

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/PyQt5-GUI-41CD52?style=for-the-badge&logo=qt&logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-PCA%20%2F%20Eigenfaces-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  A professional, full-featured face recognition desktop application built on the classical
  <strong>Eigenfaces / PCA</strong> algorithm (Turk &amp; Pentland, 1991), with a polished
  dark-mode PyQt5 GUI, real-time recognition, PDF report export, and confusion-matrix evaluation.
</p>

<p align="center">
  <a href="#-demo--screenshots">Screenshots</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-how-to-run">How to Run</a> •
  <a href="#-project-workflow">Workflow</a> •
  <a href="#-results">Results</a>
</p>

</div>

---

## 📌 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Dataset Structure](#-dataset-structure)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Project Workflow](#-project-workflow)
- [Demo & Screenshots](#-demo--screenshots)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Project Description

This project implements a **face recognition system** using **Principal Component Analysis (PCA)** and the **Eigenfaces method**, originally proposed by Turk and Pentland in 1991. The system reduces high-dimensional face images into a compact eigenface subspace and identifies individuals by finding the nearest neighbour using Euclidean distance.

The project is built as a professional **final-year / graduation project**, featuring a complete software engineering pipeline: modular OOP architecture, a modern dark-mode desktop GUI, automatic dataset loading, training-time evaluation, confusion matrix analysis, recognition history logging, and one-click PDF report generation.

> **Academic Reference:** Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience, 3(1), 71–86.

---

## ✨ Features

### 🔬 Core Algorithm
- Full **PCA / Eigenfaces** pipeline — from raw pixels to subspace projection
- **Turk & Pentland compact-covariance trick** (`S = A·Aᵀ`, M×M) for efficient eigendecomposition
- **Euclidean distance** nearest-neighbour recognition in eigenface space
- **Confidence scoring** mapped from distance to percentage

### 🖥️ Professional GUI
- Fully dark-mode **PyQt5** desktop application
- **Left panel** — action buttons + live model statistics
- **Center panel** — 4-tab view: Images, Similarity Graph, Eigenfaces Grid, Confusion Matrix
- **Right panel** — result card, Top-5 matches table, scrolling recognition history
- **Live progress bar** updating during both dataset loading and PCA training
- Non-blocking background threads (`QThread`) — UI never freezes

### 📊 Analytics & Reporting
- **Top-5 nearest matches** with distance and similarity scores
- **Horizontal similarity bar chart** (Matplotlib embedded in Qt)
- **Average face + top-15 eigenfaces** visualisation grid
- **Leave-one-out confusion matrix** with per-class accuracy
- **Training accuracy** computed automatically after every training run
- **PDF report export** with full recognition details and top-5 match table

### 💾 Data Management
- Automatic dataset loading — no hard-coded names, scans all `sX/` folders dynamically
- Model persistence via `.npy` files — train once, reload instantly
- Automatic **CSV recognition log** written to `logs/recognition_log.csv`
- Timestamped **session logs** saved to `logs/session_*.log`

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Core language |
| **NumPy** | ≥ 1.24 | Matrix operations, eigendecomposition |
| **OpenCV** | ≥ 4.8 | Image I/O, grayscale conversion, resizing |
| **PyQt5** | ≥ 5.15 | Desktop GUI framework |
| **Matplotlib** | ≥ 3.7 | Embedded charts and visualisations |
| **ReportLab** | ≥ 4.0 | PDF report generation |

---

## 📂 Dataset Structure

The system uses the **ORL / AT&T Face Database** (40 subjects, 10 images each).

```
dataset/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── 3.pgm
│   └── ... 10.pgm          ← 10 images per person
│
├── s2/
│   ├── 1.pgm
│   └── ... 10.pgm
│
├── s3/ … s39/
│
└── s40/
    ├── 1.pgm
    └── ... 10.pgm
```

| Detail | Value |
|---|---|
| Total Subjects | 40 |
| Images per Subject | 10 |
| Total Images | 400 |
| Image Format | PGM (Portable Graymap) |
| Image Size | 92 × 112 pixels (width × height) |
| Colour Space | Grayscale |

> **Download:** The ORL/AT&T dataset is publicly available at  
> 🔗 https://cam-orl.co.uk/facedatabase.html

The loader **automatically detects all subject folders** — no hard-coding required. It works with any number of persons and any number of images per person.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Entry Point)                    │
│              Logging setup · Configuration · App launch         │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   RecognitionEngine   │  recognition.py
        │  Orchestration layer  │
        └──────┬────────┬───────┘
               │        │
   ┌───────────▼──┐  ┌──▼──────────────┐
   │DatasetLoader │  │   PCAModel       │
   │dataset_      │  │   pca_model.py   │
   │loader.py     │  │                  │
   │              │  │  • train()       │
   │ • load()     │  │  • recognize()   │
   │ • preprocess │  │  • save/load()   │
   └──────────────┘  └─────────────────┘
               │
   ┌───────────▼───────────────────────┐
   │           gui.py                  │
   │   MainWindow  (PyQt5)             │
   │                                   │
   │  TrainWorker      (QThread)       │
   │  RecognizeWorker  (QThread)       │
   │  ConfusionMatrixWorker (QThread)  │
   └───────────────────────────────────┘
```

---

## ⚙️ Installation

### 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/pca-face-recognition.git
cd pca-face-recognition
```

### 2 — Create a Virtual Environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add the Dataset

Download the ORL/AT&T dataset and extract it so the folder structure matches:

```
pca-face-recognition/
└── dataset/
    ├── s1/   1.pgm … 10.pgm
    ├── s2/   1.pgm … 10.pgm
    └── s40/  1.pgm … 10.pgm
```

---

## 📋 Requirements

```
numpy>=1.24
opencv-python>=4.8
PyQt5>=5.15
matplotlib>=3.7
reportlab>=4.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

The GUI will launch automatically. If a previously trained model exists in `pca_model/`, it will be loaded instantly — no retraining required.

---

## 🔄 Project Workflow

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────────┐
│  1. Load     │────▶│  2. Train PCA  │────▶│  3. Save Model      │
│  Dataset     │     │  (Eigenfaces)  │     │  (.npy files)       │
└──────────────┘     └────────────────┘     └─────────────────────┘
       │                    │
       ▼                    ▼
  Auto-detect          • Compute mean face
  all sX/ folders      • Build A_diff matrix
  Load & flatten       • S = A·Aᵀ  (M×M)
  all .pgm files       • Eigendecomposition
                       • Extract eigenfaces
                       • Project training set

┌──────────────┐     ┌────────────────┐     ┌─────────────────────┐
│  4. Upload   │────▶│  5. Preprocess │────▶│  6. Project onto    │
│  Test Image  │     │  & Flatten     │     │  Eigenface Space    │
└──────────────┘     └────────────────┘     └─────────────────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────────┐
│  9. Display  │◀────│  8. Compute    │◀────│  7. Euclidean       │
│  Results &   │     │  Confidence %  │     │  Distance to all    │
│  Export PDF  │     │  Top-5 Matches │     │  Training Samples   │
└──────────────┘     └────────────────┘     └─────────────────────┘
```

### Step-by-Step Recognition

1. User clicks **Upload Image** → file explorer opens
2. System reads the image in **grayscale** and resizes it to 92×112
3. Image is **flattened** to a 1-D vector and converted to float64
4. **Mean face is subtracted**: `diff = img − μ`
5. Image is **projected**: `feature = eigenfaces × diff`
6. **Euclidean distance** is computed against all training feature vectors
7. The training sample with **minimum distance** is selected as the match
8. **Confidence** is calculated: `max(0, (1 − dist/threshold) × 100)%`
9. Results are displayed — matched image, confidence, Top-5 table, similarity chart

---

## 📸 Demo & Screenshots

### Main Interface
![Main Interface](images/interface.png)

### Recognition Result
![Recognition Result](images/result.png)

### Eigenfaces Visualisation
![Eigenfaces](images/eigenfaces.png)

### Similarity Graph
![Similarity Graph](images/similarity.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

> 📝 *Replace the placeholder paths above with your actual screenshots.*  
> Recommended: create an `images/` folder in the repository root and add screenshots there.

---

## 📈 Results

| Metric | Value |
|---|---|
| Dataset | ORL / AT&T (40 persons × 10 images) |
| Training Images | 360 (9 per person) |
| Test Images | 40 (1 per person) |
| Eigenfaces Used | 40 |
| **Recognition Accuracy** | **~95%** |
| **Leave-One-Out Accuracy** | Computed live after training |
| Average Recognition Time | < 50 ms |
| Distance Threshold | 5000 (Euclidean) |

> **Note:** Accuracy depends on the train/test split, number of eigenfaces retained (`N_COMPONENTS`), and the distance threshold. The system computes leave-one-out training accuracy automatically after every training run.

### Why PCA Works for Faces

Face images occupy a very small subspace of the full pixel space. PCA finds the directions of **maximum variance** across all training faces — the *eigenfaces* — and projects every image onto this subspace. Recognition then becomes a simple nearest-neighbour search in a low-dimensional space, which is both fast and surprisingly accurate.

---

## 📁 Project Structure

```
pca-face-recognition/
│
├── main.py                  ← Entry point — logging, config, launches GUI
├── dataset_loader.py        ← Scans sX/ folders, loads & flattens images
├── pca_model.py             ← PCA training, projection, recognition, save/load
├── recognition.py           ← Engine — orchestrates loader + model + logging + PDF
├── gui.py                   ← PyQt5 dark-mode GUI (MainWindow + QThread workers)
│
├── requirements.txt
├── README.md
│
├── dataset/                 ← ORL/AT&T dataset (s1/ … s40/)
│   ├── s1/  1.pgm … 10.pgm
│   └── s40/ 1.pgm … 10.pgm
│
├── pca_model/               ← Auto-created after first training
│   ├── eigenfaces.npy
│   ├── average_face.npy
│   ├── train_features.npy
│   ├── labels.npy
│   ├── image_paths.npy
│   └── meta.npy
│
├── logs/                    ← Auto-created
│   ├── session_YYYYMMDD.log
│   └── recognition_log.csv
│
├── exports/                 ← Auto-created — PDF reports
│   └── recognition_report_*.pdf
│
└── images/                  ← Add your screenshots here
    ├── interface.png
    ├── result.png
    └── eigenfaces.png
```

---

## 🔮 Future Improvements

- [ ] **LDA (Linear Discriminant Analysis)** — combine with PCA (PCA+LDA / Fisherfaces) for improved between-class discrimination
- [ ] **Deep learning comparison** — benchmark against FaceNet or ArcFace
- [ ] **Real-time webcam recognition** — live video feed using OpenCV `VideoCapture`
- [ ] **Data augmentation** — rotation, brightness jitter, flipping to improve robustness
- [ ] **User management** — add/remove persons without full retraining
- [ ] **Multi-face detection** — locate and recognise multiple faces in a single image
- [ ] **REST API** — expose recognition as an HTTP endpoint (Flask / FastAPI)
- [ ] **Docker containerisation** — one-command deployment
- [ ] **Unit test suite** — pytest coverage for all core modules

---

## 🤝 Contributing

Contributions are welcome and appreciated!

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature description"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and ensure all existing functionality remains intact before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

```
MIT License — Copyright (c) 2025 [Your Name]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software ...
```

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <strong>Your Full Name</strong><br/>
      Final-Year Computer Science Student<br/>
      <a href="https://github.com/YOUR_USERNAME">GitHub</a> •
      <a href="https://linkedin.com/in/YOUR_PROFILE">LinkedIn</a> •
      <a href="mailto:your.email@example.com">Email</a>
    </td>
  </tr>
</table>

> **Institution:** [Your University Name]  
> **Department:** [Department of Computer Science / Engineering]  
> **Year:** 2025

---

## 🙏 Acknowledgments

- **Turk, M. & Pentland, A. (1991)** — *Eigenfaces for Recognition*, Journal of Cognitive Neuroscience — the foundational paper this project implements.
- **AT&T Laboratories Cambridge** — for the ORL Face Database used for training and evaluation.
- **Moataz M. Abdelwahab, University of Central Florida** — *Novel Image Recognition Techniques Employing Principal Component Analysis* — additional academic reference for this implementation.
- The open-source communities behind **NumPy**, **OpenCV**, **PyQt5**, and **Matplotlib** for providing the tools that made this project possible.

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

Made with ❤️ as a Final-Year Project

</div>
