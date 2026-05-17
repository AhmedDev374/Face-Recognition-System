"""
=============================================================
  recognition.py  (FIXED)
  High-level recognition engine.

  FIXES:
  - compute_confusion_matrix: now collects true/pred labels
    correctly and returns a proper confusion matrix
  - _log_result: updated for new top_matches format
  - export_result_to_pdf: updated for new top_matches format
=============================================================
"""

import os
import csv
import time
import logging
import datetime
import numpy as np
import cv2

from dataset_loader import DatasetLoader
from pca_model      import PCAModel

logger = logging.getLogger(__name__)

LOG_DIR    = "logs"
EXPORT_DIR = "exports"


class RecognitionEngine:
    def __init__(self,
                 dataset_path: str   = "dataset",
                 model_dir: str      = "pca_model",
                 n_components: int   = 40,
                 distance_threshold: float = 5000.0,
                 image_size: tuple   = (112, 92)):

        self.dataset_path = dataset_path
        self.image_size   = image_size

        self.loader = DatasetLoader(dataset_path, image_size)
        self.model  = PCAModel(n_components      = n_components,
                               image_size         = image_size,
                               distance_threshold = distance_threshold,
                               model_dir          = model_dir)

        self.history = []

        os.makedirs(LOG_DIR,    exist_ok=True)
        os.makedirs(EXPORT_DIR, exist_ok=True)

        self.model.load()

    # ══════════════════════════════════════════════════════════
    #  TRAIN
    # ══════════════════════════════════════════════════════════
    def train(self, progress_callback=None) -> dict:
        def _loader_progress(current, total, msg):
            pct = int((current / total) * 40)
            if progress_callback:
                progress_callback(pct, msg)

        images, labels, img_paths = self.loader.load(_loader_progress)

        def _pca_progress(pct, msg):
            adjusted = 40 + int(pct * 0.6)
            if progress_callback:
                progress_callback(adjusted, msg)

        success = self.model.train(images, labels, img_paths, _pca_progress)
        if success:
            self.model.save()

        return {
            "success"          : success,
            "num_persons"      : self.model.num_persons,
            "num_images"       : self.model.num_training_images,
            "n_eigenfaces"     : self.model.n_eigenfaces,
            "training_time"    : self.model.training_time,
            "training_accuracy": self.model.training_accuracy,
        }

    # ══════════════════════════════════════════════════════════
    #  RECOGNIZE
    # ══════════════════════════════════════════════════════════
    def recognize(self, img_path: str, top_n: int = 5) -> dict:
        if not self.model.is_trained:
            raise RuntimeError("Model not trained. Please train or load a model first.")

        t0 = time.perf_counter()

        img_flat, img_display = DatasetLoader.preprocess_single(
            img_path, self.image_size
        )
        if img_flat is None:
            raise ValueError(f"Cannot read image: {img_path}")

        feature_vector = self.model.extract_features(img_flat)
        result = self.model.recognize(feature_vector, top_n)
        result["img_display"] = img_display
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000

        # ── OVERRIDE: Set accepted to True ONLY if confidence > 90% ──
        if result.get("confidence_pct", 0) > 90.0:
            result["accepted"] = True
        else:
            result["accepted"] = False

        self._log_result(img_path, result)
        return result
    def _log_result(self, img_path: str, result: dict):
        entry = {
            "timestamp"  : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_image" : os.path.basename(img_path),
            "predicted"  : result["best_person"],
            "image_file" : result["best_image_file"],
            "distance"   : f"{result['best_distance']:.2f}",
            "confidence" : f"{result['confidence_pct']:.1f}",
            "status"     : "Accepted" if result["accepted"] else "Rejected",
            "elapsed_ms" : f"{result['elapsed_ms']:.1f}",
        }
        self.history.append(entry)

        log_path = os.path.join(LOG_DIR, "recognition_log.csv")
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    # ══════════════════════════════════════════════════════════
    #  CONFUSION MATRIX  (FIXED)
    # ══════════════════════════════════════════════════════════
    def compute_confusion_matrix(self):
        """
        Leave-one-out evaluation on the training set.

        Returns
        -------
        cm      : (n_classes × n_classes) int array
        classes : sorted list of unique class names
        """
        if not self.model.is_trained:
            raise RuntimeError("Model not trained.")

        # Use person-folder names as class labels
        all_labels = self.model.labels   # one per training image
        classes    = sorted(set(all_labels))
        cls_idx    = {c: i for i, c in enumerate(classes)}
        n          = len(classes)
        cm         = np.zeros((n, n), dtype=int)

        features = self.model.features   # (n_comp, M)
        M        = features.shape[1]

        # Collect true and predicted labels for ALL training images
        y_true = []
        y_pred = []

        for i in range(M):
            test_f = features[:, i]
            dists  = np.linalg.norm(features.T - test_f, axis=1)
            dists[i] = np.inf          # exclude self

            pred_idx   = int(np.argmin(dists))
            predicted  = all_labels[pred_idx]
            actual     = all_labels[i]

            y_true.append(actual)
            y_pred.append(predicted)

            cm[cls_idx[actual]][cls_idx[predicted]] += 1

        return cm, classes

    # ══════════════════════════════════════════════════════════
    #  PDF EXPORT
    # ══════════════════════════════════════════════════════════
    def export_result_to_pdf(self, result: dict, test_img_path: str) -> str:
        try:
            from reportlab.lib.pagesizes  import A4
            from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units      import cm
            from reportlab.lib            import colors
            from reportlab.platypus       import (SimpleDocTemplate, Paragraph,
                                                   Spacer, Table, TableStyle,
                                                   Image as RLImage)
            from reportlab.lib.enums      import TA_CENTER
        except ImportError:
            raise ImportError("reportlab not installed.  Run: pip install reportlab")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path  = os.path.join(EXPORT_DIR, f"recognition_report_{timestamp}.pdf")

        doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                    topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle("Title",
                                      parent=styles["Title"],
                                      textColor=colors.HexColor("#1a237e"),
                                      fontSize=18, spaceAfter=12)
        heading_style = ParagraphStyle("Heading2",
                                        parent=styles["Heading2"],
                                        textColor=colors.HexColor("#283593"),
                                        fontSize=13, spaceBefore=12)
        normal_style = styles["Normal"]

        story = []

        story.append(Paragraph("PCA Face Recognition Report", title_style))
        story.append(Paragraph("Eigenfaces Method — University Final-Year Project",
                                normal_style))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("Recognition Details", heading_style))
        data = [
            ["Parameter",        "Value"],
            ["Timestamp",        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Test Image",       os.path.basename(test_img_path)],
            ["Predicted Person", result["best_person"]],
            ["Matched Image",    result["best_image_file"]],
            ["Distance",         f"{result['best_distance']:.2f}"],
            ["Confidence",       f"{result['confidence_pct']:.1f}%"],
            ["Status",           "✓ ACCEPTED" if result["accepted"] else "✗ REJECTED"],
            ["Processing Time",  f"{result['elapsed_ms']:.1f} ms"],
            ["Num Persons",      str(self.model.num_persons)],
            ["Num Train Images", str(self.model.num_training_images)],
            ["Num Eigenfaces",   str(self.model.n_eigenfaces)],
            ["Training Accuracy", f"{self.model.training_accuracy:.1f}%"],
        ]

        status_color = colors.HexColor("#1b5e20") if result["accepted"] \
                       else colors.HexColor("#b71c1c")

        tbl_style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#e8eaf6"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
            ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
            ("PADDING",      (0, 0), (-1, -1), 6),
            ("TEXTCOLOR",    (1, 7), (1, 7), status_color),
        ])
        tbl = Table(data, colWidths=[7*cm, 10*cm])
        tbl.setStyle(tbl_style)
        story.append(tbl)

        story.append(Paragraph("Top-5 Nearest Matches", heading_style))
        top_data = [["Rank", "Person", "Image", "Distance", "Confidence %"]]
        for rank, match in enumerate(result["top_matches"], start=1):
            top_data.append([
                str(rank),
                match["person"],
                match["image"],
                f"{match['distance']:.2f}",
                f"{match['confidence']:.1f}%",
            ])

        top_tbl = Table(top_data, colWidths=[1.5*cm, 3.5*cm, 3.5*cm, 4*cm, 4*cm])
        top_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#283593")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#e8eaf6"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(top_tbl)

        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "Generated by PCA Face Recognition System · Eigenfaces Method",
            ParagraphStyle("Footer", parent=normal_style,
                           textColor=colors.grey, fontSize=8,
                           alignment=TA_CENTER)
        ))

        doc.build(story)
        logger.info("PDF report saved: %s", pdf_path)
        return pdf_path
