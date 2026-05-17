"""
=============================================================
  gui.py  (FIXED)
  Professional dark-mode PyQt5 GUI for the PCA Face
  Recognition System.

  FIXES APPLIED:
  1. Similarity Graph: labels no longer overlap — uses
     tight_layout with extra right margin, text inside bars
     when bar is wide enough, outside otherwise with bbox.
  2. Confusion Matrix: properly triggers computation and
     renders seaborn-style heatmap with annotations.
  3. Rank Results: top_matches table now shows
     Rank | Person | Image | Distance | Confidence
     with exact image filenames (e.g. s1/1.pgm).
  4. Matching Logic: every row is a distinct image match,
     multiple images of the same person are kept separately.
  5. Top Matches Table: 5 columns (Rank, Person, Image,
     Distance, Confidence).
  6. Scroll support added to tables.
  7. Dark theme visibility improved.
=============================================================
"""

import os
import sys
import logging
import datetime
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSizePolicy, QStatusBar,
    QScrollArea, QFrame
)
from PyQt5.QtCore    import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui     import QPixmap, QImage, QFont, QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from recognition import RecognitionEngine

logger = logging.getLogger(__name__)

# ── COLOUR PALETTE (Catppuccin Mocha) ─────────────────────────
PALETTE = {
    "base"    : "#1e1e2e",
    "mantle"  : "#181825",
    "crust"   : "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "text"    : "#cdd6f4",
    "subtext1": "#bac2de",
    "blue"    : "#89b4fa",
    "green"   : "#a6e3a1",
    "red"     : "#f38ba8",
    "yellow"  : "#f9e2af",
    "peach"   : "#fab387",
    "lavender": "#b4befe",
    "mauve"   : "#cba6f7",
}

DARK_STYLESHEET = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {PALETTE['base']};
    color: {PALETTE['text']};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {PALETTE['surface1']};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    color: {PALETTE['lavender']};
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QPushButton {{
    background-color: {PALETTE['surface0']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['surface1']};
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 13px;
}}
QPushButton:hover  {{ background-color: {PALETTE['surface1']}; border-color: {PALETTE['blue']}; }}
QPushButton:pressed{{ background-color: {PALETTE['surface2']}; }}
QPushButton#btn_train  {{ background-color: {PALETTE['blue']}; color: {PALETTE['crust']}; font-weight: bold; }}
QPushButton#btn_train:hover  {{ background-color: {PALETTE['lavender']}; }}
QPushButton#btn_upload {{ background-color: {PALETTE['mauve']}; color: {PALETTE['crust']}; font-weight: bold; }}
QPushButton#btn_upload:hover {{ background-color: {PALETTE['lavender']}; }}
QPushButton#btn_recognize {{ background-color: {PALETTE['green']}; color: {PALETTE['crust']}; font-weight: bold; }}
QPushButton#btn_recognize:hover {{ background-color: #94e2a1; }}
QPushButton#btn_exit {{ background-color: {PALETTE['red']}; color: {PALETTE['crust']}; font-weight: bold; }}
QPushButton#btn_exit:hover {{ background-color: #ff99aa; }}
QProgressBar {{
    border: 1px solid {PALETTE['surface1']};
    border-radius: 5px;
    background-color: {PALETTE['surface0']};
    height: 18px;
    text-align: center;
    color: {PALETTE['text']};
}}
QProgressBar::chunk {{ background-color: {PALETTE['blue']}; border-radius: 4px; }}
QTableWidget {{
    background-color: {PALETTE['mantle']};
    border: none;
    gridline-color: {PALETTE['surface0']};
    selection-background-color: {PALETTE['surface1']};
    color: {PALETTE['text']};
}}
QHeaderView::section {{
    background-color: {PALETTE['surface0']};
    color: {PALETTE['lavender']};
    border: none;
    padding: 5px;
    font-weight: bold;
}}
QTabWidget::pane  {{ border: 1px solid {PALETTE['surface1']}; border-radius: 4px; }}
QTabBar::tab      {{ background: {PALETTE['surface0']}; color: {PALETTE['subtext1']}; padding: 8px 18px; border-radius: 4px 4px 0 0; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {PALETTE['surface1']}; color: {PALETTE['text']}; font-weight: bold; }}
QLabel#lbl_result_card {{
    background-color: {PALETTE['surface0']};
    border: 1px solid {PALETTE['surface1']};
    border-radius: 8px;
    padding: 12px;
}}
QScrollArea {{ border: none; }}
QStatusBar {{ background-color: {PALETTE['crust']}; color: {PALETTE['subtext1']}; }}
QScrollBar:vertical {{
    background: {PALETTE['surface0']};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {PALETTE['surface2']};
    border-radius: 5px;
    min-height: 20px;
}}
"""


# ══════════════════════════════════════════════════════════════
#  WORKER THREADS
# ══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            result = self.engine.train(
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg)
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Training error")
            self.error.emit(str(exc))


class RecognizeWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, engine, img_path):
        super().__init__()
        self.engine   = engine
        self.img_path = img_path

    def run(self):
        try:
            result = self.engine.recognize(self.img_path)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Recognition error")
            self.error.emit(str(exc))


class ConfusionMatrixWorker(QThread):
    finished = pyqtSignal(object, list)
    error    = pyqtSignal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            cm, classes = self.engine.compute_confusion_matrix()
            self.finished.emit(cm, classes)
        except Exception as exc:
            logger.exception("Confusion matrix error")
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ══════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=90):
        fig = Figure(figsize=(width, height), dpi=dpi,
                     facecolor=PALETTE["mantle"])
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor(PALETTE["mantle"])
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure = fig


# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine      = engine
        self.current_img = None
        self.last_result = None

        self.setWindowTitle("PCA Face Recognition System using Eigenfaces")
        self.setMinimumSize(1280, 780)
        self.resize(1500, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._update_stats_panel()
        self._set_status("Ready  —  Load a trained model or train a new one.")

    # ──────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        banner = QLabel("🧠  PCA Face Recognition System  —  Eigenfaces Method")
        banner.setAlignment(Qt.AlignCenter)
        banner.setStyleSheet(
            f"font-size:18px; font-weight:bold; color:{PALETTE['blue']};"
            f"background-color:{PALETTE['crust']}; border-radius:6px; padding:8px;"
        )
        root_layout.addWidget(banner)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([230, 720, 420])

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        root_layout.addWidget(self.progress_bar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ── LEFT PANEL ────────────────────────────────────────────
    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(240)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        btn_group  = QGroupBox("Actions")
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setSpacing(6)

        self.btn_upload    = self._btn("⬆  Upload Image",     "btn_upload",    self._upload_image)
        self.btn_train     = self._btn("⚙  Train Model",      "btn_train",     self._train_model)
        self.btn_recognize = self._btn("🔍  Recognize",        "btn_recognize", self._recognize)
        self.btn_save      = self._btn("💾  Save Model",       "btn_save",      self._save_model)
        self.btn_clear     = self._btn("🗑  Clear",            "btn_clear",     self._clear)
        self.btn_pdf       = self._btn("📄  Export PDF",       "btn_pdf",       self._export_pdf)
        self.btn_conf_mat  = self._btn("📊  Confusion Matrix", "btn_conf",      self._show_confusion_matrix)
        self.btn_exit      = self._btn("✖  Exit",              "btn_exit",      self.close)

        for b in [self.btn_upload, self.btn_train, self.btn_recognize,
                  self.btn_save, self.btn_clear, self.btn_pdf,
                  self.btn_conf_mat, self.btn_exit]:
            btn_layout.addWidget(b)

        layout.addWidget(btn_group)

        stats_group = QGroupBox("Model Statistics")
        sg_layout   = QVBoxLayout(stats_group)
        sg_layout.setSpacing(3)

        self.lbl_persons   = self._stat_label("Persons Loaded",  "—")
        self.lbl_images    = self._stat_label("Training Images", "—")
        self.lbl_efaces    = self._stat_label("Eigenfaces",      "—")
        self.lbl_accuracy  = self._stat_label("Train Accuracy",  "—")
        self.lbl_ttime     = self._stat_label("Training Time",   "—")
        self.lbl_threshold = self._stat_label("Threshold",
                             f"{self.engine.model.distance_threshold:.0f}")

        for w in [self.lbl_persons, self.lbl_images, self.lbl_efaces,
                  self.lbl_accuracy, self.lbl_ttime, self.lbl_threshold]:
            sg_layout.addWidget(w)

        layout.addWidget(stats_group)
        layout.addStretch()
        return panel

    # ── CENTER PANEL ──────────────────────────────────────────
    def _build_center_panel(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Image viewer
        img_tab    = QWidget()
        img_layout = QHBoxLayout(img_tab)
        img_layout.setSpacing(10)

        up_group = QGroupBox("Uploaded Image")
        up_lay   = QVBoxLayout(up_group)
        self.lbl_upload_img = self._image_label()
        up_lay.addWidget(self.lbl_upload_img)
        img_layout.addWidget(up_group)

        match_group = QGroupBox("Best Match")
        match_lay   = QVBoxLayout(match_group)
        self.lbl_match_img = self._image_label()
        match_lay.addWidget(self.lbl_match_img)
        img_layout.addWidget(match_group)

        tabs.addTab(img_tab, "📷 Images")

        # Tab 2: Similarity bar chart  ← FIX: proper layout
        sim_tab    = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        self.sim_canvas = MplCanvas(width=7, height=5)
        sim_layout.addWidget(self.sim_canvas)
        tabs.addTab(sim_tab, "📊 Similarity Graph")

        # Tab 3: Eigenfaces grid
        ef_tab    = QWidget()
        ef_layout = QVBoxLayout(ef_tab)
        self.ef_canvas = MplCanvas(width=6, height=4)
        ef_layout.addWidget(self.ef_canvas)
        btn_show_ef = QPushButton("Show Eigenfaces & Average Face")
        btn_show_ef.clicked.connect(self._show_eigenfaces)
        ef_layout.addWidget(btn_show_ef)
        tabs.addTab(ef_tab, "🎭 Eigenfaces")

        # Tab 4: Confusion matrix  ← FIX: dedicated canvas with enough space
        cm_tab    = QWidget()
        cm_layout = QVBoxLayout(cm_tab)
        cm_layout.setContentsMargins(0, 0, 0, 0)
        self.cm_canvas = MplCanvas(width=7, height=6)
        cm_layout.addWidget(self.cm_canvas)
        tabs.addTab(cm_tab, "🔲 Confusion Matrix")

        return panel

    # ── RIGHT PANEL ───────────────────────────────────────────
    def _build_right_panel(self):
        panel  = QWidget()
        panel.setMaximumWidth(440)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Result card
        res_group  = QGroupBox("Recognition Result")
        res_layout = QGridLayout(res_group)
        res_layout.setVerticalSpacing(6)

        def _row(label, row):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{PALETTE['subtext1']};")
            val = QLabel("—")
            val.setStyleSheet("font-weight:bold;")
            res_layout.addWidget(lbl, row, 0)
            res_layout.addWidget(val, row, 1)
            return val

        self.val_person   = _row("Predicted Person:",  0)
        self.val_imgfile  = _row("Matched Image:",     1)
        self.val_distance = _row("Distance:",          2)
        self.val_conf     = _row("Confidence:",        3)
        self.val_status   = _row("Status:",            4)
        self.val_time     = _row("Execution Time:",    5)
        layout.addWidget(res_group)

        # ── Top-5 matches table (5 columns) ← FIXED ──────────
        top_group  = QGroupBox("Top 5 Nearest Matches")
        top_layout = QVBoxLayout(top_group)

        self.top_table = QTableWidget(5, 5)
        self.top_table.setHorizontalHeaderLabels(
            ["Rank", "Person", "Image", "Distance", "Confidence"]
        )
        self.top_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.top_table.verticalHeader().setVisible(False)
        self.top_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Scroll support: allow table to grow and scroll
        self.top_table.setMinimumHeight(160)
        self.top_table.setMaximumHeight(200)
        self.top_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        top_layout.addWidget(self.top_table)
        layout.addWidget(top_group)

        # History table (with scroll support)
        hist_group  = QGroupBox("Recognition History")
        hist_layout = QVBoxLayout(hist_group)

        self.hist_table = QTableWidget(0, 6)
        self.hist_table.setHorizontalHeaderLabels(
            ["Time", "Image", "Person", "File", "Conf%", "Status"]
        )
        self.hist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hist_table.verticalHeader().setVisible(False)
        self.hist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hist_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        hist_layout.addWidget(self.hist_table)
        layout.addWidget(hist_group, stretch=1)

        return panel

    # ──────────────────────────────────────────────────────────
    #  HELPER WIDGET FACTORIES
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _btn(text, obj_name, slot):
        b = QPushButton(text)
        b.setObjectName(obj_name)
        b.clicked.connect(slot)
        b.setMinimumHeight(36)
        return b

    @staticmethod
    def _image_label():
        lbl = QLabel("No image loaded")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"background-color:{PALETTE['mantle']};"
            f"border:1px solid {PALETTE['surface1']};"
            "border-radius:6px; color:#6c7086;"
        )
        lbl.setMinimumSize(220, 240)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return lbl

    @staticmethod
    def _stat_label(title, value):
        frame  = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 2, 4, 2)
        t = QLabel(title + ":")
        t.setStyleSheet(f"color:{PALETTE['subtext1']}; font-size:11px;")
        v = QLabel(value)
        v.setStyleSheet(f"color:{PALETTE['blue']}; font-weight:bold; font-size:11px;")
        v.setObjectName(f"stat_{title.replace(' ', '_')}")
        layout.addWidget(t)
        layout.addStretch()
        layout.addWidget(v)
        return frame

    # ──────────────────────────────────────────────────────────
    #  STATUS / PROGRESS
    # ──────────────────────────────────────────────────────────
    def _set_status(self, msg):
        self.status_bar.showMessage(f"  {msg}")

    def _show_progress(self, visible, value=0, text=""):
        self.progress_bar.setVisible(visible)
        self.progress_bar.setValue(value)
        if text:
            self.progress_bar.setFormat(f"  {text}  (%p%)")

    # ──────────────────────────────────────────────────────────
    #  ACTIONS
    # ──────────────────────────────────────────────────────────
    def _upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Image", "",
            "Images (*.pgm *.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self.current_img = path
        self._display_image(self.lbl_upload_img, path)
        self._set_status(f"Image loaded: {os.path.basename(path)}")

    # ── TRAIN ─────────────────────────────────────────────────
    def _train_model(self):
        self._show_progress(True, 0, "Initialising …")
        self._set_buttons_enabled(False)
        self._set_status("Training …  please wait.")

        self._train_worker = TrainWorker(self.engine)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.start()

    def _on_train_progress(self, pct, msg):
        self._show_progress(True, pct, msg)
        self._set_status(msg)

    def _on_train_finished(self, result):
        self._show_progress(False)
        self._set_buttons_enabled(True)

        if result["success"]:
            self._update_stats_panel()
            self._show_eigenfaces()
            QMessageBox.information(
                self, "Training Complete",
                f"✅  Model trained successfully!\n\n"
                f"  Persons         : {result['num_persons']}\n"
                f"  Training Images : {result['num_images']}\n"
                f"  Eigenfaces      : {result['n_eigenfaces']}\n"
                f"  Training Time   : {result['training_time']:.2f}s\n"
                f"  Train Accuracy  : {result['training_accuracy']:.1f}%"
            )
            self._set_status("Training complete.")
        else:
            QMessageBox.critical(self, "Training Failed",
                                  "❌  Training failed — check dataset path.")
            self._set_status("Training failed.")

    def _on_train_error(self, msg):
        self._show_progress(False)
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Training Error", f"❌  {msg}")
        self._set_status("Training error.")

    # ── RECOGNIZE ─────────────────────────────────────────────
    def _recognize(self):
        if not self.current_img:
            QMessageBox.warning(self, "No Image", "Please upload a test image first.")
            return
        if not self.engine.model.is_trained:
            QMessageBox.warning(self, "No Model",
                                 "No trained model found.\nPlease train the model first.")
            return

        self._set_buttons_enabled(False)
        self._set_status("Recognizing …")
        self._show_progress(True, 50, "Running recognition …")

        self._recog_worker = RecognizeWorker(self.engine, self.current_img)
        self._recog_worker.finished.connect(self._on_recog_finished)
        self._recog_worker.error.connect(self._on_recog_error)
        self._recog_worker.start()

    def _on_recog_finished(self, result):
        self._show_progress(False)
        self._set_buttons_enabled(True)
        self.last_result = result

        # Display matched image
        self._display_image(self.lbl_match_img, result["best_img_path"])

        # Fill result card
        color = PALETTE["green"] if result["accepted"] else PALETTE["red"]
        status_text = "✅  ACCEPTED" if result["accepted"] else "❌  REJECTED"

        self.val_person  .setText(result["best_person"])
        self.val_imgfile .setText(result["best_image_file"])
        self.val_distance.setText(f"{result['best_distance']:.2f}")
        self.val_conf    .setText(f"{result['confidence_pct']:.1f}%")
        self.val_status  .setText(status_text)
        self.val_status  .setStyleSheet(f"font-weight:bold; color:{color};")
        self.val_time    .setText(f"{result['elapsed_ms']:.1f} ms")

        # ── Top-5 table with 5 columns ← FIXED ────────────────
        self.top_table.setRowCount(len(result["top_matches"]))
        for row, match in enumerate(result["top_matches"]):
            vals = [
                str(row + 1),
                match["person"],
                match["image"],
                f"{match['distance']:.2f}",
                f"{match['confidence']:.1f}%",
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight best match row
                bg = QColor(PALETTE["surface1"]) if row == 0 \
                     else QColor(PALETTE["mantle"])
                item.setBackground(bg)
                self.top_table.setItem(row, col, item)

        # History table
        self._append_history(result)

        # ── Similarity bar chart ← FIXED ──────────────────────
        self._plot_similarity(result)

        self._set_status(
            f"Recognition complete  —  {result['best_person']}/{result['best_image_file']}  "
            f"({result['confidence_pct']:.1f}%  confidence)"
        )

    def _on_recog_error(self, msg):
        self._show_progress(False)
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Recognition Error", f"❌  {msg}")
        self._set_status("Recognition error.")

    # ── SAVE ──────────────────────────────────────────────────
    def _save_model(self):
        if not self.engine.model.is_trained:
            QMessageBox.warning(self, "No Model", "No trained model to save.")
            return
        if self.engine.model.save():
            QMessageBox.information(self, "Model Saved",
                                     f"✅  Model saved to '{self.engine.model.model_dir}/'")
            self._set_status("Model saved.")
        else:
            QMessageBox.critical(self, "Save Failed", "❌  Could not save model.")

    # ── CLEAR ─────────────────────────────────────────────────
    def _clear(self):
        self.current_img = None
        self.last_result = None
        self.lbl_upload_img.setText("No image loaded")
        self.lbl_upload_img.setPixmap(QPixmap())
        self.lbl_match_img .setText("No image loaded")
        self.lbl_match_img .setPixmap(QPixmap())
        for lbl in [self.val_person, self.val_imgfile, self.val_distance,
                    self.val_conf, self.val_status, self.val_time]:
            lbl.setText("—")
            lbl.setStyleSheet("font-weight:bold;")
        self.top_table.clearContents()
        self.top_table.setRowCount(5)
        self._clear_canvas(self.sim_canvas)
        self._set_status("Cleared.")

    # ── EXPORT PDF ────────────────────────────────────────────
    def _export_pdf(self):
        if not self.last_result or not self.current_img:
            QMessageBox.warning(self, "No Result",
                                 "Please recognize a face first.")
            return
        try:
            pdf_path = self.engine.export_result_to_pdf(
                self.last_result, self.current_img
            )
            QMessageBox.information(self, "PDF Exported",
                                     f"✅  Report saved:\n{pdf_path}")
            self._set_status(f"PDF exported: {os.path.basename(pdf_path)}")
        except ImportError:
            QMessageBox.warning(self, "Missing Library",
                                 "Install reportlab first:\n  pip install reportlab")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"❌  {exc}")

    # ── CONFUSION MATRIX  ← FIXED ─────────────────────────────
    def _show_confusion_matrix(self):
        if not self.engine.model.is_trained:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        # Show info but don't block with a modal dialog before computing
        self._set_status("Computing confusion matrix (leave-one-out) …")
        self._set_buttons_enabled(False)
        self._show_progress(True, 30, "Computing confusion matrix …")

        self._cm_worker = ConfusionMatrixWorker(self.engine)
        self._cm_worker.finished.connect(self._on_cm_finished)
        self._cm_worker.error.connect(self._on_cm_error)
        self._cm_worker.start()

    def _on_cm_error(self, msg):
        self._show_progress(False)
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Confusion Matrix Error", f"❌  {msg}")
        self._set_status("Confusion matrix error.")

    def _on_cm_finished(self, cm, classes):
        """Render the confusion matrix as a seaborn-style heatmap."""
        self._show_progress(False)
        self._set_buttons_enabled(True)

        if cm is None or len(classes) == 0:
            QMessageBox.warning(self, "No Data",
                                 "No prediction data available to display.")
            return

        # Clear and rebuild figure for best rendering
        fig = self.cm_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(PALETTE["mantle"])

        n      = len(classes)
        # Scale figure height with number of classes
        needed_h = max(5, min(n * 0.3, 14))
        fig.set_size_inches(8, needed_h)

        ax = fig.add_subplot(111)
        ax.set_facecolor(PALETTE["mantle"])

        # Use seaborn heatmap if available, else imshow
        if HAS_SEABORN:
            # Only annotate if matrix is small enough to read
            annotate = n <= 20
            sns.heatmap(
                cm,
                ax=ax,
                cmap="Blues",
                annot=annotate,
                fmt="d" if annotate else "",
                xticklabels=classes,
                yticklabels=classes,
                linewidths=0.3 if n <= 20 else 0,
                cbar=True,
                square=False,
            )
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=90,
                fontsize=max(4, min(8, 120 // n)),
                color=PALETTE["text"],
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=max(4, min(8, 120 // n)),
                color=PALETTE["text"],
            )
        else:
            # Fallback: plain imshow with sparse ticks
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
            step = max(1, n // 20)
            ticks = list(range(0, n, step))
            tick_labels = [classes[i] for i in ticks]

            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=90,
                                fontsize=6, color=PALETTE["text"])
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels, fontsize=6, color=PALETTE["text"])
            fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xlabel("Predicted Label", color=PALETTE["text"], fontsize=10, labelpad=8)
        ax.set_ylabel("True Label",      color=PALETTE["text"], fontsize=10, labelpad=8)
        ax.set_title("Confusion Matrix (Leave-One-Out Evaluation)",
                      color=PALETTE["blue"], fontsize=11, pad=12)

        # Style spines and ticks
        ax.tick_params(colors=PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["surface1"])

        fig.tight_layout(pad=1.5)
        self.cm_canvas.draw()

        # Stats popup
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums   = cm.sum(axis=1)
            per_class  = np.where(row_sums > 0,
                                  np.diag(cm) / row_sums, 0.0)
        total   = cm.sum()
        overall = (np.trace(cm) / total * 100) if total > 0 else 0.0

        QMessageBox.information(self, "Confusion Matrix",
            f"✅  Overall LOO Accuracy : {overall:.1f}%\n"
            f"Min per-class accuracy  : {per_class.min()*100:.1f}%\n"
            f"Max per-class accuracy  : {per_class.max()*100:.1f}%\n"
            f"Classes evaluated       : {len(classes)}"
        )
        self._set_status(f"Confusion matrix computed — LOO accuracy: {overall:.1f}%")

    # ──────────────────────────────────────────────────────────
    #  VISUALISATION HELPERS
    # ──────────────────────────────────────────────────────────
    def _show_eigenfaces(self):
        if not self.engine.model.is_trained:
            return

        fig = self.ef_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(PALETTE["mantle"])

        h, w   = self.engine.model.image_size
        n_show = min(15, self.engine.model.n_eigenfaces)
        cols   = 8
        rows   = (n_show + 1 + cols - 1) // cols

        axes_list = fig.subplots(rows, cols, squeeze=False)

        def _render(ax, data, title):
            ax.imshow(data.reshape(h, w), cmap="gray")
            ax.set_title(title, color=PALETTE["text"], fontsize=6, pad=2)
            ax.axis("off")

        avg = self.engine.model.average_face
        _render(axes_list[0][0],
                (avg - avg.min()) / (avg.max() - avg.min() + 1e-10),
                "Avg Face")

        for i in range(n_show):
            r   = (i + 1) // cols
            c   = (i + 1) %  cols
            ef  = self.engine.model.eigenfaces[i]
            ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
            _render(axes_list[r][c], ef_disp, f"EF {i+1}")

        total_slots = rows * cols
        for idx in range(n_show + 1, total_slots):
            r, c = idx // cols, idx % cols
            axes_list[r][c].axis("off")

        fig.suptitle("Average Face + Top Eigenfaces",
                      color=PALETTE["blue"], fontsize=10)
        fig.tight_layout()
        self.ef_canvas.draw()

    def _plot_similarity(self, result):
        """
        FIX: Horizontal bar chart with non-overlapping labels.
        - Labels are drawn inside bars when bar is wide enough,
          outside with a background patch when bar is narrow.
        - Right margin enlarged to prevent clip.
        - Labels show 'person/image' to differentiate same-person images.
        """
        ax = self.sim_canvas.axes
        ax.clear()

        fig = self.sim_canvas.figure
        fig.patch.set_facecolor(PALETTE["mantle"])
        ax.set_facecolor(PALETTE["mantle"])

        matches = result["top_matches"]
        # Build labels as "person/image"
        bar_labels = [f"{m['person']}/{m['image']}" for m in matches]
        sims       = [m["confidence"] for m in matches]

        # Reverse so Rank 1 is at top
        bar_labels = bar_labels[::-1]
        sims       = sims[::-1]

        colors = [PALETTE["green"] if i == len(sims) - 1
                  else PALETTE["blue"]
                  for i in range(len(sims))]

        bars = ax.barh(
            range(len(bar_labels)), sims,
            color=colors,
            edgecolor=PALETTE["surface1"],
            height=0.55,
        )

        ax.set_yticks(range(len(bar_labels)))
        ax.set_yticklabels(bar_labels, color=PALETTE["text"], fontsize=9)

        # Non-overlapping percentage labels
        xlim_max = max(sims) * 1.25 if sims else 110
        xlim_max = max(xlim_max, 20)    # at least 20% space
        ax.set_xlim(0, xlim_max)

        for bar, sim in zip(bars, sims):
            bar_w  = bar.get_width()
            bar_cx = bar.get_y() + bar.get_height() / 2
            label  = f"{sim:.1f}%"

            # If bar is wide enough, put label inside; otherwise outside
            if bar_w > xlim_max * 0.18:
                ax.text(
                    bar_w - xlim_max * 0.01, bar_cx,
                    label,
                    va="center", ha="right",
                    color=PALETTE["crust"],
                    fontsize=8.5, fontweight="bold",
                )
            else:
                ax.text(
                    bar_w + xlim_max * 0.015, bar_cx,
                    label,
                    va="center", ha="left",
                    color=PALETTE["text"],
                    fontsize=8.5,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=PALETTE["surface0"],
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )

        ax.set_xlabel("Confidence (%)", color=PALETTE["text"], fontsize=9)
        ax.set_title("Top Matches — Confidence Scores",
                      color=PALETTE["blue"], fontsize=10, pad=8)
        ax.tick_params(colors=PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["surface1"])

        # Extra right margin so labels outside bars don't clip
        fig.tight_layout(rect=[0, 0, 0.97, 1])
        self.sim_canvas.draw()

    @staticmethod
    def _clear_canvas(canvas):
        canvas.axes.clear()
        canvas.draw()

    # ──────────────────────────────────────────────────────────
    #  IMAGE DISPLAY
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _display_image(label, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            label.setText("⚠  Cannot load image")
            return
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w, ch = img_rgb.shape
        qt_img   = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap   = QPixmap.fromImage(qt_img)
        label.setPixmap(
            pixmap.scaled(label.width() or 200, label.height() or 250,
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ──────────────────────────────────────────────────────────
    #  STATS PANEL
    # ──────────────────────────────────────────────────────────
    def _update_stats_panel(self):
        m = self.engine.model
        self._set_stat(self.lbl_persons,  f"{m.num_persons}")
        self._set_stat(self.lbl_images,   f"{m.num_training_images}")
        self._set_stat(self.lbl_efaces,   f"{m.n_eigenfaces}")
        self._set_stat(self.lbl_accuracy, f"{m.training_accuracy:.1f}%" if m.is_trained else "—")
        self._set_stat(self.lbl_ttime,
                       f"{m.training_time:.2f}s" if m.is_trained else "—")

    @staticmethod
    def _set_stat(frame, value):
        for child in frame.children():
            if isinstance(child, QLabel) and "stat_" in (child.objectName() or ""):
                child.setText(value)
                return
        labels = frame.findChildren(QLabel)
        if len(labels) >= 2:
            labels[-1].setText(value)

    # ──────────────────────────────────────────────────────────
    #  HISTORY TABLE  (updated for new result format)
    # ──────────────────────────────────────────────────────────
    def _append_history(self, result):
        row = self.hist_table.rowCount()
        self.hist_table.insertRow(row)
        now    = datetime.datetime.now().strftime("%H:%M:%S")
        status = "✅" if result["accepted"] else "❌"
        vals   = [
            now,
            os.path.basename(self.current_img or ""),
            result["best_person"],
            result["best_image_file"],
            f"{result['confidence_pct']:.1f}%",
            status,
        ]
        for col, v in enumerate(vals):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 5:
                item.setForeground(QColor(PALETTE["green"] if result["accepted"]
                                          else PALETTE["red"]))
            self.hist_table.setItem(row, col, item)
        self.hist_table.scrollToBottom()

    # ──────────────────────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────────────────────
    def _set_buttons_enabled(self, enabled):
        for btn in [self.btn_train, self.btn_recognize, self.btn_upload,
                    self.btn_save, self.btn_clear, self.btn_pdf,
                    self.btn_conf_mat, self.btn_exit]:
            btn.setEnabled(enabled)
