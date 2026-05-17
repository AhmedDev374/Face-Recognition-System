"""
=============================================================
  main.py
  Entry point for the PCA Face Recognition System.

  Run:
      python main.py

  Requirements:
      pip install numpy opencv-python PyQt5 matplotlib reportlab
=============================================================
"""

import sys
import logging
import os
import datetime

from PyQt5.QtWidgets import QApplication

from recognition import RecognitionEngine
from gui         import MainWindow

# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  PCA Face Recognition System  —  Eigenfaces Method")
    logger.info("  Session started: %s",
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── Configuration ─────────────────────────────────────────
    DATASET_PATH       = "dataset"     # ORL/AT&T structure: dataset/s1/1.pgm …
    MODEL_DIR          = "pca_model"
    N_COMPONENTS       = 40            # Top eigenfaces to retain
    DISTANCE_THRESHOLD = 5000.0        # Euclidean accept/reject threshold
    IMAGE_SIZE         = (112, 92)     # (height, width) — ORL standard

    # ── Initialise engine (auto-loads model if it exists) ──────
    logger.info("Initialising RecognitionEngine …")
    engine = RecognitionEngine(
        dataset_path       = DATASET_PATH,
        model_dir          = MODEL_DIR,
        n_components       = N_COMPONENTS,
        distance_threshold = DISTANCE_THRESHOLD,
        image_size         = IMAGE_SIZE,
    )
    logger.info("Engine ready.  Model loaded: %s", engine.model.is_trained)

    # ── Launch GUI ────────────────────────────────────────────
    app    = QApplication(sys.argv)
    app.setStyle("Fusion")           # base style; overridden by our stylesheet

    window = MainWindow(engine)
    window.show()

    logger.info("GUI launched.")
    exit_code = app.exec_()
    logger.info("Application closed (exit code %d).", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
