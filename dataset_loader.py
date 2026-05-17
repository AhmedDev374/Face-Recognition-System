"""
=============================================================
  dataset_loader.py
  Responsible for scanning the ORL/AT&T dataset structure:

      dataset/
          s1/   1.pgm … 10.pgm
          s2/   1.pgm … 10.pgm
          …
          s40/  1.pgm … 10.pgm

  Automatically detects all subject folders (sX) without
  any hard-coded names, so it works with any number of persons.
=============================================================
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Default image dimensions for the ORL database ─────────────
IMAGE_SIZE = (112, 92)   # (height, width)


class DatasetLoader:
    """
    Loads a face dataset where each sub-folder represents one person.

    Parameters
    ----------
    dataset_path : str   – root folder of the dataset
    image_size   : tuple – (height, width) to resize every image
    """

    def __init__(self, dataset_path: str = "dataset", image_size: tuple = IMAGE_SIZE):
        self.dataset_path = dataset_path
        self.image_size   = image_size

        # Populated after load()
        self.images    : list[np.ndarray] = []   # each: 1-D float64 vector
        self.labels    : list[str]        = []   # person folder name
        self.img_paths : list[str]        = []   # absolute file paths
        self.person_dirs: list[str]       = []   # sorted list of subject folders

    # ──────────────────────────────────────────────────────────
    def load(self, progress_callback=None) -> tuple[list, list, list]:
        """
        Walk dataset_path, read every valid image, flatten & float it.

        Parameters
        ----------
        progress_callback : callable(int, int, str) | None
            Called as (current_image_index, total_images, message)
            so the GUI can update a progress bar.

        Returns
        -------
        images, labels, img_paths  (also stored as self.*)
        """
        self.images.clear()
        self.labels.clear()
        self.img_paths.clear()

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset folder '{self.dataset_path}' not found.\n"
                "Create it and add one sub-folder per person."
            )

        # ── Discover all subject sub-directories ──────────────
        self.person_dirs = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        if not self.person_dirs:
            raise ValueError("No sub-folders found inside the dataset directory.")

        # ── Count total images for progress reporting ──────────
        all_files = []
        for person_name in self.person_dirs:
            person_dir = os.path.join(self.dataset_path, person_name)
            for f in sorted(os.listdir(person_dir)):
                if f.lower().endswith((".pgm", ".jpg", ".jpeg", ".png", ".bmp")):
                    all_files.append((person_name, os.path.join(person_dir, f)))

        total = len(all_files)
        if total == 0:
            raise ValueError("No valid image files found in any subject folder.")

        logger.info("Found %d images across %d persons", total, len(self.person_dirs))

        # ── Load every image ───────────────────────────────────
        for idx, (person_name, img_path) in enumerate(all_files):

            # Notify GUI (optional)
            if progress_callback:
                progress_callback(idx + 1, total,
                                  f"Loading {person_name} …  ({idx+1}/{total})")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Cannot read image: %s — skipped", img_path)
                continue

            # Resize to fixed dimensions (width=col, height=row for cv2)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

            # Flatten 2-D matrix → 1-D vector, convert to float64
            self.images.append(img.flatten().astype(np.float64))
            self.labels.append(person_name)
            self.img_paths.append(img_path)

        logger.info("Loaded %d images successfully", len(self.images))
        return self.images, self.labels, self.img_paths

    # ──────────────────────────────────────────────────────────
    @property
    def num_persons(self) -> int:
        """Number of unique subjects found."""
        return len(set(self.labels))

    @property
    def num_images(self) -> int:
        """Total images loaded."""
        return len(self.images)

    @property
    def images_per_person(self) -> dict[str, int]:
        """Mapping of person-name → number of training images."""
        counts: dict[str, int] = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def preprocess_single(img_path: str, image_size: tuple = IMAGE_SIZE
                          ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Read and preprocess ONE image (used during recognition).

        Returns
        -------
        img_flat    : 1-D float64 vector  (or None on error)
        img_display : 2-D uint8 grayscale (for display)
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Cannot read image: %s", img_path)
            return None, None

        img         = cv2.resize(img, (image_size[1], image_size[0]))
        img_display = img.copy()
        img_flat    = img.flatten().astype(np.float64)
        return img_flat, img_display
