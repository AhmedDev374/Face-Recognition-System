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

  Supports a train/test subject split so that no test-set
  subjects (S21–S40) are ever included in training data.

  Split configuration (controlled by TRAIN_SUBJECTS /
  TEST_SUBJECTS, or by the subject_filter parameter):
      Training : s1  – s20
      Testing  : s21 – s40
=============================================================
"""

import os
import re
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Default image dimensions for the ORL database ─────────────
IMAGE_SIZE = (112, 92)   # (height, width)

# ── Subject-range helpers ──────────────────────────────────────
TRAIN_SUBJECT_IDS = set(range(1, 21))   # s1  … s20
TEST_SUBJECT_IDS  = set(range(21, 41))  # s21 … s40


def _subject_id(folder_name: str) -> int | None:
    """
    Extract the integer subject ID from a folder name like 's7' or 'S21'.
    Returns None if the name does not match the expected sN pattern.
    """
    m = re.fullmatch(r"[sS](\d+)", folder_name)
    return int(m.group(1)) if m else None


class DatasetLoader:
    """
    Loads a face dataset where each sub-folder represents one person.

    Parameters
    ----------
    dataset_path   : str   – root folder of the dataset
    image_size     : tuple – (height, width) to resize every image
    subject_filter : set[int] | None
        When provided, only subjects whose numeric ID is in this set
        are loaded.  Pass TRAIN_SUBJECT_IDS for training or
        TEST_SUBJECT_IDS for evaluation.
        When None, ALL subjects are loaded (legacy behaviour).
    """

    def __init__(self, dataset_path: str = "dataset",
                 image_size: tuple = IMAGE_SIZE,
                 subject_filter: set | None = None):
        self.dataset_path   = dataset_path
        self.image_size     = image_size
        self.subject_filter = subject_filter   # None = load all

        # Populated after load()
        self.images    : list[np.ndarray] = []   # each: 1-D float64 vector
        self.labels    : list[str]        = []   # person folder name
        self.img_paths : list[str]        = []   # absolute file paths
        self.person_dirs: list[str]       = []   # sorted list of subject folders

    # ──────────────────────────────────────────────────────────
    def load(self, progress_callback=None) -> tuple[list, list, list]:
        """
        Walk dataset_path, read every valid image for the allowed subjects,
        flatten & float each image.

        If self.subject_filter is set, only folders whose numeric subject ID
        is in that set are loaded — guaranteeing that training and test
        subjects are fully disjoint.

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
        all_dirs = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        # ── Apply subject filter (S1–S20 vs S21–S40) ──────────
        if self.subject_filter is not None:
            filtered_dirs = []
            skipped_dirs  = []
            for d in all_dirs:
                sid = _subject_id(d)
                if sid is not None and sid in self.subject_filter:
                    filtered_dirs.append(d)
                else:
                    skipped_dirs.append(d)

            if skipped_dirs:
                logger.info(
                    "Subject filter active — skipping %d folder(s): %s",
                    len(skipped_dirs),
                    ", ".join(skipped_dirs[:10])
                    + (" …" if len(skipped_dirs) > 10 else "")
                )
            self.person_dirs = filtered_dirs
        else:
            self.person_dirs = all_dirs

        if not self.person_dirs:
            raise ValueError(
                "No subject folders matched the filter. "
                "Check that your dataset contains folders named s1–s40."
            )

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

        logger.info(
            "Found %d images across %d persons%s",
            total,
            len(self.person_dirs),
            f" (filter: {sorted(self.subject_filter)})"
            if self.subject_filter is not None else ""
        )

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
