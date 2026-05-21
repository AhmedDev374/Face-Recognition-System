"""
=============================================================
  pca_model.py
  Encapsulates the entire PCA / Eigenfaces pipeline originally
  implemented as standalone functions.  ALL mathematical logic
  is preserved exactly — only the structure is upgraded to OOP.

  Pipeline summary (Turk & Pentland, 1991):
      1. Build matrix A  (M × pixels)
      2. Compute mean face
      3. Subtract mean  → A_diff
      4. Compact covariance S = A_diff @ A_diff.T  (M × M)
      5. Eigendecomposition of S
      6. Map small eigenvectors → full image-space eigenfaces
      7. Project every training image onto eigenface subspace

  FIXES:
  - img_paths stores full absolute paths (fed from dataset_loader)
=============================================================
"""

import os
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── File names inside the model directory ─────────────────────
_FILES = {
    "eigenfaces" : "eigenfaces.npy",
    "avg_face"   : "average_face.npy",
    "features"   : "train_features.npy",
    "labels"     : "labels.npy",
    "img_paths"  : "image_paths.npy",
    "meta"       : "meta.npy",          # stores n_components, image_size, etc.
}


class PCAModel:
    """
    Trains, saves, loads, and queries a PCA / Eigenfaces model.

    Parameters
    ----------
    n_components      : int   – number of top eigenfaces to keep
    image_size        : tuple – (height, width)
    distance_threshold: float – Euclidean threshold for accept/reject
    model_dir         : str   – folder where .npy files are saved
    """

    def __init__(self,
                 n_components: int   = 40,
                 image_size: tuple   = (112, 92),
                 distance_threshold: float = 5000.0,
                 model_dir: str      = "pca_model"):

        self.n_components       = n_components
        self.image_size         = image_size
        self.distance_threshold = distance_threshold
        self.model_dir          = model_dir

        # Set after training / loading
        self.eigenfaces   : np.ndarray | None = None   # (n_components, H*W)
        self.average_face : np.ndarray | None = None   # (H*W,)
        self.features     : np.ndarray | None = None   # (n_components, M)
        self.labels       : list[str]         = []
        self.img_paths    : list[str]         = []
        self.is_trained   : bool              = False
        self.training_time: float             = 0.0
        self.training_accuracy: float         = 0.0

    # ══════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════
    def train(self, images: list, labels: list, img_paths: list,
              progress_callback=None) -> bool:
        """
        Full Eigenfaces training.  Preserves original algorithm verbatim.

        Parameters
        ----------
        images            : list of 1-D float64 arrays (flattened images)
        labels            : corresponding person names
        img_paths         : corresponding file paths
        progress_callback : callable(int pct, str msg) | None

        Returns
        -------
        True on success, False otherwise.
        """
        if not images:
            logger.error("No images provided for training.")
            return False

        start_time = time.time()
        M          = len(images)

        def _progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("[%3d%%] %s", pct, msg)

        _progress(5, f"Building image matrix  ({M} images) …")

        # ── STEP 1: Build matrix A  (M × pixel_count) ─────────
        A_matrix = np.array(images, dtype=np.float64)   # (M, H*W)

        # ── STEP 2: Compute average face ──────────────────────
        _progress(15, "Computing average face …")
        self.average_face = np.mean(A_matrix, axis=0)   # (H*W,)

        # ── STEP 3: Subtract mean ─────────────────────────────
        _progress(25, "Subtracting mean face …")
        A_diff = A_matrix - self.average_face            # (M, H*W)

        # ── STEP 4: Compact covariance S = A_diff @ A_diff.T ─
        _progress(40, "Computing covariance matrix (M×M trick) …")
        S = A_diff @ A_diff.T                            # (M, M)

        # ── STEP 5: Eigendecomposition ────────────────────────
        _progress(55, "Computing eigenvalues and eigenvectors …")
        eigenvalues, eigenvectors_small = np.linalg.eigh(S)

        # Sort descending
        sorted_idx         = np.argsort(eigenvalues)[::-1]
        eigenvalues        = eigenvalues[sorted_idx]
        eigenvectors_small = eigenvectors_small[:, sorted_idx]

        # ── STEP 6: Map small eigenvectors → image-space ──────
        n_keep = min(self.n_components, M)
        _progress(70, f"Extracting top {n_keep} eigenfaces …")

        eigenfaces_list = []
        for i in range(n_keep):
            ef   = A_diff.T @ eigenvectors_small[:, i]   # (H*W,)
            norm = np.linalg.norm(ef)
            if norm > 1e-10:
                ef = ef / norm
            eigenfaces_list.append(ef)

        self.eigenfaces = np.array(eigenfaces_list)      # (n_keep, H*W)

        # ── STEP 7: Project training images ───────────────────
        _progress(85, "Projecting training images onto eigenface space …")
        self.features  = self.eigenfaces @ A_diff.T      # (n_keep, M)
        self.labels    = list(labels)
        self.img_paths = list(img_paths)   # full absolute paths stored here
        self.is_trained = True

        self.training_time = time.time() - start_time

        # ── Compute leave-one-out training accuracy ────────────
        _progress(95, "Evaluating training accuracy …")
        self.training_accuracy = self._compute_training_accuracy()

        _progress(100, f"Training complete in {self.training_time:.2f}s  "
                       f"| Accuracy: {self.training_accuracy:.1f}%")
        return True

    # ──────────────────────────────────────────────────────────
    def _compute_training_accuracy(self) -> float:
        """
        Leave-one-out accuracy on the training set.
        For each training sample, find its nearest neighbour
        among the OTHER training samples and check if they share a label.
        """
        if self.features is None or len(self.labels) < 2:
            return 0.0

        M       = self.features.shape[1]   # number of training samples
        correct = 0

        for i in range(M):
            test_f = self.features[:, i]   # query vector
            # Distance to every training sample
            dists  = np.linalg.norm(self.features.T - test_f, axis=1)
            dists[i] = np.inf              # exclude self
            nearest  = np.argmin(dists)
            if self.labels[nearest] == self.labels[i]:
                correct += 1

        return (correct / M) * 100.0

    # ══════════════════════════════════════════════════════════
    #  RECOGNITION
    # ══════════════════════════════════════════════════════════
    def extract_features(self, img_flat: np.ndarray) -> np.ndarray:
        """
        Project a single pre-processed (flattened, float64) image
        onto the eigenface subspace.

        Returns
        -------
        feature_vector : (n_components,) array
        """
        if self.eigenfaces is None or self.average_face is None:
            raise RuntimeError("Model is not trained / loaded.")

        diff = img_flat - self.average_face             # (H*W,)
        return self.eigenfaces @ diff                   # (n_components,)

    def recognize(self, feature_vector: np.ndarray, top_n: int = 5
                  ) -> dict:
        """
        Nearest-neighbour search in eigenface space.

        Parameters
        ----------
        feature_vector : result of extract_features()
        top_n          : how many nearest matches to return

        Returns
        -------
        dict with keys:
            best_label, best_distance, confidence_pct,
            best_img_path, accepted,
            top_matches: list of (label, distance, img_path)
        """
        if self.features is None:
            raise RuntimeError("Model is not trained / loaded.")

        # Euclidean distance to every training sample
        all_distances = np.linalg.norm(
            self.features.T - feature_vector, axis=1   # (M,)
        )

        best_idx      = int(np.argmin(all_distances))
        best_distance = float(all_distances[best_idx])
        best_label    = self.labels[best_idx]
        best_img_path = self.img_paths[best_idx]

        # Confidence: maps [0, threshold] → [100%, 0%]
        confidence_pct = float(
            np.clip((1.0 - best_distance / self.distance_threshold) * 100.0,
                    0.0, 100.0)
        )
       # accepted = best_distance < self.distance_threshold
        accepted = confidence_pct >= 90.0
        # Top-N nearest matches
        sorted_idx = np.argsort(all_distances)
        top_matches = [
            (self.labels[i], float(all_distances[i]), self.img_paths[i])
            for i in sorted_idx[:top_n]
        ]

        return {
            "best_label"    : best_label,
            "best_distance" : best_distance,
            "confidence_pct": confidence_pct,
            "best_img_path" : best_img_path,
            "accepted"      : accepted,
            "top_matches"   : top_matches,
            "all_distances" : all_distances,
        }

    # ══════════════════════════════════════════════════════════
    #  PERSIST
    # ══════════════════════════════════════════════════════════
    def save(self) -> bool:
        """Save the trained model to self.model_dir."""
        if not self.is_trained:
            logger.error("Nothing to save — model not trained.")
            return False

        os.makedirs(self.model_dir, exist_ok=True)
        p = lambda name: os.path.join(self.model_dir, _FILES[name])

        np.save(p("eigenfaces"), self.eigenfaces)
        np.save(p("avg_face"),   self.average_face)
        np.save(p("features"),   self.features)
        np.save(p("labels"),     np.array(self.labels))
        np.save(p("img_paths"),  np.array(self.img_paths))
        np.save(p("meta"),       np.array([
            self.n_components,
            self.image_size[0],
            self.image_size[1],
            self.distance_threshold,
            self.training_time,
            self.training_accuracy,
        ]))

        logger.info("Model saved to '%s/'", self.model_dir)
        return True

    def load(self) -> bool:
        """Load model from self.model_dir.  Returns True on success."""
        p        = lambda name: os.path.join(self.model_dir, _FILES[name])
        required = [p(k) for k in ("eigenfaces", "avg_face",
                                    "features", "labels", "img_paths")]

        if not all(os.path.exists(f) for f in required):
            logger.warning("Saved model not found in '%s'.", self.model_dir)
            return False

        self.eigenfaces   = np.load(p("eigenfaces"))
        self.average_face = np.load(p("avg_face"))
        self.features     = np.load(p("features"))
        self.labels       = list(np.load(p("labels")))
        self.img_paths    = list(np.load(p("img_paths")))

        if os.path.exists(p("meta")):
            meta = np.load(p("meta"))
            self.n_components       = int(meta[0])
            self.image_size         = (int(meta[1]), int(meta[2]))
            self.distance_threshold = float(meta[3])
            self.training_time      = float(meta[4])
            self.training_accuracy  = float(meta[5])

        self.is_trained = True
        logger.info("Model loaded — %d eigenfaces, %d training samples",
                    self.eigenfaces.shape[0], len(self.labels))
        return True

    # ══════════════════════════════════════════════════════════
    #  PROPERTIES
    # ══════════════════════════════════════════════════════════
    @property
    def num_persons(self) -> int:
        return len(set(self.labels))

    @property
    def num_training_images(self) -> int:
        return len(self.labels)

    @property
    def n_eigenfaces(self) -> int:
        return 0 if self.eigenfaces is None else self.eigenfaces.shape[0]
