"""
=============================================================
  pca_model.py  (FIXED)
  Encapsulates the entire PCA / Eigenfaces pipeline.

  FIXES:
  - recognize() now returns structured top_matches with
    person, image, img_path, distance, confidence per match
  - img_paths stores full absolute paths (fed from dataset_loader)
=============================================================
"""

import os
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

_FILES = {
    "eigenfaces" : "eigenfaces.npy",
    "avg_face"   : "average_face.npy",
    "features"   : "train_features.npy",
    "labels"     : "labels.npy",
    "img_paths"  : "image_paths.npy",
    "meta"       : "meta.npy",
}


class PCAModel:
    def __init__(self,
                 n_components: int   = 40,
                 image_size: tuple   = (112, 92),
                 distance_threshold: float = 5000.0,
                 model_dir: str      = "pca_model"):

        self.n_components       = n_components
        self.image_size         = image_size
        self.distance_threshold = distance_threshold
        self.model_dir          = model_dir

        self.eigenfaces   = None
        self.average_face = None
        self.features     = None
        self.labels       = []
        self.img_paths    = []
        self.is_trained   = False
        self.training_time = 0.0
        self.training_accuracy = 0.0

    def train(self, images, labels, img_paths, progress_callback=None):
        if not images:
            logger.error("No images provided for training.")
            return False

        start_time = time.time()
        M = len(images)

        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("[%3d%%] %s", pct, msg)

        _progress(5,  f"Building image matrix ({M} images) …")
        A_matrix = np.array(images, dtype=np.float64)

        _progress(15, "Computing average face …")
        self.average_face = np.mean(A_matrix, axis=0)

        _progress(25, "Subtracting mean face …")
        A_diff = A_matrix - self.average_face

        _progress(40, "Computing covariance matrix (M×M trick) …")
        S = A_diff @ A_diff.T

        _progress(55, "Computing eigenvalues and eigenvectors …")
        eigenvalues, eigenvectors_small = np.linalg.eigh(S)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors_small = eigenvectors_small[:, sorted_idx]

        n_keep = min(self.n_components, M)
        _progress(70, f"Extracting top {n_keep} eigenfaces …")

        eigenfaces_list = []
        for i in range(n_keep):
            ef = A_diff.T @ eigenvectors_small[:, i]
            norm = np.linalg.norm(ef)
            if norm > 1e-10:
                ef = ef / norm
            eigenfaces_list.append(ef)

        self.eigenfaces = np.array(eigenfaces_list)

        _progress(85, "Projecting training images onto eigenface space …")
        self.features  = self.eigenfaces @ A_diff.T
        self.labels    = list(labels)
        self.img_paths = list(img_paths)   # full absolute paths stored here
        self.is_trained = True

        self.training_time = time.time() - start_time

        _progress(95, "Evaluating training accuracy …")
        self.training_accuracy = self._compute_training_accuracy()

        _progress(100, f"Training complete in {self.training_time:.2f}s "
                       f"| Accuracy: {self.training_accuracy:.1f}%")
        return True

    def _compute_training_accuracy(self):
        if self.features is None or len(self.labels) < 2:
            return 0.0

        M = self.features.shape[1]
        correct = 0

        for i in range(M):
            test_f = self.features[:, i]
            dists  = np.linalg.norm(self.features.T - test_f, axis=1)
            dists[i] = np.inf
            nearest = np.argmin(dists)
            if self.labels[nearest] == self.labels[i]:
                correct += 1

        return (correct / M) * 100.0

    def extract_features(self, img_flat):
        if self.eigenfaces is None or self.average_face is None:
            raise RuntimeError("Model is not trained / loaded.")
        diff = img_flat - self.average_face
        return self.eigenfaces @ diff

    def recognize(self, feature_vector, top_n=5):
        """
        Nearest-neighbour search. Returns structured top_matches
        with person/image/distance/confidence for each ranked result.
        """
        if self.features is None:
            raise RuntimeError("Model is not trained / loaded.")

        all_distances = np.linalg.norm(
            self.features.T - feature_vector, axis=1
        )

        best_idx      = int(np.argmin(all_distances))
        best_distance = float(all_distances[best_idx])
        best_label    = self.labels[best_idx]
        best_img_path = self.img_paths[best_idx]

        # Parse person folder and image filename from path
        best_person     = os.path.basename(os.path.dirname(best_img_path))
        best_image_file = os.path.basename(best_img_path)

        confidence_pct = float(
            np.clip((1.0 - best_distance / self.distance_threshold) * 100.0,
                    0.0, 100.0)
        )
        accepted = best_distance < self.distance_threshold

        # Build top_n structured matches — each is a distinct image
        sorted_idx = np.argsort(all_distances)
        top_matches = []
        for i in sorted_idx[:top_n]:
            dist    = float(all_distances[i])
            conf    = float(np.clip(
                (1.0 - dist / self.distance_threshold) * 100.0, 0.0, 100.0))
            ipath   = self.img_paths[i]
            person  = os.path.basename(os.path.dirname(ipath))
            imgfile = os.path.basename(ipath)
            top_matches.append({
                "person"    : person,
                "image"     : imgfile,
                "img_path"  : ipath,
                "distance"  : dist,
                "confidence": conf,
                "label"     : self.labels[i],  # legacy compat
            })

        return {
            "best_label"     : best_label,
            "best_distance"  : best_distance,
            "confidence_pct" : confidence_pct,
            "best_img_path"  : best_img_path,
            "best_person"    : best_person,
            "best_image_file": best_image_file,
            "accepted"       : accepted,
            "top_matches"    : top_matches,
            "all_distances"  : all_distances,
        }

    def save(self):
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

    def load(self):
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

    @property
    def num_persons(self):
        return len(set(self.labels))

    @property
    def num_training_images(self):
        return len(self.labels)

    @property
    def n_eigenfaces(self):
        return 0 if self.eigenfaces is None else self.eigenfaces.shape[0]
