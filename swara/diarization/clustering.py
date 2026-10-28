"""Clustering algorithms for speaker diarization (Offline AHC and Online Centroid Clustering)."""

from typing import List, Optional
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scidist

from swara.utils.logger import get_logger

logger = get_logger("swara.diarization.clustering")


class AgglomerativeClustering:
    """Agglomerative Hierarchical Clustering (AHC) using average linkage on cosine distance."""

    def __init__(
        self,
        threshold: float = 0.65,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> None:
        self.threshold = threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Clusters L2-normalized embeddings into discrete integer labels.

        Args:
            embeddings: np.ndarray of shape (N, D).

        Returns:
            np.ndarray of integer cluster IDs of shape (N,).
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return np.array([], dtype=int)
        if n_samples == 1:
            return np.array([0], dtype=int)

        # Compute pairwise cosine distance: d = 1 - dot(u, v)
        # Numerical safeguard to ensure distance >= 0
        cosine_sim = np.dot(embeddings, embeddings.T)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        dist_matrix = 1.0 - cosine_sim
        np.fill_diagonal(dist_matrix, 0.0)

        # Convert to condensed distance matrix for scipy
        condensed_dist = scidist.squareform(dist_matrix, checks=False)

        # Hierarchical average linkage
        linkage_matrix = sch.linkage(condensed_dist, method="average")

        # Determine cutoff
        criterion = "distance"
        t = self.threshold

        if self.min_speakers is not None and self.max_speakers is not None and self.min_speakers == self.max_speakers:
            # Fixed number of clusters
            criterion = "maxclust"
            t = min(n_samples, self.min_speakers)

        labels = sch.fcluster(linkage_matrix, t=t, criterion=criterion)

        # Post-adjust if max_speakers / min_speakers bounds are given
        num_clusters = len(np.unique(labels))
        if self.max_speakers is not None and num_clusters > self.max_speakers:
            labels = sch.fcluster(linkage_matrix, t=min(n_samples, self.max_speakers), criterion="maxclust")
        elif self.min_speakers is not None and num_clusters < self.min_speakers and n_samples >= self.min_speakers:
            labels = sch.fcluster(linkage_matrix, t=min(n_samples, self.min_speakers), criterion="maxclust")

        # Re-index labels from 0 to K-1 in order of appearance
        unique_labels = []
        for lbl in labels:
            if lbl not in unique_labels:
                unique_labels.append(lbl)

        label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[l] for l in labels], dtype=int)
        return mapped_labels


class OnlineCentroidClustering:
    """Streaming online clustering using running centroid tracking and cosine distance."""

    def __init__(self, match_threshold: float = 0.65, decay: float = 0.9) -> None:
        self.match_threshold = match_threshold
        self.decay = decay
        self.centroids: List[np.ndarray] = []
        self.cluster_counts: List[int] = []

    def reset(self) -> None:
        self.centroids.clear()
        self.cluster_counts.clear()

    def step(self, embedding: np.ndarray) -> int:
        """Assigns an embedding to an existing or new speaker centroid in streaming mode."""
        norm = np.linalg.norm(embedding)
        unit_emb = embedding / norm if norm > 1e-12 else embedding

        if not self.centroids:
            self.centroids.append(unit_emb.copy())
            self.cluster_counts.append(1)
            return 0

        # Compute similarity to all active centroids
        sims = [float(np.dot(c, unit_emb)) for c in self.centroids]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.match_threshold:
            # Update centroid with exponential moving average
            c = self.centroids[best_idx]
            updated = self.decay * c + (1.0 - self.decay) * unit_emb
            up_norm = np.linalg.norm(updated)
            self.centroids[best_idx] = updated / up_norm if up_norm > 1e-12 else updated
            self.cluster_counts[best_idx] += 1
            return best_idx
        else:
            # Spawn new speaker
            new_id = len(self.centroids)
            self.centroids.append(unit_emb.copy())
            self.cluster_counts.append(1)
            return new_id
