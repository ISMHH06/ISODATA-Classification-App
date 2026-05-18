from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


class ISODATA:
    """ISODATA clustering with split/merge logic."""

    def __init__(
        self,
        k_init: int = 5,
        max_iter: int = 20,
        split_std_threshold: float = 0.85,
        merge_dist_threshold: float = 2.0,
        min_cluster_size: int = 150,
        max_clusters: int | None = 8,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.k_init = k_init
        self.max_iter = max_iter
        self.split_std_threshold = split_std_threshold
        self.merge_dist_threshold = merge_dist_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.verbose = verbose
        self.history_: list[dict[str, int]] = []

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return pairwise_distances_argmin(X, centroids)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        new_centroids = []
        for k in range(len(centroids)):
            members = X[labels == k]
            new_centroids.append(members.mean(axis=0) if len(members) >= 1 else centroids[k])
        return np.array(new_centroids)

    def _split(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        new_centroids = []
        split_count = 0

        for i, centroid in enumerate(centroids):
            members = X[labels == i]

            if len(members) < self.min_cluster_size * 2:
                new_centroids.append(centroid)
                continue

            if self.max_clusters is not None and (len(new_centroids) + split_count + 1) >= self.max_clusters:
                new_centroids.append(centroid)
                continue

            std_per_dim = np.std(members, axis=0)
            mean_std = np.mean(std_per_dim)
            split_axis = np.argmax(std_per_dim)

            if mean_std > self.split_std_threshold:
                delta = np.zeros_like(centroid)
                delta[split_axis] = std_per_dim[split_axis] * 0.5
                new_centroids.append(centroid + delta)
                new_centroids.append(centroid - delta)
                split_count += 1
            else:
                new_centroids.append(centroid)

        return np.array(new_centroids), split_count

    def _merge(self, centroids: np.ndarray) -> tuple[np.ndarray, int]:
        n_clusters = len(centroids)
        pairs = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < self.merge_dist_threshold:
                    pairs.append((dist, i, j))
        pairs.sort()

        used: set[int] = set()
        merged = []
        merge_count = 0

        for _, i, j in pairs:
            if i in used or j in used:
                continue
            merged.append((centroids[i] + centroids[j]) / 2.0)
            used.add(i)
            used.add(j)
            merge_count += 1

        for i in range(n_clusters):
            if i not in used:
                merged.append(centroids[i])

        return np.array(merged), merge_count

    def _eliminate_small_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        sizes = np.array([np.sum(labels == k) for k in range(len(centroids))])
        valid_mask = sizes >= self.min_cluster_size
        n_eliminated = int((~valid_mask).sum())

        if n_eliminated == 0:
            return centroids, labels, 0

        valid_centroids = centroids[valid_mask]
        if len(valid_centroids) == 0:
            top2 = np.argsort(sizes)[-2:]
            valid_centroids = centroids[top2]

        new_labels = self._assign(X, valid_centroids)
        return valid_centroids, new_labels, n_eliminated

    def fit(self, X: np.ndarray) -> "ISODATA":
        X = np.array(X)
        np.random.seed(self.random_state)

        km_init = KMeans(n_clusters=self.k_init, random_state=self.random_state, n_init=10)
        labels = km_init.fit_predict(X)
        centroids = km_init.cluster_centers_.copy()

        if self.verbose:
            print(f"Initial K-Means: {len(centroids)} clusters")
            print(f"split_std_threshold = {self.split_std_threshold}")
            print(f"merge_dist_threshold = {self.merge_dist_threshold}")
            print(f"min_cluster_size = {self.min_cluster_size}")
            print(f"max_clusters = {self.max_clusters}")
            print()
            print(f"{'Iter':>4} | {'#Clusters':>10} | {'Splits':>7} | {'Merges':>7} | {'Eliminated':>10}")
            print("-" * 52)

        prev_labels = None

        for iteration in range(self.max_iter):
            labels = self._assign(X, centroids)
            centroids = self._update_centroids(X, labels, centroids)

            centroids, n_splits = self._split(X, labels, centroids)

            labels = self._assign(X, centroids)
            centroids = self._update_centroids(X, labels, centroids)

            centroids, n_merges = self._merge(centroids)

            labels = self._assign(X, centroids)

            centroids, labels, n_eliminated = self._eliminate_small_clusters(X, labels, centroids)

            n_clusters = len(centroids)
            if self.max_clusters is not None and len(centroids) > self.max_clusters:
                if self.verbose:
                    print(
                        f"Enforcing max_clusters={self.max_clusters} (current={len(centroids)})"
                        " - merging closest pairs"
                    )
                while len(centroids) > self.max_clusters:
                    n = len(centroids)
                    min_d = float("inf")
                    pair = (0, 1)
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = np.linalg.norm(centroids[i] - centroids[j])
                            if dist < min_d:
                                min_d = dist
                                pair = (i, j)
                    i, j = pair
                    merged = (centroids[i] + centroids[j]) / 2.0
                    new_centroids = [centroids[k] for k in range(n) if k not in pair]
                    new_centroids.append(merged)
                    centroids = np.array(new_centroids)
                labels = self._assign(X, centroids)
                centroids = self._update_centroids(X, labels, centroids)
                n_clusters = len(centroids)

            self.history_.append(
                {
                    "iteration": iteration + 1,
                    "n_clusters": n_clusters,
                    "n_splits": n_splits,
                    "n_merges": n_merges,
                    "n_eliminated": n_eliminated,
                }
            )

            if self.verbose:
                print(
                    f"{iteration + 1:>4} | {n_clusters:>10} | {n_splits:>7} | {n_merges:>7} |"
                    f" {n_eliminated:>10}"
                )

            if iteration > 1 and n_splits == 0 and n_merges == 0 and n_eliminated == 0:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            if prev_labels is not None and len(prev_labels) == len(labels):
                from sklearn.metrics import adjusted_rand_score

                ari = adjusted_rand_score(prev_labels, labels)
                if ari >= 0.999:
                    if self.verbose:
                        print(f"Label convergence (ARI={ari:.4f}) at iteration {iteration + 1}")
                    break

            prev_labels = labels.copy()

        self.centroids_ = centroids
        self.labels_ = self._assign(X, centroids)
        self.n_clusters_ = len(centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(np.array(X), self.centroids_)
