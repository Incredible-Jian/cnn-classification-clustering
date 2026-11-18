"""K-Means clustering on CNN features (scikit-learn implementations only)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler, normalize

from .feature_extraction import load_feature_bundle


def preprocess_features(features: np.ndarray, mode: str = "l2") -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if mode == "raw":
        return x
    if mode == "l2":
        return normalize(x, norm="l2", axis=1).astype(np.float32)
    if mode == "standard":
        return StandardScaler().fit_transform(x).astype(np.float32)
    if mode == "l2_standard":
        z = normalize(x, norm="l2", axis=1)
        return StandardScaler().fit_transform(z).astype(np.float32)
    raise ValueError(mode)


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Purity = sum over clusters of the majority-class count / N."""
    total = 0
    majorities = []
    for c in np.unique(y_pred):
        mask = y_pred == c
        counts = np.bincount(y_true[mask])
        total += int(counts.max())
        majorities.append(int(counts.argmax()))
    return float(total / len(y_true)), np.asarray(majorities)


def matched_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Hungarian one-to-one mapping between clusters and classes (diagnostic)."""
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(p), int(t)] += 1
    row, col = linear_sum_assignment(-cm)
    return float(cm[row, col].sum() / len(y_true))


def run_minibatch_kmeans(features: np.ndarray, k: int, seed: int = 42,
                         batch_size: int = 4096, n_init: int = 10,
                         max_iter: int = 500) -> tuple[np.ndarray, MiniBatchKMeans, float]:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        reassignment_ratio=0.01,
        verbose=0,
    )
    t0 = time.time()
    labels = model.fit_predict(features)
    elapsed = time.time() - t0
    return labels.astype(np.int64), model, elapsed


def _sample_for_visualization(y_true: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Class-balanced sample so rare classes are visible in t-SNE."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    per_class = max(1, n // len(classes))
    chosen = []
    for c in classes:
        idx = np.flatnonzero(y_true == c)
        take = min(per_class, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(chosen)
    if len(out) > n:
        out = rng.choice(out, size=n, replace=False)
    rng.shuffle(out)
    return out.astype(np.int64)


def reduce_pca(features: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    return PCA(n_components=n_components, random_state=seed).fit_transform(features)


def reduce_tsne(features: np.ndarray, seed: int = 42, perplexity: float = 30.0,
                n_iter: int = 1000) -> np.ndarray:
    """t-SNE on 50 PCA components (standard deep-feature practice)."""
    n = len(features)
    n_pca = min(50, features.shape[1], n - 1)
    z = PCA(n_components=n_pca, random_state=seed).fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (n - 1) / 3.0)),
        learning_rate="auto",
        init="pca",
        max_iter=n_iter,
        random_state=seed,
        metric="euclidean",
    )
    return tsne.fit_transform(z)


def cluster_distances(features: np.ndarray, model: MiniBatchKMeans,
                      chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the assigned centroid (in the preprocessed feature space)."""
    dists = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk):
        end = min(start + chunk, len(features))
        d = model.transform(features[start:end])
        dists[start:end] = d[np.arange(end - start), model.labels_[start:end]]
    return dists, np.asarray(model.labels_, dtype=np.int64)


def run_clustering(
    dataset: str,
    feature_path: Path,
    output_metrics: Path,
    output_features: Path,
    k: int,
    feature_source: str,
    preprocessing: str = "l2",
    seed: int = 42,
    vis_sample_size: int = 5000,
    run_tsne: bool = True,
    class_names: Optional[list[str]] = None,
) -> dict:
    features, y_true, meta = load_feature_bundle(Path(feature_path))
    x = preprocess_features(features, preprocessing)

    print(f"\n=== K-Means: {dataset} / {feature_source} / K={k} ===")
    print(f"features={features.shape} preprocessing={preprocessing}")
    y_pred, model, elapsed = run_minibatch_kmeans(x, k, seed=seed)
    print(f"MiniBatchKMeans finished in {elapsed:.1f}s, inertia={model.inertia_:.2f}")

    sample_idx = _sample_for_visualization(y_true, vis_sample_size, seed)
    sil = silhouette_score(x[sample_idx], y_pred[sample_idx], metric="euclidean",
                           random_state=seed)
    purity, majorities = cluster_purity(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    matched = matched_accuracy(y_true, y_pred, k)

    sizes = np.bincount(y_pred, minlength=k)
    metrics = {
        "dataset": dataset,
        "feature_source": feature_source,
        "preprocessing": preprocessing,
        "k": int(k),
        "n_samples": int(len(y_true)),
        "feature_dim": int(features.shape[1]),
        "algorithm": "MiniBatchKMeans (scikit-learn)",
        "batch_size": 4096,
        "n_init": 10,
        "max_iter": 500,
        "seed": seed,
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": float(sil),
        "purity": float(purity),
        "matched_accuracy": float(matched),
        "inertia": float(model.inertia_),
        "elapsed_sec": round(elapsed, 2),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_max": int(sizes.max()),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_std": float(sizes.std()),
        "empty_clusters": int((sizes == 0).sum()),
        "vis_sample_size": int(len(sample_idx)),
        "tsne_computed": bool(run_tsne),
    }
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_features.mkdir(parents=True, exist_ok=True)
    np.save(output_features / f"{dataset}_{feature_source}_cluster_labels.npy", y_pred)
    np.save(output_features / f"{dataset}_{feature_source}_cluster_centers.npy",
            model.cluster_centers_.astype(np.float32))
    np.save(output_features / f"{dataset}_{feature_source}_vis_sample_idx.npy", sample_idx)
    np.save(output_features / f"{dataset}_{feature_source}_pca2d.npy",
            reduce_pca(x[sample_idx], 2, seed).astype(np.float32))
    if run_tsne:
        print("  computing t-SNE on 50 PCA components ...", flush=True)
        np.save(output_features / f"{dataset}_{feature_source}_tsne2d.npy",
                reduce_tsne(x[sample_idx], seed).astype(np.float32))
    np.save(output_features / f"{dataset}_{feature_source}_vis_labels.npy", y_true[sample_idx])
    np.save(output_features / f"{dataset}_{feature_source}_vis_clusters.npy", y_pred[sample_idx])

    # Full assignment table for inspection / the "聚类结果" deliverable.
    dists, _ = cluster_distances(x, model)
    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "true_label": y_true,
        "cluster": y_pred,
        "distance_to_centroid": dists,
    })
    if class_names is not None:
        assignments["true_class"] = [class_names[i] for i in y_true]
    assignments.to_csv(output_features / f"{dataset}_{feature_source}_cluster_assignments.csv",
                       index=False, encoding="utf-8-sig")

    metrics["sample_indices"] = sample_idx.tolist()
    return metrics
