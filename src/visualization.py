"""Unified, publication-quality figures for the report, notebook and slides."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

FIG_DPI = 200
PALETTE = {
    "cifar10": "#2E6F9E",
    "cifar100": "#D9772B",
    "accent": "#1F9D8B",
    "grey": "#6B7280",
    "red": "#C0392B",
    "green": "#2E8B57",
}
DATASET_COLORS = [PALETTE["cifar10"], PALETTE["cifar100"]]


def setup_style() -> None:
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4B5563",
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": "#D1D5DB",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })


def save_fig(fig: plt.Figure, path: Path, dpi: int = FIG_DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _panel(ax, letter: str) -> None:
    ax.text(-0.09, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left", color="#111827")


def _colors(n: int) -> np.ndarray:
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return np.array([cmap(i % 10) for i in range(n)])
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return np.array([cmap(i % 20) for i in range(n)])
    return plt.get_cmap("hsv")(np.linspace(0, 1, n, endpoint=False))


# ---------------------------------------------------------------------------
# Dataset exploration
# ---------------------------------------------------------------------------
def plot_dataset_samples(images: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
                         class_names_zh: Sequence[str], path: Path,
                         dataset_name: str, per_class: int = 5,
                         max_classes: int = 10, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    classes = np.arange(len(class_names))
    if len(classes) > max_classes:
        classes = np.sort(rng.choice(classes, size=max_classes, replace=False))
    n_rows, n_cols = len(classes), per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.55, n_rows * 1.72))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        take = rng.choice(idx, size=n_cols, replace=False)
        for j, k in enumerate(take):
            ax = axes[r, j]
            ax.imshow(images[k])
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{class_names_zh[c]}\n{class_names[c]}", fontsize=9,
                             loc="left", pad=4)
    fig.suptitle(f"{dataset_name} 随机样本（每行一个类别）", fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(h_pad=0.9, w_pad=0.2)
    return save_fig(fig, path)


def plot_preprocessing(images: np.ndarray, labels: np.ndarray,
                       class_names_zh: Sequence[str], transform, path: Path,
                       n: int = 6, seed: int = 7) -> Path:
    """Show original 32×32 images and their augmented 96×96 versions."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.75, 4.0))
    for j, i in enumerate(idx):
        axes[0, j].imshow(images[i])
        axes[0, j].set_title(class_names_zh[labels[i]], fontsize=9)
        axes[0, j].axis("off")
        aug = transform(Image.fromarray(images[i])).numpy().transpose(1, 2, 0)
        aug = np.clip(aug * np.asarray((0.229, 0.224, 0.225)) + np.asarray((0.485, 0.456, 0.406)), 0, 1)
        axes[1, j].imshow(aug)
        axes[1, j].axis("off")
    fig.text(0.005, 0.72, "原始\n32×32", fontsize=10, va="center", ha="left")
    fig.text(0.005, 0.28, "增强后\n96×96", fontsize=10, va="center", ha="left")
    fig.suptitle("预处理示例：随机裁剪缩放、水平翻转、颜色抖动与 ImageNet 归一化",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_class_distribution(info10: dict, info100: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, info, color, letter in [
        (axes[0], info10, PALETTE["cifar10"], "A"),
        (axes[1], info100, PALETTE["cifar100"], "B"),
    ]:
        counts = info["class_counts_test"]
        x = np.arange(len(counts))
        ax.bar(x, counts, color=color, alpha=0.85, width=1.0 if len(counts) > 20 else 0.75)
        ax.set_title(f"{info['display_name']} 测试集类别分布（{len(counts)} 类）")
        ax.set_xlabel("类别编号")
        ax.set_ylabel("样本数")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if len(counts) <= 10:
            ax.set_xticks(x)
            ax.set_xticklabels(info["class_names_zh"], rotation=35, ha="right")
        else:
            ax.set_xlim(-1, len(counts))
        ax.text(0.98, 0.95, f"总计 {sum(counts):,}\n每类 {counts[0]:,} 张", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))
        _panel(ax, letter)
    fig.suptitle("CIFAR-10 与 CIFAR-100 的类别数量与样本分布", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def plot_training_curves(log_history: pd.DataFrame, dataset_name: str, path: Path) -> Path:
    df = log_history.copy()
    train = df[df.get("loss").notna()] if "loss" in df else pd.DataFrame()
    evals = df[df.get("eval_accuracy").notna()] if "eval_accuracy" in df else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    ax = axes[0]
    if len(train):
        ax.plot(train["epoch"], train["loss"], color=PALETTE["cifar10"], lw=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("训练损失")
        ax.set_title("训练损失曲线")
        ax.grid(True)
        ax.text(0.97, 0.95, f"final = {train['loss'].iloc[-1]:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
    _panel(ax, "A")

    ax = axes[1]
    if len(evals):
        ax.plot(evals["epoch"], evals["eval_accuracy"] * 100, "o-", color=PALETTE["cifar100"],
                lw=2, ms=5)
        best_i = int(np.argmax(evals["eval_accuracy"].values))
        ax.scatter([evals["epoch"].iloc[best_i]], [evals["eval_accuracy"].iloc[best_i] * 100],
                   s=90, facecolors="none", edgecolors=PALETTE["red"], zorder=5, lw=1.6)
        ax.annotate(f"best {evals['eval_accuracy'].iloc[best_i]*100:.2f}%",
                    (evals["epoch"].iloc[best_i], evals["eval_accuracy"].iloc[best_i] * 100),
                    textcoords="offset points", xytext=(8, -16), fontsize=9, color=PALETTE["red"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("验证准确率 (%)")
        ax.set_title("验证准确率曲线")
        ax.grid(True)
    _panel(ax, "B")

    ax = axes[2]
    if len(train) and "learning_rate" in train:
        ax.plot(train["epoch"], train["learning_rate"], color=PALETTE["accent"], lw=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("学习率")
        ax.set_title("余弦学习率调度")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True)
    _panel(ax, "C")
    fig.suptitle(f"{dataset_name} 分类训练过程（Hugging Face Trainer）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: Path,
                          dataset_name: str, normalize: bool = True,
                          class_names_zh: Optional[Sequence[str]] = None) -> Path:
    n = len(class_names)
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm_plot = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_plot = cm
    fig, ax = plt.subplots(figsize=(9.5, 8) if n <= 10 else (13, 11.5))
    if n <= 10:
        im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else cm_plot.max())
    else:
        im = ax.imshow(cm_plot, cmap="Blues", norm=LogNorm(vmin=max(1e-4, cm_plot[cm_plot > 0].min()),
                                                           vmax=max(cm_plot.max(), 1e-3)))
    ax.set_title(f"{dataset_name} 测试集混淆矩阵" + ("（按真实类归一化）" if normalize else ""),
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    if n <= 10:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(n):
            for j in range(n):
                val = cm_plot[i, j]
                color = "white" if val > (0.55 if normalize else cm_plot.max() * 0.55) else "#111827"
                text = f"{val:.2f}" if normalize else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    else:
        ticks = list(range(0, n, 5))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([class_names[i] for i in ticks], rotation=90, fontsize=5)
        ax.set_yticklabels([class_names[i] for i in ticks], fontsize=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("行归一化比例" if normalize else "样本数")
    ax.set_aspect("equal")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_top_confusions(pairs: pd.DataFrame, class_names_zh: Sequence[str], path: Path,
                        dataset_name: str, top_n: int = 15) -> Path:
    df = pairs.head(top_n).iloc[::-1]
    labels = [f"{r.true_class_zh} → {r.pred_class_zh}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.barh(labels, df["count"], color=PALETTE["cifar100"], alpha=0.9)
    for y, (cnt, rate) in enumerate(zip(df["count"], df["rate"])):
        ax.text(cnt + max(df["count"]) * 0.01, y, f"{int(cnt)} ({rate*100:.1f}%)",
                va="center", fontsize=8)
    ax.set_xlabel("被误分类的样本数")
    ax.set_title(f"{dataset_name} 最容易混淆的类别对（Top {top_n}）",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_per_class_accuracy(report: pd.DataFrame, path: Path, dataset_name: str,
                            top_n: int = 15) -> Path:
    df = report.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.2 if len(df) <= 20 else 7.2))
    if len(df) > 20:
        show = pd.concat([df.head(top_n), df.tail(top_n)])
        labels = [f"{r.class_zh} ({r.class_zh})" for r in show.itertuples()]
        labels = [f"{r.class_zh}" for r in show.itertuples()]
        colors = [PALETTE["green"]] * top_n + [PALETTE["red"]] * top_n
        y = np.arange(len(show))
        ax.barh(y, show["accuracy"] * 100, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axhline(top_n - 0.5, color="#9CA3AF", lw=1, ls="--")
        ax.text(0.99, 0.01, f"共 {len(df)} 类；上 {top_n} 名（绿）与下 {top_n} 名（红）",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#4B5563")
    else:
        x = np.arange(len(df))
        colors = _colors(len(df))
        ax.bar(x, df["accuracy"] * 100, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(df["class_zh"], rotation=30, ha="right")
        ax.axhline(100 / len(df), color=PALETTE["red"], ls="--", lw=1.2,
                   label=f"随机水平 {100/len(df):.1f}%")
        ax.legend()
    ax.set_xlabel("类别" if len(df) <= 20 else "准确率 (%)")
    ax.set_ylabel("准确率 (%)" if len(df) <= 20 else "类别")
    ax.set_title(f"{dataset_name} 各类别测试准确率", fontsize=14, fontweight="bold")
    ax.grid(axis="x" if len(df) > 20 else "y")
    ax.set_axisbelow(True)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Clustering visualisation
# ---------------------------------------------------------------------------
def plot_pca_tsne(pca: np.ndarray, tsne: np.ndarray, y_true: np.ndarray,
                  y_pred: np.ndarray, class_names: Sequence[str], path: Path,
                  dataset_name: str, k: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.0))
    true_colors = _colors(len(class_names))
    cluster_colors = _colors(k)

    panels = [
        (axes[0, 0], pca, y_true, true_colors, "PCA · 真实类别", "A"),
        (axes[0, 1], pca, y_pred, cluster_colors, f"PCA · K-Means 聚类（K={k}）", "B"),
        (axes[1, 0], tsne, y_true, true_colors, "t-SNE · 真实类别", "C"),
        (axes[1, 1], tsne, y_pred, cluster_colors, f"t-SNE · K-Means 聚类（K={k}）", "D"),
    ]
    for ax, coords, labels, colors, title, letter in panels:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors[labels], s=5 if k > 20 else 8,
                   alpha=0.65, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("维度 1")
        ax.set_ylabel("维度 2")
        ax.grid(True, alpha=0.35)
        _panel(ax, letter)
    # Legends only when readable (CIFAR-10 true labels; cluster IDs <= 20).
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=true_colors[i],
                      markersize=7, label=class_names[i]) for i in range(len(class_names))]
    if len(class_names) <= 10:
        axes[0, 0].legend(handles=handles, loc="best", ncol=2, fontsize=8)
        handles_c = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cluster_colors[i],
                            markersize=7, label=f"C{i}") for i in range(k)]
        axes[0, 1].legend(handles=handles_c, loc="best", ncol=2, fontsize=8)
    else:
        axes[0, 0].text(0.02, 0.98, f"{len(class_names)} 个细类，颜色=类别 ID",
                        transform=axes[0, 0].transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="none"))
        axes[0, 1].text(0.02, 0.98, f"{k} 个簇，颜色=簇 ID",
                        transform=axes[0, 1].transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="none"))
    fig.suptitle(f"{dataset_name} CNN 特征的 PCA 与 t-SNE 二维可视化", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_cluster_gallery(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                         distances: np.ndarray, class_names: Sequence[str],
                         class_names_zh: Sequence[str], path: Path, dataset_name: str,
                         k: int, max_clusters: int = 10, n_images: int = 6) -> Path:
    sizes = np.bincount(y_pred, minlength=k)
    clusters = np.argsort(-sizes)[:max_clusters]
    n_rows, n_cols = len(clusters), n_images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.55, n_rows * 1.85))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(clusters):
        mask = np.flatnonzero(y_pred == c)
        if len(mask) == 0:
            continue
        order = mask[np.argsort(distances[mask])[:n_cols]]
        counts = np.bincount(y_true[mask], minlength=len(class_names))
        majority = int(counts.argmax())
        purity = counts[majority] / len(mask)
        for j in range(n_cols):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(order):
                idx = order[j]
                ax.imshow(images[idx])
                if y_true[idx] != majority:
                    h, w = images[idx].shape[:2]
                    ax.add_patch(patches.Rectangle((0, 0), w, h, fill=False,
                                                   edgecolor=PALETTE["red"], lw=2.2))
        title = (f"簇 {c} | n={len(mask)} | 主类 {class_names_zh[majority]} "
                 f"({class_names[majority]}) | 纯度 {purity:.2f}")
        axes[r, 0].set_title(title, fontsize=8.5, loc="left", pad=3)
    fig.suptitle(f"{dataset_name} 聚类样本展示（每簇按到质心距离取最近样本）",
                 fontsize=14, fontweight="bold", y=1.002)
    fig.text(0.01, -0.01, "红框：该样本真实类别不同于簇主类", fontsize=8, color=PALETTE["red"])
    fig.tight_layout(h_pad=0.8, w_pad=0.2)
    return save_fig(fig, path)


def plot_comparison_tsne(tsne10: np.ndarray, tsne100: np.ndarray,
                         y10: np.ndarray, y100: np.ndarray,
                         clusters10: np.ndarray, clusters100: np.ndarray,
                         path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    panels = [
        (axes[0, 0], tsne10, y10, _colors(10), "CIFAR-10 · t-SNE 真实类别", "A"),
        (axes[0, 1], tsne10, clusters10, _colors(10), "CIFAR-10 · t-SNE 聚类", "B"),
        (axes[1, 0], tsne100, y100, _colors(100), "CIFAR-100 · t-SNE 真实类别", "C"),
        (axes[1, 1], tsne100, clusters100, _colors(100), "CIFAR-100 · t-SNE 聚类", "D"),
    ]
    for ax, coords, labels, colors, title, letter in panels:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors[labels], s=5, alpha=0.65,
                   linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        _panel(ax, letter)
    fig.suptitle("类别数量增加后特征分布的对比（t-SNE）", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Comparison dashboard / model diagram
# ---------------------------------------------------------------------------
def plot_comparison_dashboard(m10: dict, m100: dict, c10: dict, c100: dict,
                              r10: pd.DataFrame, r100: pd.DataFrame, path: Path) -> Path:
    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # A: classification metrics
    ax = fig.add_subplot(gs[0, 0])
    metrics = ["accuracy", "top3_accuracy", "top5_accuracy"]
    labels = ["Accuracy", "Top-3", "Top-5"]
    x = np.arange(len(metrics))
    w = 0.36
    v10 = [m10[k] * 100 for k in metrics]
    v100 = [m100[k] * 100 for k in metrics]
    b1 = ax.bar(x - w / 2, v10, w, label="CIFAR-10", color=PALETTE["cifar10"], alpha=0.9)
    b2 = ax.bar(x + w / 2, v100, w, label="CIFAR-100", color=PALETTE["cifar100"], alpha=0.9)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("准确率 (%)")
    ax.set_title("分类性能对比")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _panel(ax, "A")

    # B: clustering metrics
    ax = fig.add_subplot(gs[0, 1])
    caveat = ["ari", "nmi", "purity", "silhouette"]
    labels = ["ARI", "NMI", "Purity", "Silhouette"]
    x = np.arange(len(caveat))
    v10 = [c10[k] for k in caveat]
    v100 = [c100[k] for k in caveat]
    b1 = ax.bar(x - w / 2, v10, w, label=f"CIFAR-10 (K=10)", color=PALETTE["cifar10"], alpha=0.9)
    b2 = ax.bar(x + w / 2, v100, w, label=f"CIFAR-100 (K=100)", color=PALETTE["cifar100"], alpha=0.9)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", fontsize=7.8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(v10), max(v100)) * 1.25 + 0.02)
    ax.set_title("聚类性能对比")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _panel(ax, "B")

    # C: per-class accuracy distribution
    ax = fig.add_subplot(gs[1, 0])
    data = [r10["accuracy"].values * 100, r100["accuracy"].values * 100]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=False,
                    medianprops=dict(color="#111827", lw=1.5))
    for patch, color in zip(bp["boxes"], DATASET_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for i, (vals, color) in enumerate(zip(data, DATASET_COLORS), start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, color=color, alpha=0.55,
                   edgecolors="none")
        ax.text(i, 104, f"均值 {np.mean(vals):.1f}%", ha="center", fontsize=9,
                fontweight="bold", color=color)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["CIFAR-10\n(10 类)", "CIFAR-100\n(100 类)"])
    ax.set_ylabel("单类准确率 (%)")
    ax.set_ylim(0, 110)
    ax.set_title("各类别准确率分布")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _panel(ax, "C")

    # D: compact key-metric table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    rows = [
        ("类别数", f"{m10['num_classes']}", f"{m100['num_classes']}"),
        ("随机猜测准确率", f"{m10['chance_level']*100:.1f}%", f"{m100['chance_level']*100:.1f}%"),
        ("测试 Accuracy", f"{m10['accuracy']*100:.2f}%", f"{m100['accuracy']*100:.2f}%"),
        ("Macro F1", f"{m10['macro_f1']:.3f}", f"{m100['macro_f1']:.3f}"),
        ("聚类 ARI", f"{c10['ari']:.4f}", f"{c100['ari']:.4f}"),
        ("聚类 NMI", f"{c10['nmi']:.4f}", f"{c100['nmi']:.4f}"),
        ("聚类 Purity", f"{c10['purity']:.4f}", f"{c100['purity']:.4f}"),
        ("聚类 Silhouette", f"{c10['silhouette']:.4f}", f"{c100['silhouette']:.4f}"),
    ]
    table = ax.table(cellText=[[a, b, c] for a, b, c in rows],
                     colLabels=["指标", "CIFAR-10", "CIFAR-100"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(fontweight="bold")
        elif col == 1:
            cell.set_facecolor("#EAF2F8")
        else:
            cell.set_facecolor("#FBEEE4")
    ax.set_title("关键指标一览", fontsize=13, fontweight="bold", pad=12)
    _panel(ax, "D")

    fig.suptitle("CIFAR-10 与 CIFAR-100 综合实验指标对比", fontsize=17, fontweight="bold", y=0.98)
    return save_fig(fig, path)


def plot_framework(path: Path, img_size: int = 96) -> Path:
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF2F8", ec="#2E6F9E", fs=9.5, bold=False):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#111827", linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#4B5563"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=2, shrinkB=2))

    box(0.15, 2.6, 1.55, 1.0, "输入图像\n32×32×3", fc="#F3F4F6", ec="#6B7280", bold=True)
    box(2.0, 2.6, 1.55, 1.0, f"统一预处理\nResize {img_size}×{img_size}\nImageNet 归一化", fc="#F3F4F6", ec="#6B7280")
    box(3.85, 2.35, 3.1, 1.5, "共享 ResNet18 Backbone\nImageNet-1K 预训练\n4 个 Stage + GAP", fc="#E8F4EF", ec="#1F9D8B", bold=True)
    box(7.25, 2.55, 1.55, 1.1, "512 维\n特征向量", fc="#FFF7E6", ec="#D9772B", bold=True)
    box(9.15, 4.0, 2.25, 0.95, "CIFAR-10 分类头\nLinear(512→10)", fc="#EAF2F8", ec="#2E6F9E")
    box(9.15, 1.2, 2.25, 0.95, "CIFAR-100 分类头\nLinear(512→100)", fc="#FBEEE4", ec="#D9772B")
    box(11.85, 4.0, 1.55, 0.95, "10 类\nSoftmax", fc="#EAF2F8", ec="#2E6F9E")
    box(11.85, 1.2, 1.55, 0.95, "100 类\nSoftmax", fc="#FBEEE4", ec="#D9772B")
    box(9.15, 2.75, 2.25, 0.95, "K-Means 聚类头\nK=10 / K=100", fc="#F5F3FF", ec="#7C3AED")
    box(11.85, 2.75, 1.55, 0.95, "ARI / NMI\nPCA / t-SNE", fc="#F5F3FF", ec="#7C3AED")

    arrow(1.70, 3.1, 2.0, 3.1)
    arrow(3.55, 3.1, 3.85, 3.1)
    arrow(6.95, 3.1, 7.25, 3.1)
    arrow(8.80, 3.35, 9.15, 4.45)
    arrow(8.80, 2.85, 9.15, 1.7)
    arrow(8.80, 3.1, 9.15, 3.22)
    arrow(11.40, 4.47, 11.85, 4.47)
    arrow(11.40, 1.67, 11.85, 1.67)
    arrow(11.40, 3.22, 11.85, 3.22)

    ax.text(7.5, 5.85, "统一预处理 → 共享预训练 CNN → 特征向量 → 分类 / 聚类",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")
    ax.text(0.2, 0.35, "注：分类训练使用 Hugging Face Trainer；聚类直接调用 sklearn MiniBatchKMeans，不额外训练深度聚类网络。",
            fontsize=8.5, color="#4B5563")
    return save_fig(fig, path)
