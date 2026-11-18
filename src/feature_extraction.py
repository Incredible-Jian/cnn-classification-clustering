"""CNN feature extraction for the K-Means experiments."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import CIFARDataset, DatasetBundle, build_transforms
from .models import DualHeadResNet18, FeatureBackbone


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    img_size: int,
    task_id: int,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    ds = CIFARDataset(images, labels, indices, build_transforms(img_size, train=False), task_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    backbone = backbone.to(device).eval()
    feats, ys = [], []
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        x = batch["pixel_values"].to(device, non_blocking=False)
        f = backbone(x)
        feats.append(f.detach().cpu().numpy().astype(np.float32))
        ys.append(batch["labels"].numpy())
        if step % 20 == 0:
            done = min(step * batch_size, len(ds))
            print(f"  {desc}: {done}/{len(ds)} images ({done/max(time.time()-t0,1e-6):.0f} img/s)", flush=True)
    features = np.concatenate(feats, axis=0)
    labels_out = np.concatenate(ys, axis=0)
    print(f"  {desc}: done {features.shape} in {time.time()-t0:.1f}s", flush=True)
    return features, labels_out


def save_feature_bundle(path: Path, features: np.ndarray, labels: np.ndarray,
                        meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )


def load_feature_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    return data["features"], data["labels"], json.loads(str(data["meta"]))


def extract_for_bundle(
    bundle: DatasetBundle,
    output_dir: Path,
    frozen_img_size: int = 224,
    finetuned_img_size: Optional[int] = None,
    finetuned_checkpoint: Optional[Path] = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict:
    """Extract test-set features with the frozen and the fine-tuned backbone."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    print(f"\n=== Feature extraction: {bundle.spec.display_name} (frozen ImageNet backbone, {frozen_img_size}px) ===")
    frozen = FeatureBackbone(pretrained=True)
    feats, labels = extract_features(
        frozen, bundle.images_test, bundle.labels_test, bundle.test_indices,
        frozen_img_size, bundle.spec.task_id, batch_size, device,
        desc=f"{bundle.spec.name}-frozen",
    )
    frozen_path = output_dir / f"{bundle.spec.name}_test_frozen.npz"
    meta = {
        "dataset": bundle.spec.name,
        "source": "frozen ImageNet-1K pretrained ResNet18 backbone",
        "img_size": frozen_img_size,
        "feature_dim": int(feats.shape[1]),
        "n_samples": int(feats.shape[0]),
        "normalization": "ImageNet mean/std",
    }
    save_feature_bundle(frozen_path, feats, labels, meta)
    summary["frozen"] = {"path": str(frozen_path), "shape": list(feats.shape), **meta}

    if finetuned_checkpoint is not None:
        from .models import load_checkpoint
        model = load_checkpoint(finetuned_checkpoint, device)
        ft_size = finetuned_img_size or bundle.info["img_size"]
        print(f"\n=== Feature extraction: {bundle.spec.display_name} (fine-tuned backbone, {ft_size}px) ===")
        feats_ft, labels_ft = extract_features(
            model.backbone, bundle.images_test, bundle.labels_test, bundle.test_indices,
            ft_size, bundle.spec.task_id, batch_size, device,
            desc=f"{bundle.spec.name}-finetuned",
        )
        ft_path = output_dir / f"{bundle.spec.name}_test_finetuned.npz"
        meta_ft = {
            "dataset": bundle.spec.name,
            "source": "fine-tuned ResNet18 backbone (classification checkpoint)",
            "img_size": ft_size,
            "feature_dim": int(feats_ft.shape[1]),
            "n_samples": int(feats_ft.shape[0]),
            "normalization": "ImageNet mean/std",
        }
        save_feature_bundle(ft_path, feats_ft, labels_ft, meta_ft)
        summary["finetuned"] = {"path": str(ft_path), "shape": list(feats_ft.shape), **meta_ft}

    return summary
