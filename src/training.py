"""Hugging Face Trainer configuration and classification metrics.

All optimisation, back-propagation, logging and checkpointing are delegated to
``transformers.Trainer``; no custom training loop is used.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# TensorFlow is installed in the base environment but its native runtime is
# broken on this machine; disable TF/Flax discovery before importing Trainer.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support)
from transformers import Trainer, TrainerCallback, TrainingArguments

from .models import DualHeadResNet18, build_model
from .datasets import DatasetBundle


class CIFARCollator:
    """Stack the dataset dictionaries into Trainer-ready tensors."""

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        task_ids = torch.tensor([f["task_ids"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels, "task_ids": task_ids}


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def compute_metrics_factory(num_classes: int):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits = np.asarray(eval_pred.predictions)
        labels = np.asarray(eval_pred.label_ids)
        preds = logits.argmax(axis=1)
        out = {
            "accuracy": float(accuracy_score(labels, preds)),
            "top3_accuracy": topk_accuracy(logits, labels, 3),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
        if num_classes >= 100:
            out["top5_accuracy"] = topk_accuracy(logits, labels, 5)
        try:
            out["log_loss"] = float(log_loss(labels, torch.softmax(torch.tensor(logits), 1).numpy(),
                                             labels=list(range(num_classes))))
        except Exception:
            out["log_loss"] = float("nan")
        return out
    return compute_metrics


class ExperimentLogCallback(TrainerCallback):
    """Print concise, notebook-visible logs and persist them as plain text."""

    def __init__(self, dataset_name: str, log_path: Path):
        self.dataset_name = dataset_name
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.t0 = time.time()

    def _write(self, line: str) -> None:
        stamp = f"[{time.time()-self.t0:7.1f}s]"
        full = f"{stamp} [{self.dataset_name}] {line}"
        print(full, flush=True)
        self.lines.append(full)
        self.log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if "loss" in logs:
            self._write(
                f"step {state.global_step:4d}/{state.max_steps} | epoch {state.epoch:5.2f} | "
                f"train_loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.2e}"
            )
        elif "eval_loss" in logs:
            self._write(
                f"eval @ epoch {state.epoch:5.2f} | eval_loss {logs['eval_loss']:.4f} | "
                f"acc {logs.get('eval_accuracy', float('nan')):.4f}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        self._write(f"training finished | best_metric={state.best_metric} | total_steps={state.global_step}")


@dataclass
class TrainResult:
    model: DualHeadResNet18
    trainer: Trainer
    log_history: pd.DataFrame
    history: dict
    val_metrics: dict
    checkpoint_path: Path
    elapsed_sec: float


def _make_args(dataset_name: str, output_dir: Path, epochs: int,
               batch_size: int, eval_batch_size: int, lr: float,
               logging_steps: int, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        # safetensors avoids torch.load's CVE-2025-32434 restriction on torch 2.5.
        save_safetensors=True,
    )


def train_classifier(
    bundle: DatasetBundle,
    checkpoint_dir: Path,
    metrics_dir: Path,
    epochs: int = 3,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 3e-4,
    logging_steps: int = 25,
    seed: int = 42,
) -> TrainResult:
    checkpoint_dir = Path(checkpoint_dir)
    metrics_dir = Path(metrics_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir.parent / "logs" / f"trainer_{bundle.spec.name}.log"

    model = build_model(pretrained=True)
    args = _make_args(bundle.spec.name, checkpoint_dir / "trainer", epochs, batch_size,
                      eval_batch_size, lr, logging_steps, seed)
    callback = ExperimentLogCallback(bundle.spec.name, log_path)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=bundle.train,
        eval_dataset=bundle.val,
        data_collator=CIFARCollator(),
        compute_metrics=compute_metrics_factory(bundle.spec.num_classes),
        callbacks=[callback],
    )

    print(f"\n=== Training {bundle.spec.display_name} ===")
    print(f"train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)} "
          f"img_size={bundle.info['img_size']} epochs={epochs} batch={batch_size} lr={lr}")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    val_metrics = trainer.evaluate()
    print(f"[{bundle.spec.name}] validation: {json.dumps(val_metrics, ensure_ascii=False)}")

    ckpt_path = checkpoint_dir / f"resnet18_{bundle.spec.name}.pth"
    torch.save(model.state_dict(), ckpt_path)

    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(metrics_dir / f"{bundle.spec.name}_train_log.csv", index=False)

    history = {
        "dataset": bundle.spec.name,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": lr,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler": "cosine",
        "optimizer": "AdamW",
        "img_size": bundle.info["img_size"],
        "train_size": len(bundle.train),
        "val_size": len(bundle.val),
        "test_size": len(bundle.test),
        "elapsed_sec": round(elapsed, 2),
        "train_runtime_sec": round(train_result.metrics.get("train_runtime", elapsed), 2),
        "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "total_steps": trainer.state.global_step,
        "final_train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "val_metrics": val_metrics,
        "log_history_records": len(log_history),
    }
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / f"{bundle.spec.name}_training_summary.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep the Trainer state for inspection, but drop the large optimizer
    # checkpoints; the final weights are stored in resnet18_<dataset>.pth.
    trainer_state = checkpoint_dir / "trainer" / "trainer_state.json"
    if trainer_state.exists():
        shutil.copy2(trainer_state, metrics_dir / f"{bundle.spec.name}_trainer_state.json")
    if (checkpoint_dir / "trainer").exists():
        shutil.rmtree(checkpoint_dir / "trainer", ignore_errors=True)

    return TrainResult(model, trainer, log_history, history, val_metrics, ckpt_path, elapsed)


def evaluate_test(
    trainer: Trainer,
    bundle: DatasetBundle,
    metrics_dir: Path,
    figures_dir: Optional[Path] = None,
) -> dict:
    """Run Trainer.predict on the untouched test split and save all metrics."""
    metrics_dir = Path(metrics_dir)
    pred = trainer.predict(bundle.test)
    logits = np.asarray(pred.predictions, dtype=np.float64)
    labels = np.asarray(pred.label_ids, dtype=np.int64)
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    cm = confusion_matrix(labels, preds, labels=list(range(bundle.spec.num_classes)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(bundle.spec.num_classes)), zero_division=0)
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    metrics = {
        "dataset": bundle.spec.name,
        "display_name": bundle.spec.display_name,
        "num_classes": bundle.spec.num_classes,
        "test_size": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "top3_accuracy": topk_accuracy(logits, labels, 3),
        "top5_accuracy": topk_accuracy(logits, labels, 5),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class_accuracy_mean": float(per_class_acc.mean()),
        "per_class_accuracy_std": float(per_class_acc.std()),
        "per_class_accuracy_min": float(per_class_acc.min()),
        "per_class_accuracy_max": float(per_class_acc.max()),
        "test_loss": float(pred.metrics.get("test_loss", float("nan"))),
        "elapsed_sec": round(float(pred.metrics.get("test_runtime", 0.0)), 2),
        "chance_level": float(1.0 / bundle.spec.num_classes),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{bundle.spec.name}_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    np.save(metrics_dir / f"{bundle.spec.name}_test_logits.npy", logits.astype(np.float32))
    np.save(metrics_dir / f"{bundle.spec.name}_test_labels.npy", labels)
    np.save(metrics_dir / f"{bundle.spec.name}_test_probs.npy", probs.astype(np.float32))
    np.save(metrics_dir / f"{bundle.spec.name}_confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=bundle.spec.class_names, columns=bundle.spec.class_names).to_csv(
        metrics_dir / f"{bundle.spec.name}_confusion_matrix.csv", encoding="utf-8-sig")

    report = pd.DataFrame({
        "class": bundle.spec.class_names,
        "class_zh": bundle.spec.class_names_zh,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": per_class_acc,
    })
    report.to_csv(metrics_dir / f"{bundle.spec.name}_classification_report.csv",
                  index=False, encoding="utf-8-sig")

    # Top confused class pairs (excluding the diagonal).
    pairs = []
    for i in range(bundle.spec.num_classes):
        for j in range(bundle.spec.num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_idx": i, "true_class": bundle.spec.class_names[i],
                    "true_class_zh": bundle.spec.class_names_zh[i],
                    "pred_idx": j, "pred_class": bundle.spec.class_names[j],
                    "pred_class_zh": bundle.spec.class_names_zh[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / max(cm[i].sum(), 1)),
                })
    pairs = sorted(pairs, key=lambda r: r["count"], reverse=True)
    pd.DataFrame(pairs[:50]).to_csv(metrics_dir / f"{bundle.spec.name}_top_confusions.csv",
                                    index=False, encoding="utf-8-sig")

    metrics["confusion_matrix_shape"] = list(cm.shape)
    metrics["n_correct"] = int(cm.diagonal().sum())
    metrics["most_confused_pair"] = pairs[0] if pairs else None
    return metrics


def save_predictions_csv(bundle: DatasetBundle, metrics_dir: Path) -> None:
    logits = np.load(metrics_dir / f"{bundle.spec.name}_test_logits.npy")
    labels = np.load(metrics_dir / f"{bundle.spec.name}_test_labels.npy")
    preds = logits.argmax(1)
    df = pd.DataFrame({
        "true_idx": labels,
        "true_class": [bundle.spec.class_names[i] for i in labels],
        "pred_idx": preds,
        "pred_class": [bundle.spec.class_names[i] for i in preds],
        "confidence": np.max(torch.softmax(torch.tensor(logits), 1).numpy(), axis=1),
        "correct": preds == labels,
    })
    df.to_csv(metrics_dir / f"{bundle.spec.name}_test_predictions.csv", index=False,
              encoding="utf-8-sig")
