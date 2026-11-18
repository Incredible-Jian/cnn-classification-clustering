"""CIFAR-10 / CIFAR-100 loading, stratified splitting and preprocessing.

The experiments use the official CIFAR images.  For reproducibility on a
CPU-only machine the official 50,000-image training split is sub-sampled in a
*class-balanced* way; the official 10,000-image test split is always used in
full and is never touched during training or model selection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_ZH = {
    "airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫",
    "deer": "鹿", "dog": "狗", "frog": "青蛙", "horse": "马",
    "ship": "船", "truck": "卡车",
}

CIFAR100_ZH = {
    "apple": "苹果", "aquarium_fish": "水族箱鱼", "baby": "婴儿", "bear": "熊",
    "beaver": "海狸", "bed": "床", "bee": "蜜蜂", "beetle": "甲虫",
    "bicycle": "自行车", "bottle": "瓶子", "bowl": "碗", "boy": "男孩",
    "bridge": "桥", "bus": "公交车", "butterfly": "蝴蝶", "camel": "骆驼",
    "can": "易拉罐", "castle": "城堡", "caterpillar": "毛虫", "cattle": "牛",
    "chair": "椅子", "chimpanzee": "黑猩猩", "clock": "时钟", "cloud": "云",
    "cockroach": "蟑螂", "couch": "沙发", "crab": "螃蟹", "crocodile": "鳄鱼",
    "cup": "杯子", "dinosaur": "恐龙", "dolphin": "海豚", "elephant": "大象",
    "flatfish": "比目鱼", "forest": "森林", "fox": "狐狸", "girl": "女孩",
    "hamster": "仓鼠", "house": "房子", "kangaroo": "袋鼠", "keyboard": "键盘",
    "lamp": "台灯", "lawn_mower": "割草机", "leopard": "豹", "lion": "狮子",
    "lizard": "蜥蜴", "lobster": "龙虾", "man": "男人", "maple_tree": "枫树",
    "motorcycle": "摩托车", "mountain": "山", "mouse": "老鼠", "mushroom": "蘑菇",
    "oak_tree": "橡树", "orange": "橙子", "orchid": "兰花", "otter": "水獭",
    "palm_tree": "棕榈树", "pear": "梨", "pickup_truck": "皮卡", "pine_tree": "松树",
    "plain": "平原", "plate": "盘子", "plum": "李子", "poppy": "罂粟",
    "porcupine": "豪猪", "possum": "负鼠", "rabbit": "兔子", "raccoon": "浣熊",
    "ray": "鳐鱼", "road": "道路", "rocket": "火箭", "rose": "玫瑰",
    "sea": "海", "seal": "海豹", "shark": "鲨鱼", "shrew": "鼩鼱",
    "skunk": "臭鼬", "skyscraper": "摩天楼", "snail": "蜗牛", "snake": "蛇",
    "spider": "蜘蛛", "squirrel": "松鼠", "streetcar": "有轨电车", "sunflower": "向日葵",
    "sweet_pepper": "甜椒", "table": "桌子", "tank": "坦克", "telephone": "电话",
    "television": "电视", "tiger": "老虎", "tractor": "拖拉机", "train": "火车",
    "trout": "鳟鱼", "tulip": "郁金香", "turtle": "乌龟", "wardrobe": "衣柜",
    "whale": "鲸鱼", "willow_tree": "柳树", "wolf": "狼", "woman": "女人",
    "worm": "蠕虫",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    class_names: Tuple[str, ...]
    class_names_zh: Tuple[str, ...]
    task_id: int
    epoch_size: int = 50_000
    image_shape: Tuple[int, int, int] = (32, 32, 3)

    @property
    def display_name(self) -> str:
        return "CIFAR-10" if self.name == "cifar10" else "CIFAR-100"


def _load_names(name: str) -> List[str]:
    path = DATA_DIR / f"{name}_class_names.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if name == "cifar10":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"]
    raise FileNotFoundError(f"Missing class-name file: {path}. Run scripts/download_cifar_parquet.py")


def get_spec(name: str) -> DatasetSpec:
    if name not in {"cifar10", "cifar100"}:
        raise ValueError(name)
    names = _load_names(name)
    zh_map = CIFAR10_ZH if name == "cifar10" else CIFAR100_ZH
    zh = tuple(zh_map.get(n, n) for n in names)
    return DatasetSpec(
        name=name,
        num_classes=len(names),
        class_names=tuple(names),
        class_names_zh=zh,
        task_id=0 if name == "cifar10" else 1,
        epoch_size=50_000 if name == "cifar10" else 50_000,
    )


def load_arrays(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_images, train_labels, test_images, test_labels)."""
    train = np.load(DATA_DIR / f"{name}_train.npz")
    test = np.load(DATA_DIR / f"{name}_test.npz")
    return train["images"], train["labels"], test["images"], test["labels"]


def _class_indices(labels: np.ndarray, num_classes: int) -> List[np.ndarray]:
    return [np.flatnonzero(labels == c) for c in range(num_classes)]


def stratified_subset(
    labels: np.ndarray,
    per_class: int,
    num_classes: int,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Pick exactly ``per_class`` samples per class, excluding given indices."""
    excluded = set() if exclude is None else set(int(i) for i in exclude)
    chunks = []
    for idx in _class_indices(labels, num_classes):
        pool = np.array([i for i in idx if int(i) not in excluded], dtype=np.int64)
        if len(pool) < per_class:
            raise ValueError(f"Class has only {len(pool)} samples, need {per_class}")
        chunks.append(rng.choice(pool, size=per_class, replace=False))
    out = np.concatenate(chunks).astype(np.int64)
    rng.shuffle(out)
    return out


class CIFARDataset(Dataset):
    """Small wrapper returning the dict expected by the Hugging Face Trainer."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 transform, task_id: int):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.task_id = int(task_id)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, object]:
        j = int(self.indices[i])
        image = Image.fromarray(self.images[j])
        return {
            "pixel_values": self.transform(image),
            "labels": int(self.labels[j]),
            "task_ids": self.task_id,
        }


def build_transforms(img_size: int, train: bool, strong_aug: bool = True):
    if train:
        aug = [
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if strong_aug:
            aug.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15))
        aug += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(aug)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    train: CIFARDataset
    val: CIFARDataset
    test: CIFARDataset
    images_train: np.ndarray
    labels_train: np.ndarray
    images_test: np.ndarray
    labels_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    info: dict


def build_datasets(
    name: str,
    img_size: int = 96,
    train_per_class: int = 200,
    val_per_class: int = 50,
    seed: int = 42,
    strong_aug: bool = True,
) -> DatasetBundle:
    spec = get_spec(name)
    tr_x, tr_y, te_x, te_y = load_arrays(name)
    rng = np.random.default_rng(seed)

    train_idx = stratified_subset(tr_y, train_per_class, spec.num_classes, rng)
    val_idx = stratified_subset(tr_y, val_per_class, spec.num_classes, rng, exclude=train_idx)
    test_idx = np.arange(len(te_y), dtype=np.int64)

    # Train/validation are disjoint subsets of the official training split;
    # the test split is a separate array and is never indexed during training.
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(train_idx) == spec.num_classes * train_per_class
    assert len(val_idx) == spec.num_classes * val_per_class

    train_ds = CIFARDataset(tr_x, tr_y, train_idx, build_transforms(img_size, True, strong_aug), spec.task_id)
    val_ds = CIFARDataset(tr_x, tr_y, val_idx, build_transforms(img_size, False), spec.task_id)
    test_ds = CIFARDataset(te_x, te_y, test_idx, build_transforms(img_size, False), spec.task_id)

    info = {
        "dataset": name,
        "display_name": spec.display_name,
        "num_classes": spec.num_classes,
        "image_shape": list(spec.image_shape),
        "official_train_size": int(len(tr_y)),
        "official_test_size": int(len(te_y)),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(test_ds)),
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "img_size": img_size,
        "seed": seed,
        "class_names": list(spec.class_names),
        "class_names_zh": list(spec.class_names_zh),
        "class_counts_train": np.bincount(tr_y[ train_idx], minlength=spec.num_classes).tolist(),
        "class_counts_val": np.bincount(tr_y[val_idx], minlength=spec.num_classes).tolist(),
        "class_counts_test": np.bincount(te_y, minlength=spec.num_classes).tolist(),
    }
    return DatasetBundle(spec, train_ds, val_ds, test_ds, tr_x, tr_y, te_x, te_y,
                         train_idx, val_idx, test_idx, info)


def denormalize(tensor: np.ndarray | "torch.Tensor") -> np.ndarray:
    """Convert an ImageNet-normalised CHW tensor to an HWC uint8 image."""
    import torch
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    arr = np.asarray(tensor, dtype=np.float32).transpose(1, 2, 0)
    arr = arr * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def save_dataset_info(bundles: Sequence[DatasetBundle], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {b.spec.name: b.info for b in bundles}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
