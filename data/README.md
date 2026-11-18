# 数据集目录

本目录不包含 CIFAR 原始数据（约 596 MB），以避免污染 Git 仓库。

## 自动下载

```bash
python scripts/download_cifar_parquet.py
```

脚本会从 Hugging Face 镜像下载 CIFAR-10 / CIFAR-100 的官方图像副本，并转换为：

```
data/raw/cifar10_train.npz   (50000 × 32 × 32 × 3)
data/raw/cifar10_test.npz    (10000 × 32 × 32 × 3)
data/raw/cifar100_train.npz  (50000 × 32 × 32 × 3)
data/raw/cifar100_test.npz   (10000 × 32 × 32 × 3)
data/raw/*_class_names.json
```

下载源：`https://hf-mirror.com/datasets/uoft-cs/cifar10` 与 `.../cifar100`，
内容为官方 CIFAR 数据集的逐字节副本。

## 说明

- 训练/验证集从官方 50,000 张训练集中按类别分层抽取；
- 测试集完整保留官方 10,000 张，不参与训练；
- Notebook 首次运行时会自动检测并调用上述下载脚本。
