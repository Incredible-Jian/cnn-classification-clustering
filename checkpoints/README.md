# 模型权重

由于两个训练权重合计约 90 MB，未直接提交到 Git 仓库，统一通过 **GitHub Release** 提供：

- 下载页：<https://github.com/Incredible-Jian/cnn-classification-clustering/releases/tag/v1.0-weights>

| 文件 | 说明 | 大小 |
|---|---|---|
| `resnet18_cifar10.pth` | CIFAR-10 双头模型权重（使用 10 类头） | 约 45 MB |
| `resnet18_cifar100.pth` | CIFAR-100 双头模型权重（使用 100 类头） | 约 45 MB |

## 下载

```bash
mkdir -p checkpoints/cifar10 checkpoints/cifar100
curl -L -o checkpoints/cifar10/resnet18_cifar10.pth \
  https://github.com/Incredible-Jian/cnn-classification-clustering/releases/download/v1.0-weights/resnet18_cifar10.pth
curl -L -o checkpoints/cifar100/resnet18_cifar100.pth \
  https://github.com/Incredible-Jian/cnn-classification-clustering/releases/download/v1.0-weights/resnet18_cifar100.pth
```

## 说明

- 权重由 `main.ipynb` 使用 Hugging Face `Trainer` 实际训练得到；
- 每个权重包含共享 ResNet18 Backbone 与 10 类 / 100 类两个分类头；
- ImageNet 预训练权重 `resnet18-f37072fd.pth` 会在首次运行时自动下载到 `checkpoints/pretrained/`，无需手动准备；
- 如不使用预训练权重，也可以直接重新运行 Notebook 完成训练（CPU 约 20 分钟）。
