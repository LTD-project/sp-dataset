# FTFNet: A Frequency–Time Fusion Network for Slip Prediction in Dexterous Robotic Manipulation

<p align="center">
  <img src="image/FTFNet.png" alt="FTFNet Architecture" width="900">
</p>

<p align="center">
  <em>Figure 1: The architecture of the proposed FTFNet.</em>
</p>

---

## 📖 简介 / Introduction

This repository contains the official implementation of the **FTFNet** architecture and the associated **sp-dataset** for tactile data processing, as described in our paper. This project aims to provide a reproducible pipeline for tactile sensing research.

本仓库包含了论文中描述的 **FTFNet** 架构的官方实现以及相关的 **sp-dataset** 触觉数据处理代码。本项目旨在为触觉感知研究提供一个可复现的流程。

## 📁 项目结构 / Repository Structure

```
.
├── model/
│   └── FTFNet.py                    # FTFNet 模型架构实现
├── dataset.py                        # 数据加载、增强和数据集划分逻辑
├── main.py                           # 训练、评估和推理的主脚本
├── sp-dataset.zip                    # 压缩的触觉数据集（实验数据）
├── tool/                             # 预测结果指标计算和曲线绘制工具
├── result_FTFNet_loto/               # Leave-One-Task-Out (LOTO) 交叉验证结果
│   └── fold_*/                       # 每个 fold 的结果目录
│       ├── best_model.pth            # 训练好的模型参数
│       ├── normalizer_params.json    # 数据归一化参数（均值、标准差等）
│       ├── loss_curve.png            # 损失曲线图
│       └── result.txt                # 评估结果文本
├── image/
│   └── FTFNet.png                    # 模型架构图
└── README.md                         # 项目文档
```

## 🚀 快速开始 / Getting Started

### 环境要求 / Requirements

- Python 3.10+
- PyTorch 2.5.1+
- NumPy
- Matplotlib
- scikit-learn
- scipy
- pandas

### 安装依赖 / Installation

```bash
pip install torch numpy matplotlib scikit-learn scipy pandas
```

### 数据准备 / Data Preparation

数据集以压缩格式提供。运行代码前，请先解压文件：

```bash
unzip sp-dataset.zip
```

确保解压后的数据文件夹位于根目录，或按照 `dataset.py` 中的指定路径放置。

### 运行代码 / Running the Code

开始训练或评估过程，运行 `main.py` 脚本：

```bash
python main.py
```

## 🧠 核心组件 / Key Components

### FTFNet 模型 (`model/FTFNet.py`)

这是论文中提出的核心模型，设计用于有效处理触觉信息。模型结合了频域和时域特征，通过频率-时间融合网络实现滑移预测。

### 数据集处理 (`dataset.py`)

处理整个数据流程，包括：
- **数据加载**：从 sp-dataset 加载数据
- **数据增强**：提高模型鲁棒性的增强技术
- **数据集划分**：训练集、验证集和测试集的划分
- **数据归一化**：标准化处理，确保数据一致性

### 工具函数 (`tool/`)

包含多个实用工具：
- `metrics.py`: 计算各种评估指标（MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR）
- `Loss_plot.py`: 绘制训练和验证损失曲线
- `Plot_pred.py`: 可视化预测结果
- `Error_compute.py`: 计算和可视化预测误差
- `Early_stop.py`: 实现早停机制，防止过拟合

### LOTO 交叉验证结果 (`result_FTFNet_loto/`)

该目录包含 Leave-One-Task-Out 交叉验证实验保存的模型权重：
- `best_model.pth`: 训练过程中验证集上表现最好的模型参数
- `normalizer_params.json`: 存储训练时使用的缩放因子，确保推理时数据预处理的一致性
- 其他可视化结果和评估指标

## 📊 实验与可复现性 / Experiments & Reproducibility

提供的代码和资源允许完整复现论文中报告的结果。通过使用本仓库中的脚本和提供的模型检查点，您可以验证 FTFNet 在 SP-dataset 上的性能。


## 🤝 贡献 / Contributing

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 📧 联系方式 / Contact

如有关于代码或数据集的问题，请通过以下方式联系：
- Email: xy_l@tongji.edu.cn
- 或在本仓库中提交 Issue



## 🙏 致谢 / Acknowledgments

感谢所有为本项目做出贡献的研究人员和开发者。
