# FTFNet: A Frequency–Time Fusion Network for Slip Prediction in Dexterous Robotic Manipulation

<p align="center">
  <img src="image/FTFNet.png" alt="FTFNet Architecture" width="900">
</p>

<p align="center">
  <em>Figure 1: The architecture of the proposed FTFNet.</em>
</p>

---

## 📖 Introduction

This repository contains the official implementation of the **FTFNet** architecture and the associated **sp-dataset** for tactile data processing, as described in our paper. This project aims to provide a reproducible pipeline for tactile sensing research.

## 📁 Repository Structure

```
.
├── model/
│   └── FTFNet.py                    # FTFNet model architecture implementation
├── dataset.py                        # Data loading, augmentation, and dataset splitting logic
├── main.py                           # Main script for training, evaluation, and inference
├── sp-dataset.zip                    # Compressed tactile dataset (experimental data)
├── tool/                             # Prediction result metric calculation and curve plotting tools
├── result_FTFNet_loto/               # Leave-One-Task-Out (LOTO) cross-validation results
│   └── fold_*/                       # Results directory for each fold
│       ├── best_model.pth            # Trained model parameters
│       ├── normalizer_params.json    # Data normalization parameters (mean, std, etc.)
│       ├── loss_curve.png            # Loss curve plot
│       └── result.txt                # Evaluation results text
├── image/
│   └── FTFNet.png                    # Model architecture diagram
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.5.1+
- NumPy
- Matplotlib
- scikit-learn
- scipy
- pandas

### Installation

```bash
pip install torch numpy matplotlib scikit-learn scipy pandas
```

### Data Preparation

The dataset is provided in a compressed format. Before running the code, please unzip the file:

```bash
unzip sp-dataset.zip
```

Ensure the extracted data folder is placed in the root directory or as specified in `dataset.py`.

### Running the Code

To start the training or evaluation process, run the `main.py` script:

```bash
python main.py
```

## 🧠 Key Components

### FTFNet Model (`model/FTFNet.py`)

This is the core model proposed in our paper, designed to effectively process tactile information. The model combines frequency-domain and time-domain features to achieve slip prediction through a frequency-time fusion network.

### Dataset Processing (`dataset.py`)

Handles the entire data pipeline, including:
- **Data Loading**: Load data from sp-dataset
- **Data Augmentation**: Augmentation techniques to improve model robustness
- **Dataset Splitting**: Division of training, validation, and test sets
- **Data Normalization**: Standardization processing to ensure data consistency

### Utility Functions (`tool/`)

Contains multiple utility tools:
- `metrics.py`: Calculate various evaluation metrics (MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR)
- `Loss_plot.py`: Plot training and validation loss curves
- `Plot_pred.py`: Visualize prediction results
- `Error_compute.py`: Calculate and visualize prediction errors
- `Early_stop.py`: Implement early stopping mechanism to prevent overfitting

### LOTO Cross-Validation Results (`result_FTFNet_loto/`)

This directory contains model weights saved from Leave-One-Task-Out cross-validation experiments:
- `best_model.pth`: Model parameters with the best performance on the validation set during training
- `normalizer_params.json`: Stores scaling factors used during training to ensure consistent data preprocessing during inference
- Other visualization results and evaluation metrics

## 📊 Experiments & Reproducibility

The provided code and resources allow for the full reproduction of the results reported in the paper. By using the scripts in this repository and the provided model checkpoints, you can verify the performance of FTFNet on the SP-dataset.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests to improve this project.

## 📧 Contact

For any questions regarding the code or dataset, please submit an Issue in this repository.

## 🙏 Acknowledgments

We thank all researchers and developers who have contributed to this project.
