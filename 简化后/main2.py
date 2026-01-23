import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 导入简化后的函数
from dataset2 import get_file_map, create_loaders
from model.FTFNet import FTFNet_model
from tool.Loss_plot import plot_losses
from tool.Plot_pred import plot_prediction_with_mean_std
from tool.Error_compute import plot_error, compute_error_over_time
from tool.Pred_plot import visual
from tool.metrics import metric
from tool.Early_stop import AdvancedEarlyStopping

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file_map transform to trials list 
def list_trials_from_map(file_map):
    trials = []
    for obj, files in file_map.items():
        for f in files:
            trials.append((obj, f.name))
    return trials

def _select_validation_trial(trials, held_out_trial):
    same_object_trials = [t for t in trials if t[0] == held_out_trial[0]]
    if len(same_object_trials) > 1:
        for t in same_object_trials:
            if t != held_out_trial: return t
    for t in trials:
        if t != held_out_trial: return t
    raise ValueError("No available trials for validation.")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    train_losses, val_losses = [], []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = AdvancedEarlyStopping(patience=20, min_delta=1e-4, min_epochs=30, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)[:, :, -1] 
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)[:, :, -1]
                val_loss += criterion(outputs, y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.4f} Val: {val_loss:.4f}")

        early_stopping(val_loss, model, best_model_path, epoch)
        if early_stopping.early_stop: break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return train_losses, val_losses

# --- test ---
def test(model, test_loader, save_dir):
    model.eval()
    predictions, ground_truth = [], []
    
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)[:, :, -1]
            
            pred = outputs.cpu().numpy()
            true = y.cpu().numpy()
            predictions.append(pred)
            ground_truth.append(true)

            if i % 20 == 0:
                input_seq = x[0, :, -1].cpu().numpy()
                gt_full = np.concatenate([input_seq, true[0]])
                pd_full = np.concatenate([input_seq, pred[0]])
                visual(gt_full, pd_full, os.path.join(save_dir, f"test_{i}.pdf"))

    preds = np.vstack(predictions)
    trues = np.vstack(ground_truth)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print(f"Test Result -> MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write(f"MSE: {mse}\nMAE: {mae}\nRMSE: {rmse}\n")
    
    np.save(os.path.join(save_dir, "pred.npy"), preds)

def main():
    config = {
        "data_dir": "Processed_Finally_data3",
        "batch_size": 32,
        "epochs": 200,
        "lr": 0.001,
        "seq_len": 96,
        "pred_len": 96,
        "norm_type": "global", # 或 "per_object"
        "results_root": "results_FTFNet_loto2"
    }

    # get all trial
    file_map = get_file_map(config["data_dir"])
    trials = list_trials_from_map(file_map)
    
    if not trials:
        raise RuntimeError("No trial found.")

    #  Leave-one-trial-out train
    for fold_idx, held_out_trial in enumerate(trials, start=1):
        val_trial = _select_validation_trial(trials, held_out_trial)
        
        train_loader, val_loader, test_loader, normalizer = create_loaders(
            data_root=config["data_dir"],
            held_out_trial=held_out_trial,
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            normalization_type=config["norm_type"]
        )

        # save path
        fold_name = f"fold_{fold_idx}_{held_out_trial[0]}_{Path(held_out_trial[1]).stem}"
        save_dir = os.path.join(config["results_root"], fold_name)
        os.makedirs(save_dir, exist_ok=True)

        normalizer.save_params(os.path.join(save_dir, "normalizer_params.json"))

        # 初始化模型
        model = FTFNet_model(
            sensor_num=1,
            input_dim=3,
            seq_length=config["seq_len"],
            pre_length=config["pred_len"],
            embed_dim=32, 
            depth=1, 
            embed_size=32, 
            hidden_size=256, 
            dropout=0.4
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()

        print(f"\n===== Fold {fold_idx}/{len(trials)}: Test={held_out_trial} =====")

        # train
        train_loss, val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            config["epochs"], device, save_dir
        )

        # test and plot
        test(model, test_loader, save_dir)
        plot_losses(train_loss, val_loss, save_dir)

if __name__ == "__main__":
    main()