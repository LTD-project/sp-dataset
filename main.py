import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from dataset import create_leave_one_trial_out_loaders,list_all_trials
from model.FTFNet import FTFNet_model
from tool.Loss_plot import plot_losses
from tool.Plot_pred import plot_prediction_with_mean_std
from tool.Error_compute import plot_error, compute_error_over_time
from tool.Error_compute_with_zero import plot_error_with_zero, compute_error_over_time_with_zero
from tool.Pred_plot import visual
from tool.metrics import metric
from tool.Early_stop import AdvancedEarlyStopping, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    train_losses = []
    val_losses = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    early_stopping = AdvancedEarlyStopping(
        patience=20,
        min_delta=1e-4,
        min_epochs=30,
        window=3,
        verbose=True,
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for sequences, targets, _ in train_loader:
            sequences, targets = sequences.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = outputs[:, :, -1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets, _ in val_loader:
                sequences, targets = sequences.float().to(device), targets.float().to(device)
                outputs = model(sequences)
                outputs = outputs[:, :, -1]
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        early_stopping(val_loss, model, best_model_path, epoch)
        if early_stopping.early_stop:
            break

    print("Load the best model and conduct prediction visualization")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    time, train_mean_error, train_std_error = compute_error_over_time(model, train_loader, device)
    time, val_mean_error, val_std_error = compute_error_over_time(model, val_loader, device)
    plot_error(time, train_mean_error, train_std_error, val_mean_error, val_std_error, save_dir)

    return train_losses, val_losses


def test(model, test_loader, save_dir, object_type):
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.eval()
    best_mse = float("inf")
    best_input = None
    best_true = None
    best_pred = None

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for i, (sequences, targets, _) in enumerate(test_loader):
            sequences, targets = sequences.float().to(device), targets.float().to(device)
            outputs = model(sequences)
            outputs = outputs[:, :, -1]
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            pred = outputs
            true = targets
            predictions.append(pred)
            ground_truth.append(true)

            batch_mse = np.mean((pred - true) ** 2)
            if batch_mse < best_mse:
                best_mse = batch_mse
                input_np = sequences.detach().cpu().numpy()
                best_input = input_np[0, :, -1]
                best_true = true[0, :]
                best_pred = pred[0, :]

            if i % 20 == 0:
                input_arr = sequences.detach().cpu().numpy()
                gt = np.concatenate((input_arr[0, :, -1], true[0, :]), axis=0)
                pd = np.concatenate((input_arr[0, :, -1], pred[0, :]), axis=0)
                visual(gt, pd, os.path.join(save_dir, str(i) + ".pdf"))

    if best_input is not None:
        gt = np.concatenate((best_input, best_true), axis=0)
        pd = np.concatenate((best_input, best_pred), axis=0)
        visual(gt, pd, os.path.join(save_dir, "best_prediction.pdf"))

    preds = np.concatenate(predictions, axis=0)
    trues = np.concatenate(ground_truth, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print(f"mse:{mse}, mae:{mae}, rmse:{rmse}")
    with open(os.path.join(save_dir, "result.txt"), "a") as f:
        f.write(f"mse:{mse}, mae:{mae}, rse:{rse}, rmse:{rmse}\n\n")

    np.save(os.path.join(save_dir, "pred.npy"), preds)


def _select_validation_trial(trials, held_out_trial):
    same_object_trials = [t for t in trials if t[0] == held_out_trial[0]]
    if len(same_object_trials) > 1:
        sorted_trials = sorted(same_object_trials, key=lambda x: x[1])
        idx = sorted_trials.index(held_out_trial)
        val_idx = (idx + 1) % len(sorted_trials)
        val_trial = sorted_trials[val_idx]
        if val_trial != held_out_trial:
            return val_trial

    for trial in trials:
        if trial != held_out_trial:
            return trial
    raise ValueError("no available validation trials in the dataset")


def main():
    data_dir = "sp-dataset"
    batch_size = 32
    epochs = 200
    learning_rate = 0.001
    weight_decay = 1e-5

    input_dim = 3
    seq_length = 96
    label_length = 48
    pred_steps = 96
    depth = 1
    embed_dim = 32
    embed_size = 32
    hidden_size = 256
    dropout = 0.4
    object_types = None

    augmentation_params = {
        "noise_level": 0.01,
        "scaling_sigma": 0.1,
    }

    trials = list_all_trials(data_dir, object_types)
    if not trials:
        raise RuntimeError("No trial was found in the dataset")

    results_root = "results_FTFNet_loto"
    os.makedirs(results_root, exist_ok=True)

    for fold_idx, held_out_trial in enumerate(trials, start=1):
        val_trial = _select_validation_trial(trials, held_out_trial)

        train_loader, val_loader, test_loader, normalizer = create_leave_one_trial_out_loaders(
            data_root=data_dir,
            held_out_trial=held_out_trial,
            val_trial=val_trial,
            batch_size=batch_size,
            sequence_length=seq_length,
            label_length=label_length,
            prediction_length=pred_steps,
            object_type=object_types,
            augmentation_params=augmentation_params,
        )

        fold_name = f"fold_{fold_idx}_{held_out_trial[0]}_{os.path.splitext(held_out_trial[1])[0]}"
        save_dir = os.path.join(results_root, fold_name)
        os.makedirs(save_dir, exist_ok=True)

        normalizer_path = os.path.join(save_dir, "normalizer_params.json")
        normalizer.save_normalizer_params(normalizer_path)

        model = FTFNet_model(
            sensor_num=1,
            input_dim=input_dim,
            seq_length=seq_length,
            embed_dim=embed_dim,
            depth=depth,
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout,
            pre_length=pred_steps,
        ).to(device)

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print(
            f"\n===== Fold {fold_idx}/{len(trials)}: test trial={held_out_trial}, val trial={val_trial} ====="
        )

        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs,
            device=device,
            save_dir=save_dir,
        )

        test(model, test_loader, save_dir, held_out_trial[0])
        plot_losses(train_losses, val_losses, save_dir)


if __name__ == "__main__":
    main()

