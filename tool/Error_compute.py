import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_error_over_time(model, data_loader, device, sampling_interval=1/32):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for sequences, targets, _ in data_loader:
            sequences = sequences.float().to(device)
            targets = targets.float().to(device)
            pred_len = targets.shape[1]

            outputs = model(sequences)  
            outputs = outputs[:, -pred_len:, -1]  
            error = torch.abs(outputs - targets)  
            all_errors.append(error.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  
    mean_error = np.mean(all_errors, axis=0)         
    std_error = np.std(all_errors, axis=0)           
    time_axis = np.arange(all_errors.shape[1]) * sampling_interval  
    return time_axis, mean_error, std_error
    
    
def plot_error(time_axis, train_mean_error, train_std_error, val_mean_error, val_std_error, save_dir):
    plt.ylim(0, 1.0)  # ⬅️ 新增代码
    # 平滑处理
    from scipy.ndimage import uniform_filter1d
    train_mean_error_smooth = uniform_filter1d(train_mean_error, size=3)
    val_mean_error_smooth = uniform_filter1d(val_mean_error, size=3)
    
    plt.plot(time_axis, train_mean_error, label='Train')
    plt.plot(time_axis, val_mean_error, label='Val')
    plt.fill_between(time_axis, train_mean_error - train_std_error, train_mean_error + train_std_error, color='blue', alpha=0.1)
    plt.fill_between(time_axis, val_mean_error - val_std_error, val_mean_error + val_std_error, color='brown', alpha=0.1)
    plt.xlabel('Time [s]')
    plt.ylabel('Error [N]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Predict_error_over_time.png'), dpi=300)
    plt.close()