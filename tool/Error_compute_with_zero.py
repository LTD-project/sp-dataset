import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_error_over_time_with_zero(model, data_loader, device, sampling_interval=1/32):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for sequences, targets, _ in data_loader:
            sequences = sequences.float().to(device)
            targets = targets.float().to(device)
            pred_len = targets.shape[1]
            outputs = outputs[:, -pred_len:, -1]
            error = torch.abs(outputs - targets)  
            all_errors.append(error.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  
    mean_error = np.mean(all_errors, axis=0)         
    std_error = np.std(all_errors, axis=0)           
    time_axis = np.arange(all_errors.shape[1]) * sampling_interval  

    # 在开头插入0点
    time_axis = np.insert(time_axis, 0, 0.0)
    mean_error = np.insert(mean_error, 0, 0.0)
    std_error = np.insert(std_error, 0, 0.0)
    
    return time_axis, mean_error, std_error

def plot_error_with_zero(time_axis, train_mean_error, train_std_error, val_mean_error, val_std_error, save_dir):
    plt.ylim(0, 1.0)
    plt.plot(time_axis, train_mean_error, label='Train')
    plt.plot(time_axis, val_mean_error, label='Val')
    plt.fill_between(time_axis, train_mean_error - train_std_error, train_mean_error + train_std_error, color='blue', alpha=0.1)
    plt.fill_between(time_axis, val_mean_error - val_std_error, val_mean_error + val_std_error, color='brown', alpha=0.1)
    plt.xlabel('Time [s]')
    plt.ylabel('Error [N]')
    plt.title('Average Total Force Prediction Error')
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.3))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Predict_error_over_time_with_zero.png'))
    plt.show() 