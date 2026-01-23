import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def plot_prediction_with_mean_std(predictions, ground_truth, times, pred_steps=1, save_dir='results', sampling_rate=10, title=None, object_type=''):

    os.makedirs(save_dir, exist_ok=True)
    
    time_in_seconds = times / sampling_rate
    
    fig, ax = plt.subplots(figsize=(10, 6))

    is_multi_step = len(predictions.shape) == 3 or (len(predictions.shape) == 2 and pred_steps > 1)
    

    if is_multi_step:
        if len(predictions.shape) == 3:  # [sample, time_step, pred_len]
            pred_mean = np.mean(predictions, axis=0)  # [time_step, pred_len]
            pred_std = np.std(predictions, axis=0)
            gt_mean = np.mean(ground_truth, axis=0)
            gt_std = np.std(ground_truth, axis=0)
        
            ax.plot(time_in_seconds, gt_mean[:, 0], 'b-', linewidth=2, label='Average actual values')
            ax.fill_between(time_in_seconds, gt_mean[:, 0] - gt_std[:, 0], gt_mean[:, 0] + gt_std[:, 0], 
                           color='blue', alpha=0.2, label='Standard deviation of the true value')
            
            colors = ['r', 'g', 'orange', 'purple', 'brown']
            for i in range(min(pred_steps, pred_mean.shape[1])):
                color = colors[i % len(colors)]
                label = f'The average value predicted in step {i+1}'
                std_label = f'The predicted standard deviation in step {i+1}'
                
                ax.plot(time_in_seconds, pred_mean[:, i], color=color, linewidth=2, label=label)
                ax.fill_between(time_in_seconds, pred_mean[:, i] - pred_std[:, i], 
                               pred_mean[:, i] + pred_std[:, i], 
                               color=color, alpha=0.2, label=std_label)
        else:  
            pred_mean = np.mean(predictions, axis=0)
            pred_std = np.std(predictions, axis=0)
            gt_mean = np.mean(ground_truth, axis=0)
            gt_std = np.std(ground_truth, axis=0)
            
            # Plot true value
            ax.plot(time_in_seconds, gt_mean, 'b-', linewidth=2, label='Average actual values')
            ax.fill_between(time_in_seconds, gt_mean - gt_std, gt_mean + gt_std, 
                           color='blue', alpha=0.2, label='Standard deviation of the true value')
            
            # Plot predicted value
            ax.plot(time_in_seconds, pred_mean, 'r-', linewidth=2, label='Predicted average value')
            ax.fill_between(time_in_seconds, pred_mean - pred_std, pred_mean + pred_std, 
                           color='red', alpha=0.2, label='Prediction standard deviation')
    else:
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        gt_mean = np.mean(ground_truth, axis=0)
        gt_std = np.std(ground_truth, axis=0)
        
        # Plot true value
        ax.plot(time_in_seconds, gt_mean, 'b-', linewidth=2, label='Average actual values')
        ax.fill_between(time_in_seconds, gt_mean - gt_std, gt_mean + gt_std, 
                       color='blue', alpha=0.2, label='Standard deviation of the true valu')
        
        # Plot predicted value
        ax.plot(time_in_seconds, pred_mean, 'r-', linewidth=2, label='Predicted average valu')
        ax.fill_between(time_in_seconds, pred_mean - pred_std, pred_mean + pred_std, 
                       color='red', alpha=0.2, label='Prediction standard deviation')
 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tactile force value')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{object_type} tactile prediction results - Mean value and standard deviation')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    

    ax.legend(loc='best')
    
    file_name = f'{object_type}_prediction_mean_std.png' if object_type else 'prediction_mean_std.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name), dpi=400)
    plt.show()
