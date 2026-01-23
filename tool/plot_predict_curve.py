import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()



def plot_results(predictions, ground_truth, times, pred_steps=1, save_dir='results', sampling_rate=10):

    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(pred_steps, 1, figsize=(10, 6 + 4 * pred_steps))
    
    if pred_steps == 1:
        axs = [axs]  

    # Only display the first 100 samples to keep the chart clear.
    sample_size = min(100, len(predictions))
    
    time_in_seconds = times / sampling_rate
    
    if pred_steps == 1 or predictions.shape[1] == 1:
        axs[0 if len(axs) == 1 else 1].plot(time_in_seconds, ground_truth[:sample_size].flatten(), 'b-', label='True Value')
        axs[0 if len(axs) == 1 else 1].plot(time_in_seconds, predictions[:sample_size].flatten(), 'r-', label='Predicted Value')
        axs[0 if len(axs) == 1 else 1].set_xlabel('Time (s)')
        axs[0 if len(axs) == 1 else 1].set_ylabel('Tactile force value')
        axs[0 if len(axs) == 1 else 1].set_title('Comparison of predicted values with actual values')
        axs[0 if len(axs) == 1 else 1].legend()
        axs[0 if len(axs) == 1 else 1].grid(True)
    else:
        for i in range(pred_steps):
            '''
                ground_truth shape [batch×num, 3] 
                ground_truth[:sample_size, i]
            '''
            axs[i+1].plot(time_in_seconds, ground_truth[:sample_size, i], 'b-', label='True Value')
            axs[i+1].plot(time_in_seconds, predictions[:sample_size, i], 'r-', label=f'The {i+1}th step prediction')
            axs[i+1].set_xlabel('Time (s)')
            axs[i+1].set_ylabel('Tactile force value')
            axs[i+1].set_title(f'Comparison of predicted value and actual value in step {i+1}')
            axs[i+1].legend()
            axs[i+1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tactile_prediction_results.png'))
    plt.ioff()
    plt.show()