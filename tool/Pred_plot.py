import numpy as np
import torch
import matplotlib.pyplot as plt

def visual(true, preds, save_dir):
    plt.ylim(-1.0, 1.0)  
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, linestyle='-')
    plt.legend()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()



def visual_mean(gt_mean, gt_std, pd_mean, pd_std, save_dir , sampling_rate=1/32):

    x = np.arange(gt_mean.shape[0])  * sampling_rate

    plt.figure(figsize=(8, 5), dpi=400)
    plt.plot(x, gt_mean, color='brown', label='Ground Truth')
    plt.fill_between(x, gt_mean - gt_std, gt_mean + gt_std, color='orange', alpha=0.3)

    plt.plot(x, pd_mean, color='blue', label='Prediction')
    plt.fill_between(x, pd_mean - pd_std, pd_mean + pd_std, color='skyblue', alpha=0.3)

    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Prediction vs Ground Truth (mean ± std)')
    plt.savefig(save_dir + '/'+ 'pred_plot_mean' + '.png', bbox_inches='tight')
    plt.close()
    
    