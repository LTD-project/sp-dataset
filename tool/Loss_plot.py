import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_loss_data, val_loss_data, save_dir):
    x = list(range(1, len(train_loss_data) + 1))  
    plt.figure(figsize=(10, 5), dpi=400)
    plt.plot(x, train_loss_data, label='Train Loss')
    plt.plot(x, val_loss_data, label='Val Loss')
    plt.title('Train and Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + '/'+ 'loss_curve' + '.png')  
    plt.close()  
    