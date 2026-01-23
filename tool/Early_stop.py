import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), path)
            
class AdvancedEarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, min_epochs=30, window=3, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.window = window
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_losses = []
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, path, epoch):
        self.val_losses.append(val_loss)
        
        if epoch < self.min_epochs:
            if self.verbose:
                print(f'Early stopping not active before epoch {self.min_epochs}')
            return
            
        if len(self.val_losses) >= self.window:
            smoothed_loss = np.mean(self.val_losses[-self.window:])
        else:
            smoothed_loss = val_loss
            
        if self.best_loss is None:
            self.best_loss = smoothed_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), path)
        elif smoothed_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered at epoch {epoch}, best epoch was {self.best_epoch}')
        else:
            self.best_loss = smoothed_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), path)
            if self.verbose:
                print(f'New best model saved at epoch {epoch} with loss {smoothed_loss:.4f}')