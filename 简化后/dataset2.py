import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TactileDataAugmenter:
    def __init__(self, noise_level=0.01, scaling_sigma=0.1):
        self.noise_level = noise_level
        self.scaling_sigma = scaling_sigma

    def apply_augmentation(self, data):
        aug_type = np.random.choice(["noise", "scaling", "none"])
        if aug_type == "noise":
            return data + np.random.normal(0, self.noise_level, data.shape)
        if aug_type == "scaling":
            return data * np.random.normal(1, self.scaling_sigma)
        return data


class TactileDataNormalizer:
    def __init__(self, normalization_type="global"):
        self.mode = normalization_type  # "global" or "per_object"
        self.scalers = {"features": {}, "target": {}}
        self.is_fitted = False

    def _get_scaler(self, key, obj_type):
        s_dict = self.scalers[key]
        return s_dict["global"] if self.mode == "global" else s_dict.get(obj_type)

    def fit(self, data_dict):
        """data_dict: {obj_type: [file_paths]}"""
        all_feat, all_targ = [], []
        
        for obj_type, files in data_dict.items():
            obj_feat = []
            for f in files:
                df = pd.read_csv(f)
                obj_feat.append(df.iloc[:, -3:].values)
            
            if not obj_feat: continue
            
            feats = np.vstack(obj_feat)
            targs = feats[:, -1:].copy() 
            
            if self.mode == "per_object":
                self.scalers["features"][obj_type] = StandardScaler().fit(feats)
                self.scalers["target"][obj_type] = StandardScaler().fit(targs)
            
            all_feat.append(feats)
            all_targ.append(targs)

        if self.mode == "global" and all_feat:
            self.scalers["features"]["global"] = StandardScaler().fit(np.vstack(all_feat))
            self.scalers["target"]["global"] = StandardScaler().fit(np.vstack(all_targ))
        
        self.is_fitted = True

    def transform(self, data, obj_type, is_target=False) :
        key = "target" if is_target else "features"
        scaler = self._get_scaler(key, obj_type)
        if not scaler: raise ValueError(f"Scaler not found for {obj_type}")
        
        if is_target and data.ndim == 1:
            return scaler.transform(data.reshape(-1, 1)).flatten()
        return scaler.transform(data)

    def save_params(self, path):
        data = {k: {obj: {"m": s.mean_.tolist(), "s": s.scale_.tolist()} 
                for obj, s in v.items()} for k, v in self.scalers.items()}
        with open(path, 'w') as f: json.dump({"mode": self.mode, "data": data}, f)

class TactileForceDataset(Dataset):
    def __init__(self, file_map, seq_len=96, pred_len=96, 
                 normalizer=None, augmenter=None, aug_prob=0.5):
        self.samples = []
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.aug_prob = aug_prob

        for obj_type, files in file_map.items():
            for f in files:
                df = pd.read_csv(f)
                feat = df.iloc[:, -3:].values
                target = df.iloc[:, -1].values
                
                for i in range(len(df) - seq_len - pred_len + 1):
                    x = feat[i : i + seq_len]
                    y = target[i + seq_len : i + seq_len + pred_len]
                    self.samples.append({"x": x, "y": y, "obj": obj_type})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x, y, obj = s['x'].copy(), s['y'].copy(), s['obj']

        if self.augmenter and np.random.random() < self.aug_prob:
            x = self.augmenter.apply_augmentation(x)
        
        if self.normalizer:
            x = self.normalizer.transform(x, obj)
            y = self.normalizer.transform(y, obj, is_target=True)

        return torch.FloatTensor(x), torch.FloatTensor(y), obj

def get_file_map(data_root, object_types = None):
    root = Path(data_root)
    if object_types is None:
        object_types = [d.name for d in root.iterdir() if d.is_dir()]
    
    return {obj: sorted(list((root/obj).glob("*.csv"))) for obj in object_types}

def create_loaders(data_root, held_out_trial, batch_size=32, **kwargs):
    all_files = get_file_map(data_root)
    
    train_files, val_files, test_files = {}, {}, {}
    held_obj, held_csv = held_out_trial
    
    for obj, files in all_files.items():
        if obj == held_obj:
            test_files[obj] = [f for f in files if f.name == held_csv]
            rem = [f for f in files if f.name != held_csv]
            val_files[obj] = [rem[-1]] if rem else []
            train_files[obj] = rem[:-1] if rem else []
        else:
            train_files[obj] = files

    # Normalizer Fit
    norm = TactileDataNormalizer(normalization_type=kwargs.get("normalization_type", "global"))
    norm.fit(train_files)

    # Dataset & Loader
    ds_args = {"seq_len": kwargs.get("seq_len", 96), "pred_len": kwargs.get("pred_len", 96), "normalizer": norm}
    
    train_ds = TactileForceDataset(train_files, augmenter=TactileDataAugmenter(), **ds_args)
    val_ds = TactileForceDataset(val_files, **ds_args)
    test_ds = TactileForceDataset(test_files, **ds_args)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        norm
    )
    

if __name__ == "__main__":

    DATA_ROOT = "Processed_Finally_data3" 
  
    all_files_dict = get_file_map(DATA_ROOT)
    print(all_files_dict)
    
    for obj, files in all_files_dict.items():
        print(f"  - {obj}: {len(files)} 个 trial 文件")

    # 3. 自动选取一个 Trial 用于 Leave-one-trial-out 测试
    # 我们选取第一个物体中的第一个文件作为测试集
    first_obj = list(all_files_dict.keys())[0]
    first_file_name = all_files_dict[first_obj][0].name
    held_out_trial = (first_obj, first_file_name)
    
    print(f"\n--- 测试 Leave-one-trial-out 模式 ---")
    print(f"排除(作为测试集)的 Trial: 物体={held_out_trial[0]}, 文件={held_out_trial[1]}")

    # 4. 创建 DataLoaders
    try:
        train_loader, val_loader, test_loader, normalizer = create_loaders(
            data_root=DATA_ROOT,
            held_out_trial=held_out_trial,
            batch_size=32,
            seq_len=96,
            pred_len=96,
            normalization_type="global" # or "per_object"
        )

        # 5. 验证数据加载情况
        print(f"\n加载完成:")
        print(f"  训练集样本数: {len(train_loader.dataset)}")
        print(f"  验证集样本数: {len(val_loader.dataset)}")
        print(f"  测试集样本数: {len(test_loader.dataset)}")

        # 6. 检查数据 Tensor 维度
        # 获取第一个 Batch
        x_batch, y_batch, obj_names = next(iter(train_loader))
        
        print(f"\n--- Batch 数据检查 ---")
        print(f"输入 X (特征) 形状: {x_batch.shape}")   # 预期: [32, 96, 3]
        print(f"输出 Y (目标) 形状: {y_batch.shape}")   # 预期: [32, 96]
        print(f"物体标签示例: {obj_names[:3]}")

        # 7. 测试归一化是否正常工作
        # 检查均值是否接近 0 (针对第一个特征)
        print(f"\n--- 归一化效果验证 ---")
        print(f"Batch 特征均值: {x_batch.mean().item():.4f} (理想情况应接近0)")
        
        # 8. 保存归一化参数到本地
        norm_save_path = "normalizer_params.json"
        normalizer.save_params(norm_save_path)
        print(f"\n归一化参数已保存至: {norm_save_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n运行出错: {e}")