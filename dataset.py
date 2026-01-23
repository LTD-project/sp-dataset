import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union, Literal, Dict
from sklearn.preprocessing import StandardScaler
import torch
from scipy.interpolate import interp1d
from scipy.signal import resample
import json


class TactileDataAugmenter:
    def __init__(
        self,
        noise_level = 0.01,
        scaling_sigma = 0.1,
    ):
        """
        Parameters:
            noise_level: The standard deviation of Gaussian noise
            scaling_sigma: The intensity of the scaling transformation
        """
        self.noise_level = noise_level
        self.scaling_sigma = scaling_sigma

    def add_gaussian_noise(self, data):
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise

    def scaling(self, data):
        scaling_factor = np.random.normal(1, self.scaling_sigma)
        return data * scaling_factor


    def apply_augmentation(self, data, augmentation_types = None):
        available_augmentations = {
            "noise": self.add_gaussian_noise,
            "scaling": self.scaling
        }

        if augmentation_types is None:
            num_augments = np.random.randint(1, 2)
            augmentation_types = np.random.choice(
                list(available_augmentations.keys()),
                size=num_augments,
                replace=False,
            )
        augmented_data = data.copy()
        for aug_type in augmentation_types:
            if aug_type in available_augmentations:
                augmented_data = available_augmentations[aug_type](augmented_data)

        return augmented_data


class TactileDataNormalizer:
    """
        'global': Perform unified normalization on the entire dataset
        'per_object': Perform normalization separately for each type of object
    """
    def __init__(self, normalization_type = "global"):
        self.normalization_type = normalization_type
        self.global_scaler_features = StandardScaler()
        self.global_scaler_target = StandardScaler()
        self.object_scalers_features = {}
        self.object_scalers_target = {}
        self.is_fitted = False

    def fit(self, data_root, object_type = None, selected_files = None):
        if object_type is None:
            object_types = [
                d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
            ]
        elif isinstance(object_type, str):
            object_types = [object_type]
        else:
            object_types = object_type

        all_features = []
        all_targets = []

        for obj_type in object_types:
            obj_dir = os.path.join(data_root, obj_type)
            if not os.path.isdir(obj_dir):
                continue
            if selected_files is not None and obj_type in selected_files:
                csv_files = selected_files[obj_type]
            elif selected_files is not None:
                continue
            else:
                csv_files = [f for f in os.listdir(obj_dir) if f.endswith(".csv")]

            obj_features = []
            obj_targets = []

            for file_name in csv_files:
                file_path = os.path.join(obj_dir, file_name)
                if not os.path.exists(file_path):
                    continue
                df = pd.read_csv(file_path)
                features = df.iloc[:, -3:].values
                target = df.iloc[:, -1].values.reshape(-1, 1)
                obj_features.append(features)
                obj_targets.append(target)

            if obj_features:
                obj_features = np.vstack(obj_features)
                obj_targets = np.vstack(obj_targets)

                if self.normalization_type == "per_object":
                    self.object_scalers_features[obj_type] = StandardScaler().fit(obj_features)
                    self.object_scalers_target[obj_type] = StandardScaler().fit(obj_targets)

                all_features.append(obj_features)
                all_targets.append(obj_targets)

        if all_features:
            all_features = np.vstack(all_features)
            all_targets = np.vstack(all_targets)

            if self.normalization_type == "global":
                self.global_scaler_features.fit(all_features)
                self.global_scaler_target.fit(all_targets)

        self.is_fitted = True

    def transform_features(self, data, object_type):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.normalization_type == "global":
            return self.global_scaler_features.transform(data)
        else:
            if object_type not in self.object_scalers_features:
                raise ValueError(f"No scaler found for object type: {object_type}")
            return self.object_scalers_features[object_type].transform(data)

    def transform_target(self, data, object_type):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if self.normalization_type == "global":
            return self.global_scaler_target.transform(data)
        else:
            if object_type not in self.object_scalers_target:
                raise ValueError(f"No scaler found for object type: {object_type}")
            return self.object_scalers_target[object_type].transform(data)

    def save_normalizer_params(self, save_path):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before saving parameters")

        params = {}
        if self.normalization_type == "global":
            params = {
                "normalization_type": "global",
                "features": {
                    "mean": self.global_scaler_features.mean_.tolist(),
                    "std": self.global_scaler_features.scale_.tolist(),
                },
                "target": {
                    "mean": self.global_scaler_target.mean_.tolist(),
                    "std": self.global_scaler_target.scale_.tolist(),
                },
            }
        else:
            params = {
                "normalization_type": "per_object",
                "objects": {},
            }
            for obj_type in self.object_scalers_features.keys():
                params["objects"][obj_type] = {
                    "features": {
                        "mean": self.object_scalers_features[obj_type].mean_.tolist(),
                        "std": self.object_scalers_features[obj_type].scale_.tolist(),
                    },
                    "target": {
                        "mean": self.object_scalers_target[obj_type].mean_.tolist(),
                        "std": self.object_scalers_target[obj_type].scale_.tolist(),
                    },
                }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(params, f, indent=4)

    def load_normalizer_params(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot find the normalization parameter file: {load_path}")

        with open(load_path, "r") as f:
            params = json.load(f)

        if params["normalization_type"] != self.normalization_type:
            raise ValueError(
                f"Normalization type mismatch: In the file, it is{params['normalization_type']}, but currently it is {self.normalization_type}"
            )

        if self.normalization_type == "global":
            self.global_scaler_features.mean_ = np.array(params["features"]["mean"])
            self.global_scaler_features.scale_ = np.array(params["features"]["std"])
            self.global_scaler_target.mean_ = np.array(params["target"]["mean"])
            self.global_scaler_target.scale_ = np.array(params["target"]["std"])
        else:
            for obj_type, obj_params in params["objects"].items():
                self.object_scalers_features[obj_type] = StandardScaler()
                self.object_scalers_features[obj_type].mean_ = np.array(
                    obj_params["features"]["mean"]
                )
                self.object_scalers_features[obj_type].scale_ = np.array(
                    obj_params["features"]["std"]
                )

                self.object_scalers_target[obj_type] = StandardScaler()
                self.object_scalers_target[obj_type].mean_ = np.array(
                    obj_params["target"]["mean"]
                )
                self.object_scalers_target[obj_type].scale_ = np.array(
                    obj_params["target"]["std"]
                )

        self.is_fitted = True

class TactileForceDataset(Dataset):
    def __init__(
        self,
        data_root,
        sequence_length = 96,
        prediction_length = 96,
        label_length = 48,
        split = "train",  # ["train", "val", "test"]
        train_ratio = 0.7,
        val_ratio = 0.15,
        object_type = None,
        normalizer = None,
        augmenter = None,
        augmentation_prob = 0.5,
        selected_files = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.label_length = label_length
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
        self.selected_files = selected_files

        if object_type is None:
            self.object_types = [
                d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
            ]
        elif isinstance(object_type, str):
            self.object_types = [object_type]
        elif isinstance(object_type, list):
            self.object_types = object_type
        else:
            raise ValueError("object_type must be None, string, or list of strings")

        for obj_type in self.object_types:
            if not os.path.isdir(os.path.join(data_root, obj_type)):
                raise ValueError(f"Object type '{obj_type}' not found in {data_root}")

        self.sequences = []
        self.object_labels = []

        for obj_type in self.object_types:
            if self.selected_files is not None:
                files = self.selected_files.get(obj_type, [])
                if not files:
                    continue
                self._load_sequences_from_specific_files(obj_type, files)
            else:
                self._load_and_split_object_sequences(obj_type, train_ratio, val_ratio, split)

    def _load_sequences_from_specific_files(self, object_type, files):
        object_dir = os.path.join(self.data_root, object_type)
        for csv_file in files:
            file_path = os.path.join(object_dir, csv_file)
            if not os.path.exists(file_path):
                continue
            self._append_sequences_from_file(object_type, file_path)

    def _append_sequences_from_file(self, object_type, file_path):
        df = pd.read_csv(file_path)
        features = df.iloc[:, -3:].values  # shape: [T, 3]
        target = df.iloc[:, -1].values     # shape: [T,]

        for i in range(len(df) - self.sequence_length - self.prediction_length + 1):
            x = features[i : i + self.sequence_length].copy() 
            y = target[i + self.sequence_length : i + self.sequence_length + self.prediction_length].copy()

            self.sequences.append((x, y)) 
            self.object_labels.append(object_type)

    def _load_and_split_object_sequences(self, object_type, train_ratio, val_ratio, split):
        object_dir = os.path.join(self.data_root, object_type)
        csv_files = [f for f in os.listdir(object_dir) if f.endswith(".csv")]
        n_files = len(csv_files)
        n_train_files = int(n_files * train_ratio)
        n_val_files = int(n_files * val_ratio)

        if split == "train":
            selected_files = csv_files[:n_train_files]
        elif split == "val":
            selected_files = csv_files[n_train_files : n_train_files + n_val_files]
        else:
            selected_files = csv_files[n_train_files + n_val_files :]

        self._load_sequences_from_specific_files(object_type, selected_files)

    def __len__(self):       
        return len(self.sequences)

    def __getitem__(self, idx):       # return (x, y, obj_type)
        x, y = self.sequences[idx]  
        obj_type = self.object_labels[idx]

        if self.augmenter is not None and np.random.random() < self.augmentation_prob:
            x = self.augmenter.apply_augmentation(x)

        if self.normalizer is not None:
            x = self.normalizer.transform_features(x, obj_type)
            y_normalized = self.normalizer.transform_target(y, obj_type).flatten()
            y = y_normalized

        return x, y, obj_type


def list_all_trials(data_root, object_type = None):
    """list all trial (object, csv_file)."""
    if object_type is None:
        object_types = [
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ]
    elif isinstance(object_type, str):
        object_types = [object_type]
    else:
        object_types = object_type

    trials = []
    for obj_type in object_types:
        obj_dir = os.path.join(data_root, obj_type)
        if not os.path.isdir(obj_dir):
            continue
        csv_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".csv")])
        for csv_file in csv_files:
            trials.append((obj_type, csv_file))
    return trials


def _build_trial_mapping(data_root, object_types):
    mapping = {}
    for obj_type in object_types:
        obj_dir = os.path.join(data_root, obj_type)
        if not os.path.isdir(obj_dir):
            continue
        csv_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".csv")])
        mapping[obj_type] = csv_files
    return mapping


def create_leave_one_trial_out_loaders(
    data_root,
    held_out_trial,
    val_trial = None,
    batch_size = 32,
    sequence_length = 96,
    label_length = 48,
    prediction_length = 96,
    object_type = None,
    num_workers = 0,
    augmentation_params = None,
    normalization_type = "global",
):
    if object_type is None:
        object_types = [
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ]
    elif isinstance(object_type, str):
        object_types = [object_type]
    else:
        object_types = object_type

    trial_mapping = _build_trial_mapping(data_root, object_types)
    held_out_obj, held_out_file = held_out_trial

    if held_out_obj not in trial_mapping or held_out_file not in trial_mapping[held_out_obj]:
        raise ValueError(f"Held-out trial {held_out_trial} dose not exist in the dataset")

    train_files = {
        obj: [
            f
            for f in files
            if not (obj == held_out_obj and f == held_out_file)
            and not (val_trial and obj == val_trial[0] and f == val_trial[1])
        ]
        for obj, files in trial_mapping.items()
    }

    if val_trial is None:
        for obj, files in train_files.items():
            if files:
                derived_file = files[-1]
                train_files[obj] = files[:-1]
                val_trial = (obj, derived_file)
                break
        if val_trial is None:
            raise ValueError("There are not enough trials available to divide the validation set.")

    val_obj, val_file = val_trial
    if val_obj not in trial_mapping or val_file not in trial_mapping[val_obj]:
        raise ValueError(f"val trial {val_trial} dose not exist in the dataset中")
    if val_obj == held_out_obj and val_file == held_out_file:
        raise ValueError("val trial cannot be the same as the test trial")

    val_files = {val_obj: [val_file]}
    test_files = {held_out_obj: [held_out_file]}

    train_files = {obj: files for obj, files in train_files.items() if files}
    if not any(train_files.values()):
        raise ValueError("no trial available for training.")

    normalizer = TactileDataNormalizer(normalization_type=normalization_type)
    normalizer.fit(data_root, object_types, selected_files=train_files)

    augmenter = None
    if augmentation_params is not None:
        augmenter = TactileDataAugmenter(**augmentation_params)

    train_dataset = TactileForceDataset(
        data_root=data_root,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        label_length=label_length,
        split="train",
        object_type=object_types,
        normalizer=normalizer,
        augmenter=augmenter,
        augmentation_prob=0.5,
        selected_files=train_files,
    )
    val_dataset = TactileForceDataset(
        data_root=data_root,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        label_length=label_length,
        split="val",
        object_type=object_types,
        normalizer=normalizer,
        augmenter=None,
        selected_files=val_files,
    )
    test_dataset = TactileForceDataset(
        data_root=data_root,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        label_length=label_length,
        split="test",
        object_type=object_types,
        normalizer=normalizer,
        augmenter=None,
        selected_files=test_files,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(
        f"Leave-one-trial-out: test trial={held_out_trial}, val trial={val_trial}, "
        f"train sample={len(train_dataset)}, val sample={len(val_dataset)}, test sample={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, normalizer

if __name__ == "__main__":
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
        print(11, trials)

        results_root = "results_leave_one_trial_out"
        os.makedirs(results_root, exist_ok=True)
        
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
