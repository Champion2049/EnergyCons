import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class InMemoryKoreaDataset(Dataset):
    def __init__(
        self,
        path,
        buildings,
        appliance,
        windowsize=496,
        active_threshold=15.0,
        active_ratio=None,
        active_oversample=None,
        normalization_enabled=False,
        transform_params=None,
    ):
        super().__init__()

        self.normalization_enabled = normalization_enabled
        self.appliance = appliance
        self.windowsize = windowsize
        self.active_threshold = active_threshold if active_threshold is not None else 0.0

        self.data_raw = [] # To store loaded raw dataframes
        self.datamap = {} # Maps global index to (dataframe_index, window_start_index)

        # Filter filenames based on the provided buildings list
        all_filenames = os.listdir(path)
        relevant_filenames = [
            f for f in all_filenames
            if any(f.startswith(b) and f.endswith('.csv') for b in buildings)
        ]

        if not relevant_filenames:
            raise ValueError(f"No CSV files found in '{path}' matching prefixes in '{buildings}'. Please check path and filenames.")

        columns_to_use = ["main", self.appliance]

        print(f"Loading {len(relevant_filenames)} relevant CSV files...")
        for filename in relevant_filenames:
            filepath = os.path.join(path, filename)
            try:
                df = pd.read_csv(filepath, usecols=columns_to_use, sep=",")
                
                # --- IMPORTANT CHANGE: Robust NaN handling ---
                initial_rows = len(df)
                df = df.dropna() # Drop rows with any NaN values
                if len(df) < initial_rows:
                    print(f"  Warning: Removed {initial_rows - len(df)} rows with NaN values from {filename}")
                # --- End Important Change ---

                if "main" not in df.columns or self.appliance not in df.columns:
                    print(f"Warning: Skipping {filename}. Missing 'main' or '{self.appliance}' column after NaN removal.")
                    continue

                if df.empty:
                    print(f"Warning: Skipping {filename}. No data left after processing or CSV was empty.")
                    continue

                self.data_raw.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}. Skipping this file.")
                continue

        if not self.data_raw:
            raise ValueError("No valid dataframes were loaded. Check your CSV file content and names.")

        data_idx_counter = 0
        window_idx_counter = 0
        for df_idx, subseq_df in enumerate(self.data_raw):
            n_windows = subseq_df.shape[0] - windowsize + 1
            if n_windows <= 0:
                print(f"Warning: Dataframe {df_idx} (from file {relevant_filenames[df_idx]}) has {subseq_df.shape[0]} samples, which is less than windowsize {windowsize}. Skipping.")
                continue

            self.datamap.update(
                {window_idx_counter + i: (df_idx, i) for i in range(n_windows)}
            )
            data_idx_counter += 1
            window_idx_counter += n_windows

        self.total_size = window_idx_counter

        if self.total_size == 0:
            raise ValueError("No data windows could be created from the loaded CSVs. Check window size and data length.")

        # Determine standardization parameters
        if self.normalization_enabled:
            if transform_params:
                self.sample_mean = transform_params["sample_mean"]
                self.sample_std = transform_params["sample_std"]
                self.target_mean = transform_params["target_mean"]
                self.target_std = transform_params["target_std"]
            else:
                all_mains_data = np.concatenate([df['main'].values for df in self.data_raw])
                all_appliance_data = np.concatenate([df[self.appliance].values for df in self.data_raw])
                self.sample_mean = np.mean(all_mains_data)
                self.sample_std = np.std(all_mains_data)
                self.target_mean = np.mean(all_appliance_data)
                self.target_std = np.std(all_appliance_data)
                if self.sample_std == 0: self.sample_std = 1.0
                if self.target_std == 0: self.target_std = 1.0
        else:
            self.sample_mean, self.sample_std = 0.0, 1.0
            self.target_mean, self.target_std = 0.0, 1.0

        self.transform_params = {
            "sample_mean": self.sample_mean,
            "sample_std": self.sample_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        df_idx, window_start_idx = self.datamap[idx]

        raw_sample = self.data_raw[df_idx].iloc[window_start_idx : window_start_idx + self.windowsize]["main"]
        raw_target_full_window = self.data_raw[df_idx].iloc[window_start_idx : window_start_idx + self.windowsize][self.appliance]

        sample = (raw_sample - self.sample_mean) / self.sample_std
        target = (raw_target_full_window.mean() - self.target_mean) / self.target_std

        classification = torch.tensor(
            (raw_target_full_window > self.active_threshold).any().astype(int), dtype=torch.float32
        ).detach().cpu()

        return (
            torch.tensor(sample.values, dtype=torch.float32).detach().cpu(),
            torch.tensor(target, dtype=torch.float32).detach().cpu(),
            classification,
        )
