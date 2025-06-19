import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import torch

import gluonts.time_feature

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


class PatchingFeature(FeatureGenerator):
    def __init__(self, window_fraction: float = 0.2) -> None:
        
        if not (0 < window_fraction < 1):
            raise ValueError("window_fraction must be between 0 and 1 (exclusive).")
        
        self.window_fraction = window_fraction

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # X_torch = torch.from_numpy(df.xs(0, level="item_id")["target"].to_numpy())
        X_torch = torch.from_numpy(df["target"].to_numpy())
        window_size = int(self.window_fraction * X_torch.shape[0])
        X_transformed = create_lag_matrix(X_torch, window_size=window_size)
        
        for i in range(X_transformed.shape[1]):
            df[f"lag_{i}"] = X_transformed[:, i].numpy()
        return df



def create_lag_matrix(series: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Build lag features for a 1D time series.

    Each row i contains values from indices (i - 2*window_size + 1) to (i - window_size),
    padded with NaN where indices are out of bounds.

    Args:
        series (torch.Tensor): 1D tensor of shape (T,) or (T, 1).
        window_size (int): Number of lag steps.

    Returns:
        torch.Tensor: Tensor of shape (T, window_size) with lagged features.
    """
    # Ensure a flat 1D tensor
    if series.ndim == 2 and series.size(1) == 1:
        series = series.squeeze(1)

    # Create a tensor of shape (length, window_size) filled with NaN
    length = series.size(0)
    lags = torch.full(
        (length, window_size),
        float("nan"),
        dtype=series.dtype,
        device=series.device,
    )

    # Fill the tensor with the lagged window values
    for idx in range(length):
        start = idx - 2 * window_size + 1
        end = idx - window_size

        if end >= 0:
            valid_start = max(start, 0)
            window_vals = series[valid_start : end + 1]

            pad = window_size - window_vals.size(0)
            if pad > 0:
                lags[idx, pad:] = window_vals
            else:
                lags[idx] = window_vals

    return lags