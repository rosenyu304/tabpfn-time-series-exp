import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import torch

from typing import List, Optional, Tuple, Literal
import gluonts.time_feature

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)
from tabpfn_time_series.features.auto_features import detrend

class PolynomialTrendFeature(FeatureGenerator):
    def __init__(self, 
                 z_scaling: bool = False, 
                 minmax_scaling: bool = False, 
                 do_detrend: bool = False, 
                 pre_detrend: bool = False, 
                 constant_scale: float = 1.0,
                 detrend_type: Literal["first_diff", "loess", "linear", "constant"] = (
                                    "first_diff"
                                ),
                 degree: List[int] = [5, 10, 15, 20]) -> None:
        
        self.z_scaling = z_scaling
        self.minmax_scaling = minmax_scaling
        self.do_detrend = do_detrend
        self.pre_detrend = pre_detrend
        self.degree = degree
        self.constant_scale = constant_scale
        self.detrend_type = detrend_type

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # get rid of the nan values
        values = np.array(df.target)
        values = values[~np.isnan(values)]
        
        if self.pre_detrend:
            values = detrend(values, self.detrend_type)
        
        for degree in self.degree:
            coeff_polynomial = np.polyfit(x=np.arange(len(values)), y=values, deg=degree)
            p = np.poly1d(coeff_polynomial)
            
            polynomial_trend = p(np.arange(len(np.array(df.target))))
            
            if self.constant_scale:
                polynomial_trend = polynomial_trend * self.constant_scale
        
            if self.do_detrend and self.z_scaling:
                polynomial_trend = detrend(polynomial_trend, self.detrend_type)
                df[f'polynomial_trend_{degree}'] = (polynomial_trend - polynomial_trend.mean())/(polynomial_trend.std())
            elif self.do_detrend and self.minmax_scaling:
                polynomial_trend = detrend(polynomial_trend, self.detrend_type)
                df[f'polynomial_trend_{degree}'] = 2*(polynomial_trend - polynomial_trend.min())/(polynomial_trend.max() - polynomial_trend.min()) - 1
            elif self.z_scaling:
                df[f'polynomial_trend_{degree}'] = (polynomial_trend - polynomial_trend.mean())/(polynomial_trend.std())
            elif self.minmax_scaling:
                df[f'polynomial_trend_{degree}'] = 2*(polynomial_trend - polynomial_trend.min())/(polynomial_trend.max() - polynomial_trend.min()) - 1
            elif self.do_detrend:
                df[f'polynomial_trend_{degree}'] = detrend(polynomial_trend, self.detrend_type)
            else:
                df[f'polynomial_trend_{degree}'] = polynomial_trend
            
        return df


class LinearTrendFeature(FeatureGenerator):
    def __init__(self, z_scaling: bool = True, do_detrend: bool = True) -> None:
        self.z_scaling = z_scaling
        self.do_detrend = do_detrend

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        # get rid of the nan values
        values = np.array(df.target)
        values = values[~np.isnan(values)]
        
        coeff_linear = np.polyfit(x=np.arange(len(values)), y=values, deg=1)
        p = np.poly1d(coeff_linear)
        
        linear_trend = p(np.arange(len(np.array(df.target))))
        
        if self.do_detrend and self.z_scaling:
            linear_trend = linear_trend - linear_trend.mean()
            df['linear_trend'] = (linear_trend - linear_trend.mean())/(linear_trend.std())
        elif self.z_scaling:
            df['linear_trend'] = (linear_trend - linear_trend.mean())/(linear_trend.std())
        elif self.do_detrend:
            df['linear_trend'] = linear_trend - linear_trend.mean()
        else:
            df['linear_trend'] = linear_trend
            
        return df


class QuadraticTrendFeature(FeatureGenerator):
    def __init__(self, z_scaling: bool = True, do_detrend: bool = True) -> None:
        self.z_scaling = z_scaling
        self.do_detrend = do_detrend

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        # get rid of the nan values
        values = np.array(df.target)
        values = values[~np.isnan(values)]
        
        coeff_quadratic = np.polyfit(x=np.arange(len(values)), y=values, deg=2)
        p = np.poly1d(coeff_quadratic)
        
        quadratic_trend = p(np.arange(len(np.array(df.target))))
        
        if self.do_detrend and self.z_scaling:
            quadratic_trend = quadratic_trend - quadratic_trend.mean()
            df['quadratic_trend'] = (quadratic_trend - quadratic_trend.mean())/(quadratic_trend.std())
        elif self.z_scaling:
            df['quadratic_trend'] = (quadratic_trend - quadratic_trend.mean())/(quadratic_trend.std())
        elif self.do_detrend:
            df['quadratic_trend'] = quadratic_trend - quadratic_trend.mean()
        else:
            df['quadratic_trend'] = quadratic_trend
        return df



class CubicTrendFeature(FeatureGenerator):
    def __init__(self, z_scaling: bool = True, do_detrend: bool = True) -> None:
        self.z_scaling = z_scaling
        self.do_detrend = do_detrend

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        # get rid of the nan values
        values = np.array(df.target)
        values = values[~np.isnan(values)]
        
        coeff_cubic = np.polyfit(x=np.arange(len(values)), y=values, deg=3)
        p = np.poly1d(coeff_cubic)
        
        cubic_trend = p(np.arange(len(np.array(df.target))))
        
        if self.do_detrend and self.z_scaling:
            cubic_trend = cubic_trend - cubic_trend.mean()
            df['cubic_trend'] = (cubic_trend - cubic_trend.mean())/(cubic_trend.std())
        elif self.z_scaling:
            df['cubic_trend'] = (cubic_trend - cubic_trend.mean())/(cubic_trend.std())
        elif self.do_detrend:
            df['cubic_trend'] = cubic_trend - cubic_trend.mean()
        else:
            df['cubic_trend'] = cubic_trend
        return df


class QuarticTrendFeature(FeatureGenerator):
    def __init__(self, z_scaling: bool = True, do_detrend: bool = True) -> None:
        self.z_scaling = z_scaling
        self.do_detrend = do_detrend

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:   
        
        df = df.copy()
        
        # get rid of the nan values
        values = np.array(df.target)
        values = values[~np.isnan(values)]
        
        coeff_quartic = np.polyfit(x=np.arange(len(values)), y=values, deg=4)
        p = np.poly1d(coeff_quartic)
        
        quartic_trend = p(np.arange(len(np.array(df.target))))
        
        if self.do_detrend and self.z_scaling:
            quartic_trend = quartic_trend - quartic_trend.mean()
            df['quartic_trend'] = (quartic_trend - quartic_trend.mean())/(quartic_trend.std())
        elif self.z_scaling:
            df['quartic_trend'] = (quartic_trend - quartic_trend.mean())/(quartic_trend.std())
        elif self.do_detrend:
            df['quartic_trend'] = quartic_trend - quartic_trend.mean()
        else:
            df['quartic_trend'] = quartic_trend
        return df


class AmplitudeModulationTrendFeature(FeatureGenerator):
    def __init__(self, k_max: int = 5) -> None:
        self.k_max = k_max
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Apply amplitude modulation to a univariate time series.

        Args:
            time_series_df (pd.DataFrame):
                DataFrame containing a single column of length T.

        Returns:
            np.ndarray:
                Augmented time series of length T.
        """
        df = df.copy()
        
        # extract the raw values
        raw_values = np.array(df.target)
        values = raw_values[~np.isnan(raw_values)]
        T = raw_values.shape[0]
        
        # padded values for final extrapolation
        padded_values = raw_values.copy()
        padded_values[np.isnan(padded_values)] = values[-1]
        
        # 1. sample number of changepoints k ∈ {0,1,..,5}
        k = np.random.randint(0, self.k_max)

        # 2. sample k unique changepoints in {1,..,T-1}
        if k > 0:
            change_points = np.random.choice(
                np.arange(1, T), size=k, replace=False
            )
            change_points = sorted(change_points.tolist())
        else:
            change_points = []

        # 3. build boundary vector c = [0, c1,..,ck, T]
        # boundaries = [0] + change_points + [T]
        boundaries = list(np.linspace(0, T, self.k_max).astype(int))
        # print(f"boundaries: {boundaries}")

        # 4. sample amplitudes a ∼ N(1,1) for each knot (k+2 points)
        amplitudes = np.random.normal(loc=1.0, scale=1.0,
                                    size=len(boundaries))

        # 5. interpolate trend t over [0..T)
        trend = np.empty(T, dtype=float)
        for idx in range(len(boundaries) - 1):
            start, end = boundaries[idx], boundaries[idx + 1]
            a_start, a_end = amplitudes[idx], amplitudes[idx + 1]
            segment_length = end - start

            if segment_length > 0:
                # linear interpolation from a_start to a_end
                rel_pos = np.linspace(
                    0, 1, num=segment_length, endpoint=False
                )
                trend[start:end] = a_start + (a_end - a_start) * rel_pos

        # 6. apply amplitude modulation: y_aug = y ⊙ t
        augmented_series = padded_values * trend
        # print(f"trend: {trend}")
        
        # df['amplitude_modulation_trend'] = augmented_series
        # df['amplitude_modulation_trend'] = (augmented_series - augmented_series.mean())/(augmented_series.std())
        df['amplitude_modulation_trend'] = (augmented_series - augmented_series.min())/(augmented_series.max() - augmented_series.min())
        return df


class PatchingFeature(FeatureGenerator):
    def __init__(self, input_window_size: Optional[int] = None, window_fraction: Optional[float] = 0.2) -> None:
        
        if not (0 < window_fraction < 1):
            raise ValueError("window_fraction must be between 0 and 1 (exclusive).")
        
        self.window_fraction = window_fraction
        self.input_window_size = input_window_size

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        X_torch = torch.from_numpy(df["target"].to_numpy())
        
        if self.input_window_size is None:
            window_size = int(self.window_fraction * X_torch.shape[0])
        else:
            window_size = self.input_window_size
        
        lag_features = create_lag_matrix(X_torch, window_size=window_size)
        
        for i in range(lag_features.shape[1]):
            df[f"lag_{i}"] = lag_features[:, i].numpy()
        
        return df

class PatchingFeatureMean(FeatureGenerator):
    def __init__(self, input_window_size: Optional[int] = None, window_fraction: Optional[float] = 0.2) -> None:
        
        if not (0 < window_fraction < 1):
            raise ValueError("window_fraction must be between 0 and 1 (exclusive).")
        
        self.window_fraction = window_fraction
        self.input_window_size = input_window_size

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        X_torch = torch.from_numpy(df["target"].to_numpy())
        
        if self.input_window_size is None:
            window_size = int(self.window_fraction * X_torch.shape[0])
        else:
            window_size = self.input_window_size
        
        lag_features = create_lag_matrix(X_torch, window_size=window_size)
        
        lag_mean = lag_features.mean(dim=1).reshape(-1,1)
        df["lag_mean"] = lag_mean.numpy()
        
        return df


class PatchingFeatureMedian(FeatureGenerator):
    def __init__(self, input_window_size: Optional[int] = None, window_fraction: Optional[float] = 0.2) -> None:
        
        if not (0 < window_fraction < 1):
            raise ValueError("window_fraction must be between 0 and 1 (exclusive).")
        
        self.window_fraction = window_fraction
        self.input_window_size = input_window_size

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        X_torch = torch.from_numpy(df["target"].to_numpy())
        
        if self.input_window_size is None:
            window_size = int(self.window_fraction * X_torch.shape[0])
        else:
            window_size = self.input_window_size
        
        lag_features = create_lag_matrix(X_torch, window_size=window_size)
        
        lag_median = lag_features.median(dim=1).values.reshape(-1,1)
        df["lag_median"] = lag_median.numpy()
        
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