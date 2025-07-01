import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


import gluonts.time_feature
from scipy import fft
from scipy.signal import find_peaks

from typing import List, Optional, Tuple, Literal, Dict

class RunningIndexFeature(BaseEstimator, TransformerMixin):
    """
    Adds a running index feature to the DataFrame.

    Parameters
    ----------
    mode : str, default="per_item"
        - "per_item": running index resets for each item_id.
        - "global_timestamp": running index is assigned based on unique sorted timestamps across all item_ids.

    Attributes
    ----------
    timestamp_to_index_ : dict
        Mapping from timestamp to index (only for global_timestamp mode).
    """

    def __init__(self, mode="per_item"):
        self.mode = mode
        self.timestamp_to_index_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain columns: "item_id", "timestamp", "target"
        y : Ignored

        Returns
        -------
        self
        """
        if self.mode == "global_timestamp":
            unique_timestamps = np.sort(X["timestamp"].unique())
            self.timestamp_to_index_ = {ts: i for i, ts in enumerate(unique_timestamps)}
        return self

    def transform(self, X):
        """
        Transform the DataFrame by adding the running index feature.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        X_out : pd.DataFrame
            With new column:
                - "running_index" (per_item mode)
                - "timestamp_index" (global_timestamp mode)
        """
        X = X.copy()
        if self.mode == "per_item":
            X["running_index"] = X.groupby("item_id").cumcount()
        elif self.mode == "global_timestamp":
            if self.timestamp_to_index_ is None:
                raise RuntimeError("Must call fit before transform in global_timestamp mode.")
            X["timestamp_index"] = X["timestamp"].map(self.timestamp_to_index_)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return X

class AutoSeasonalFeatureSklearn(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_top_k: int = 5,
        do_detrend: bool = True,
        detrend_type: Literal["first_diff", "loess", "linear", "constant"] = "linear",
        use_peaks_only: bool = True,
        apply_hann_window: bool = True,
        zero_padding_factor: int = 2,
        round_to_closest_integer: bool = True,
        validate_with_acf: bool = False,
        sampling_interval: float = 1.0,
        magnitude_threshold: Optional[float] = 0.05,
        relative_threshold: bool = True,
        exclude_zero: bool = True,
    ):
        self.max_top_k = max_top_k
        self.do_detrend = do_detrend
        self.detrend_type = detrend_type
        self.use_peaks_only = use_peaks_only
        self.apply_hann_window = apply_hann_window
        self.zero_padding_factor = zero_padding_factor
        self.round_to_closest_integer = round_to_closest_integer
        self.validate_with_acf = validate_with_acf
        self.sampling_interval = sampling_interval
        self.magnitude_threshold = magnitude_threshold
        self.relative_threshold = relative_threshold
        self.exclude_zero = exclude_zero
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns the seasonal periods from the target column of the training data.

        Args:
            X (pd.DataFrame): Training data. Must contain a 'target' column.
                              The index must be ('item_id', 'timestamp').
            y: Ignored.
        """
        if "target" not in X.columns:
            raise ValueError("AutoSeasonalFeature requires a 'target' column in X for fitting.")
        
        # Detect periods from the target data
        detected_periods_and_magnitudes = self._find_seasonal_periods(
            X.target,
            max_top_k=self.max_top_k,
            do_detrend=self.do_detrend,
            detrend_type=self.detrend_type,
            use_peaks_only=self.use_peaks_only,
            apply_hann_window=self.apply_hann_window,
            zero_padding_factor=self.zero_padding_factor,
            round_to_closest_integer=self.round_to_closest_integer,
            validate_with_acf=self.validate_with_acf,
            sampling_interval=self.sampling_interval,
            magnitude_threshold=self.magnitude_threshold,
            relative_threshold=self.relative_threshold,
            exclude_zero=self.exclude_zero,
        )
        print(f"Found {len(detected_periods_and_magnitudes)} seasonal periods: {detected_periods_and_magnitudes}")

        # Store the learned periods (the "state") with a trailing underscore
        self.detected_periods_ = [period for period, _ in detected_periods_and_magnitudes]
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sine/cosine features for the seasonal periods learned during 'fit'.
        """
        # Check if fit has been called
        if not self.fitted_:
            raise RuntimeError("Must call fit before transform.")
        
        X_copy = X.copy()
        
        # Generate features only if periods were detected
        if self.detected_periods_:
            running_index = np.arange(len(X_copy))
            for i, period in enumerate(self.detected_periods_):
                # Standardize column names for consistency across all time series
                X_copy[f"sin_#{i}"] = np.sin(2 * np.pi * running_index / period)
                X_copy[f"cos_#{i}"] = np.cos(2 * np.pi * running_index / period)

        # Add placeholder zero columns for missing periods up to max_top_k
        # This ensures the output DataFrame always has a consistent number of columns
        for i in range(len(self.detected_periods_), self.max_top_k):
            X_copy[f"sin_#{i}"] = 0.0
            X_copy[f"cos_#{i}"] = 0.0
            
        return X_copy
    
    @staticmethod
    def _find_seasonal_periods(
        target_values: pd.Series, **kwargs
    ) -> List[Tuple[float, float]]:
        """
        Identify dominant seasonal periods in a time series using FFT.
        (Static method remains the same as provided, but is now private)
        """
        # ... (The full code for find_seasonal_periods from your prompt goes here) ...
        # ... For brevity, I am omitting the full 100+ lines of this static method ...
        # --- PASTE THE `find_seasonal_periods` METHOD HERE ---
        # Convert the Pandas Series to a NumPy array
        values = np.array(target_values, dtype=float)

        # This handles the case where a test set (with NaN targets) might be passed
        # during a premature fit-transform call. In our flow, this only sees train data.
        values = values[~np.isnan(values)]
        
        if len(values) < 4: # Not enough data for FFT
            return []

        N_original = len(values)

        # Detrend the signal
        if kwargs.get("do_detrend", True):
            values = detrend(values, kwargs.get("detrend_type", "linear"))

        # Apply a Hann window
        if kwargs.get("apply_hann_window", True):
            values = values * np.hanning(N_original)

        # Zero-pad the signal
        zero_padding_factor = kwargs.get("zero_padding_factor", 2)
        if zero_padding_factor > 1:
            padded_length = int(N_original * zero_padding_factor)
            values = np.pad(values, (0, padded_length - N_original), 'constant')
        N = len(values)

        # FFT computation
        fft_values = fft.rfft(values)
        fft_magnitudes = np.abs(fft_values)
        freqs = np.fft.rfftfreq(N, d=kwargs.get("sampling_interval", 1.0))
        fft_magnitudes[0] = 0.0 # Exclude DC component

        # Thresholding
        magnitude_threshold = kwargs.get("magnitude_threshold")
        if magnitude_threshold is not None and kwargs.get("relative_threshold", True):
            threshold_value = magnitude_threshold * np.max(fft_magnitudes)
        else:
            threshold_value = magnitude_threshold

        # Peak finding
        if kwargs.get("use_peaks_only", True):
            peak_indices, _ = find_peaks(fft_magnitudes, height=threshold_value)
            if len(peak_indices) == 0:
                peak_indices = np.arange(len(fft_magnitudes))
            sorted_peak_indices = peak_indices[np.argsort(fft_magnitudes[peak_indices])[::-1]]
            top_indices = sorted_peak_indices[:kwargs.get("max_top_k", 5)]
        else:
            sorted_indices = np.argsort(fft_magnitudes)[::-1]
            if threshold_value is not None:
                sorted_indices = [i for i in sorted_indices if fft_magnitudes[i] >= threshold_value]
            top_indices = sorted_indices[:kwargs.get("max_top_k", 5)]

        # Convert frequencies to periods
        periods = np.zeros_like(freqs)
        non_zero = freqs > 0
        periods[non_zero] = 1.0 / freqs[non_zero]
        top_periods = periods[top_indices]

        if kwargs.get("round_to_closest_integer", True):
            top_periods = np.round(top_periods)

        if kwargs.get("exclude_zero", True):
            non_zero_mask = top_periods != 0
            top_periods = top_periods[non_zero_mask]
            top_indices = top_indices[non_zero_mask]

        if len(top_periods) > 0:
            unique_period_indices = np.unique(top_periods, return_index=True)[1]
            top_periods = top_periods[unique_period_indices]
            top_indices = top_indices[unique_period_indices]

        results = [(top_periods[i], fft_magnitudes[top_indices[i]]) for i in range(len(top_indices))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # (ACF validation logic would go here if enabled)

        return results





class PeriodicSinCosineFeature(BaseEstimator, TransformerMixin):
    """
    Adds sine and cosine features for given periods based on the running index.
    """
    def __init__(self, periods: List[float], name_suffix: str = None):
        self.periods = periods
        self.name_suffix = name_suffix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        running_index = np.arange(len(X_copy))
        for i, period in enumerate(self.periods):
            name = f"{self.name_suffix}_{i}" if self.name_suffix else f"{period}"
            X_copy[f"sin_{name}"] = np.sin(2 * np.pi * running_index / period)
            X_copy[f"cos_{name}"] = np.cos(2 * np.pi * running_index / period)
        return X_copy





class CalendarFeatureSklearn(BaseEstimator, TransformerMixin):
    """
    Wrapper for CalendarFeature to provide sklearn-style transform interface.

    Parameters
    ----------
    components : list of str, optional
        Calendar components to extract.
    seasonal_features : dict, optional
        Seasonal features to extract.

    Notes
    -----
    Stateless; fit does nothing.
    """

    def __init__(self, components=None, seasonal_features=None):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "second_of_minute": [60],
            "minute_of_hour": [60],
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        
        # LET'S JUST SET LEVEL
        # X_copy = X_copy.set_index(["item_id", "timestamp"])
        
        # Ensure the index is a DatetimeIndex
        timestamps = pd.DatetimeIndex(pd.to_datetime(X_copy["timestamp"]))

        # Add basic calendar components
        for component in self.components:
            X_copy[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, periods in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if periods is not None:
                for period in periods:
                    period = period - 1  # Adjust for 0-based indexing
                    X_copy[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    X_copy[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                X_copy[feature_name] = feature
        
        return X_copy.reset_index()



def detrend(
    x: np.ndarray, detrend_type: Literal["first_diff", "loess", "linear"]
) -> np.ndarray:
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])

    elif detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        lowess = nonparametric.lowess(x, indices, frac=0.1)
        trend = lowess[:, 1]
        return x - trend

    elif detrend_type in ["linear", "constant"]:
        from scipy.signal import detrend as scipy_detrend

        return scipy_detrend(x, type=detrend_type)

    else:
        raise ValueError(f"Invalid detrend method: {detrend_type}")




# Example usage:
# from feature_pipeline import RunningIndexFeature, AutoSeasonalFeatureSklearn, CalendarFeatureSklearn
# train_df, test_df = ... # split by time
# pipeline = [
#     RunningIndexFeature(mode="per_item"),
#     RunningIndexFeature(mode="global_timestamp"),
#     AutoSeasonalFeatureSklearn(),
#     CalendarFeatureSklearn(),
# ]
# for feat in pipeline:
#     feat.fit(train_df)
# train_feat = train_df.copy()
# test_feat = test_df.copy()
# for feat in pipeline:
#     train_feat = feat.transform(train_feat)
#     test_feat = feat.transform(test_feat) 