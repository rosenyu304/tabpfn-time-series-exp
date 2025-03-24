import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import gluonts.time_feature

from tabpfn_time_series.experimental.features.feature_generator_base import (
    FeatureGenerator,
)


class RunningIndexFeature(FeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["running_index"] = range(len(df))
        return df


class CalendarFeature(FeatureGenerator):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")

        # Add basic calendar components
        for component in self.components:
            df[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, periods in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if periods is not None:
                for period in periods:
                    period = period - 1  # Adjust for 0-based indexing
                    df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature

        return df


class AdditionalCalendarFeature(CalendarFeature):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        additional_seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(components=components)

        self.seasonal_features = {
            **additional_seasonal_features,
            **self.seasonal_features,
        }


class PeriodicSinCosineFeature(FeatureGenerator):
    def __init__(self, periods: List[float]):
        self.periods = periods

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for period in self.periods:
            df[f"sin_{period}"] = np.sin(2 * np.pi * np.arange(len(df)) / period)
            df[f"cos_{period}"] = np.cos(2 * np.pi * np.arange(len(df)) / period)

        return df


class AutoSeasonalFeatures(FeatureGenerator):
    def __init__(self, max_top_k: int = 10):
        self.max_top_k = max_top_k

    @staticmethod
    def find_seasonal_periods(
        target_values: pd.Series,
        max_top_k: int = 10,
        do_detrend: bool = True,
        use_peaks_only: bool = True,
        apply_hann_window: bool = True,
        zero_padding_factor: int = 3,
        round_to_closest_integer: bool = True,
        validate_with_acf: bool = False,
    ) -> List[float]:
        from scipy.signal import fft, find_peaks
        from scipy.stats import acf

        target_values = np.array(target_values)

        if do_detrend:
            target_values = np.diff(target_values, prepend=target_values[0])

        N = len(target_values)
        if apply_hann_window:
            window = np.hanning(N)
            target_values = target_values * window

        # Apply zero-padding if requested for better frequency resolution
        if zero_padding_factor > 1:
            padded_length = int(N * zero_padding_factor)
            padded_values = np.zeros(padded_length)
            padded_values[:N] = target_values
            target_values = padded_values
            N = padded_length

        fft_values = fft.rfft(target_values)
        fft_magnitudes = np.abs(fft_values)
        all_freq = fft.rfftfreq(N, d=1.0)  # One sample per unit time

        # Get frequencies
        if use_peaks_only:
            # Find peaks in the FFT spectrum
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(fft_magnitudes)
            # Sort peaks by magnitude
            peak_magnitudes = fft_magnitudes[peaks]
            sorted_peak_indices = peaks[np.argsort(peak_magnitudes)[::-1]]
            # Get top k peaks
            top_indices = sorted_peak_indices[:max_top_k]
        else:
            freq_indices = np.argsort(fft_magnitudes)[::-1]  # Sort in descending order
            # Get top k seasonal periods (excluding the first which is often trend)
            top_indices = freq_indices[:max_top_k]
            # top_indices = freq_indices[1:top_k+1]

        # Convert frequencies to periods (avoiding division by zero)
        periods = np.zeros_like(all_freq)
        non_zero_mask = all_freq > 0
        periods[non_zero_mask] = 1.0 / all_freq[non_zero_mask]

        top_periods = periods[top_indices]
        print(f"debug, periods: {top_periods}")

        if round_to_closest_integer:
            top_periods = np.round(top_periods)

        if validate_with_acf:
            # Calculate ACF for validation
            original_length = (
                len(target_values)
                if not zero_padding_factor > 1
                else int(N / zero_padding_factor)
            )
            nlags = original_length
            acf_values = acf(target_values[:original_length], nlags=nlags)

            # Find peaks in ACF
            acf_peaks, _ = find_peaks(
                acf_values, height=1.96 / np.sqrt(original_length)
            )

            # Filter top periods to those that have corresponding peaks in ACF
            validated_indices = []
            validated_periods = []

            for i, period in enumerate(top_periods):
                # Check if period is close to any ACF peak
                period_int = int(round(period))
                if period_int < len(acf_values) and any(
                    abs(period_int - peak) <= 1 for peak in acf_peaks
                ):
                    validated_indices.append(top_indices[i])
                    validated_periods.append(period)

            # If we have validated periods, use them instead
            if validated_periods:
                top_periods = np.array(validated_periods)

        return top_periods
