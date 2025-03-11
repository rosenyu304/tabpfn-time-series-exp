from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

import gluonts.time_feature
from autogluon.timeseries import TimeSeriesDataFrame


class FeatureGenerator(ABC):
    """Abstract base class for feature generators"""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataframe"""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)


class RunningIndexFeature(FeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["running_index"] = range(len(df))
        return df


class CalendarFeature(FeatureGenerator):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, float]] = None,
    ):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "hour_of_day": 24,
            "day_of_week": 7,
            "day_of_month": 30.5,
            "day_of_year": 365,
            "week_of_year": 52,
            "month_of_year": 12,
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")

        # Add basic calendar components
        for component in self.components:
            df[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, seasonality in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if seasonality is not None:
                period = seasonality - 1  # Adjust for 0-based indexing
                df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature

        return df


class AdditionalCalendarFeature(CalendarFeature):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        additional_seasonal_features: Optional[Dict[str, float]] = None,
    ):
        super().__init__(components=components)

        self.seasonal_features = {
            **additional_seasonal_features,
            **self.seasonal_features,
        }


class FeatureTransformer:
    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both train and test data with the configured feature generators"""

        self._validate_input(train_tsdf, test_tsdf, target_column)
        tsdf = pd.concat([train_tsdf, test_tsdf])

        # Apply all feature generators
        for generator in self.feature_generators:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(generator)

        # Split train and test tsdf
        train_tsdf = tsdf.iloc[: len(train_tsdf)]
        test_tsdf = tsdf.iloc[len(train_tsdf) :]

        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf

    @staticmethod
    def _validate_input(
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str,
    ):
        if target_column not in train_tsdf.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        if not test_tsdf[target_column].isna().all():
            raise ValueError("Test data should not contain target values")
