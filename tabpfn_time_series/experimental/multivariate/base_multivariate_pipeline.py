import numpy as np
import pandas as pd
import logging

from abc import abstractmethod
from autogluon.timeseries import TimeSeriesDataFrame
from gluonts.model.forecast import QuantileForecast

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.experimental.pipeline.pipeline import TabPFNTSPipeline
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG


logger = logging.getLogger(__name__)


class TabPFNBaseMultiVariatePipeline(TabPFNTSPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _predict_batch(self, test_data_input):
        pass

    def _preprocess_test_data(self, test_data_input):
        # Pre-allocate list with known size
        time_series = [None] * len(test_data_input)

        for i, item in enumerate(test_data_input):
            target = item["target"]  # shape: (target_dim, prediction_length)
            assert target.shape[0] > 1

            # raise error if NaN values are present in target
            if np.isnan(target).any():
                raise ValueError(f"NaN values present in target for item_id: {i}")

            # Create timestamp index once
            timestamp = pd.date_range(
                start=item["start"].to_timestamp(),
                periods=target.shape[1],
                freq=item["freq"],
            )

            # Create DataFrame directly with final structure
            time_series[i] = pd.DataFrame(
                target.T,
                columns=[f"target_{j}" for j in range(target.shape[0])],
                index=pd.MultiIndex.from_product(
                    [[item["item_id"]], timestamp], names=["item_id", "timestamp"]
                ),
            )

        # Concat pre-allocated list
        train_tsdf = TimeSeriesDataFrame(pd.concat(time_series))

        # assert no more NaN in train_tsdf target
        for col in train_tsdf.columns:
            if "target" in col:
                assert not train_tsdf[col].isnull().any()

        # Slice if needed
        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )

        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        # Quick fix for now ("target" column is always appended to the end of the dataframe)
        # We don't wan't it.
        train_tsdf.drop(columns=["target"], inplace=True)
        test_tsdf.drop(columns=["target"], inplace=True)

        return train_tsdf, test_tsdf

    @staticmethod
    def _preds_by_variate_to_quantile_forecast(
        test_data_input,
        all_preds_by_variate,
        test_tsdf,
        ds_freq: str,
    ) -> QuantileForecast:
        # Generate QuantileForecast objects for each time series
        forecasts = [None] * len(test_data_input)
        forecast_keys = list(map(str, TABPFN_TS_DEFAULT_QUANTILE_CONFIG))

        all_item_ids = [sample["item_id"] for sample in test_data_input]
        for i, item_id in enumerate(all_item_ids):
            forecast_start_timestamp = test_tsdf.index.get_level_values(1)[0]
            forecast_arrays = []
            for single_var_pred in all_preds_by_variate:
                # We only need the quantiles for final output
                pred = single_var_pred.loc[item_id].drop(columns=["target"])

                # single_variate_pred shape: (horizon, quantile)
                # reshape to (1, quantile, horizon)
                forecast_arrays.append(pred.values.T[np.newaxis, ...])

            forecast_arrays = np.concatenate(
                forecast_arrays, axis=0
            )  # result shape: (num_variates, quantiles, horizon)
            forecast_arrays = forecast_arrays.transpose(
                1, 2, 0
            )  # result shape: (quantiles, horizon, num_variates)

            forecasts[i] = QuantileForecast(
                item_id=item_id,
                forecast_arrays=forecast_arrays,
                forecast_keys=forecast_keys,
                start_date=forecast_start_timestamp.to_period(ds_freq),
            )

        return forecasts
