import logging

from tabpfn_time_series.experimental.multivariate.base_multivariate_pipeline import (
    TabPFNBaseMultiVariatePipeline,
)


logger = logging.getLogger(__name__)


class TabPFNARMultiVariatePipeline(TabPFNBaseMultiVariatePipeline):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # Drop all target columns to start with
        target_columns = [
            c for c in train_tsdf.columns if "target" in c
        ]  # Assume all target variates are prefixed with "target"
        current_train_tsdf = train_tsdf.drop(columns=target_columns)
        current_test_tsdf = test_tsdf.drop(columns=target_columns)

        all_preds_by_variate = []

        # Predict each variate individually, conditioned on previously predicted variates
        for i, var in enumerate(target_columns):
            logger.debug(f"Predicting variate {var}")

            # Set the target column
            current_train_tsdf[var] = train_tsdf[var]
            current_test_tsdf[var] = test_tsdf[var]
            current_train_tsdf.rename(columns={var: "target"}, inplace=True)
            current_test_tsdf.rename(columns={var: "target"}, inplace=True)

            # Predict current variate
            current_pred = self.tabpfn_predictor.predict(
                current_train_tsdf, current_test_tsdf
            )
            all_preds_by_variate.append(current_pred)

            # Add the prediction of the current variate to the context
            current_train_tsdf.rename(columns={"target": var}, inplace=True)
            current_test_tsdf.rename(columns={"target": var}, inplace=True)
            current_test_tsdf[var] = current_pred["target"]

        forecasts = self._preds_by_variate_to_quantile_forecast(
            test_data_input,
            all_preds_by_variate,
            test_tsdf,
            self.ds_freq,
        )

        return forecasts
