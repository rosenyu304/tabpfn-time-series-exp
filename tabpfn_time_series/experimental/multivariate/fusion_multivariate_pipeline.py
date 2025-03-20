import logging
import pandas as pd

from tabpfn_time_series.experimental.multivariate.base_multivariate_pipeline import (
    TabPFNBaseMultiVariatePipeline,
)


logger = logging.getLogger(__name__)


class TabPFNFusionMultiVariatePipeline(TabPFNBaseMultiVariatePipeline):
    def __init__(
        self,
        use_str_variate_indicator: bool = True,
        use_int_variate_indicator: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_str_variate_indicator = use_str_variate_indicator
        self.use_int_variate_indicator = use_int_variate_indicator

        # At least one of the two must be True
        assert self.use_str_variate_indicator or self.use_int_variate_indicator

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # Fuse the multivariate data into a single table
        fused_train_tsdf, fused_test_tsdf = self._fuse_multivariate_data(
            train_tsdf, test_tsdf
        )

        # Predict the fused data
        fused_predictions = self.tabpfn_predictor.predict(
            fused_train_tsdf,
            fused_test_tsdf,
        )

        # Defuse the predictions into per-variate predictions
        all_preds_by_variate = self._defuse_multivariate_predictions(
            fused_predictions, fused_test_tsdf
        )

        forecasts = self._preds_by_variate_to_quantile_forecast(
            test_data_input,
            all_preds_by_variate,
            test_tsdf,
            self.ds_freq,
        )

        return forecasts

    def _fuse_multivariate_data(self, train_tsdf, test_tsdf):
        target_columns = [
            c for c in train_tsdf.columns if "target" in c
        ]  # Assume all target variates are prefixed with "target"

        # Get non-target columns once
        feature_columns = [c for c in train_tsdf.columns if c not in target_columns]

        # Pre-allocate lists for better performance
        all_sub_train_tsdf = []
        all_sub_test_tsdf = []

        # Process each item_id separately
        for item_id in train_tsdf.index.get_level_values(0).unique():
            item_train = train_tsdf.loc[item_id]
            item_test = test_tsdf.loc[item_id]

            # Extract features once per item
            item_train_features = item_train[feature_columns]
            item_test_features = item_test[feature_columns]

            # Pre-allocate dataframes for this item
            item_train_dfs = []
            item_test_dfs = []

            for i, variate in enumerate(target_columns):
                # Create new dataframes with features and target
                train_df = item_train_features.copy()
                test_df = item_test_features.copy()

                # Add variate identifier and target
                if self.use_str_variate_indicator:
                    train_df["variate_str"] = variate
                    test_df["variate_str"] = variate

                if self.use_int_variate_indicator:
                    train_df["variate_int"] = i
                    test_df["variate_int"] = i

                train_df["target"] = item_train[variate]
                test_df["target"] = item_test[variate]

                item_train_dfs.append(train_df)
                item_test_dfs.append(test_df)

            # Concatenate all variates for this item
            item_train_combined = pd.concat(item_train_dfs)
            item_test_combined = pd.concat(item_test_dfs)

            # Add item_id and reset index
            item_train_combined["item_id"] = item_id
            item_test_combined["item_id"] = item_id

            # Reset and set index in one operation
            item_train_combined = item_train_combined.reset_index().set_index(
                ["item_id", "timestamp"]
            )
            item_test_combined = item_test_combined.reset_index().set_index(
                ["item_id", "timestamp"]
            )

            all_sub_train_tsdf.append(item_train_combined)
            all_sub_test_tsdf.append(item_test_combined)

        # Concatenate all items
        fused_train_tsdf = pd.concat(all_sub_train_tsdf)
        fused_test_tsdf = pd.concat(all_sub_test_tsdf)

        return fused_train_tsdf, fused_test_tsdf

    def _defuse_multivariate_predictions(self, fused_predictions, fused_test_tsdf):
        all_preds_by_variate = []

        variate_indicators = (
            "variate_str" if self.use_str_variate_indicator else "variate_int"
        )

        for variate in fused_test_tsdf[variate_indicators].unique():
            mask = fused_test_tsdf[variate_indicators] == variate
            curr_variate_pred = fused_predictions[mask.values]
            all_preds_by_variate.append(curr_variate_pred)

        return all_preds_by_variate
