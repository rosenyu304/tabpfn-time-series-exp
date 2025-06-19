from datasets import load_dataset
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X



dataset_metadata = {
    "monash_tourism_monthly": {"prediction_length": 24}, # {"prediction_length": 24},
    "m4_hourly": {"prediction_length": 48},
    "electricity_15min": {"prediction_length": 48},
}

dataset_choice = "monash_tourism_monthly"
num_time_series_subset = 16

for binary in range(2):
    if binary<1:
        PATHC = True
    else:
        PATHC = False

    for i in range(num_time_series_subset):
        id_ = i

        dataset = load_dataset("autogluon/chronos_datasets", dataset_choice)

        tsdf = TimeSeriesDataFrame(to_gluonts_univariate(dataset['train']))
        prediction_length = dataset_metadata[dataset_choice]['prediction_length']

        tsdf = tsdf[tsdf.index.get_level_values('item_id').isin(tsdf.item_ids[[id_]])]
        train_tsdf, test_tsdf_ground_truth = tsdf.train_test_split(prediction_length=prediction_length)
        test_tsdf = generate_test_X(train_tsdf, prediction_length)

        from tabpfn_time_series.plot import plot_actual_ts
        import matplotlib.pyplot as plt
        # import plotext as plt

        # fig = plt.figure(figsize=(5, 5))
        plot_actual_ts(train_tsdf, test_tsdf_ground_truth)
        # plt.savefig(f"plots/origin_exp1_{id_}.png")

        from tabpfn_time_series import FeatureTransformer
        from tabpfn_time_series.features import (
            RunningIndexFeature,
            CalendarFeature,
            AutoSeasonalFeature,
        )
        from patch_features import PatchingFeature


        if PATHC:
            selected_features = [
                RunningIndexFeature(),
                CalendarFeature(),
                AutoSeasonalFeature(),
                PatchingFeature(),
            ]
        else:
            selected_features = [
                RunningIndexFeature(),
                CalendarFeature(),
                AutoSeasonalFeature(),
            ]

        feature_transformer = FeatureTransformer(selected_features)

        train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)


        from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

        predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
        )

        pred = predictor.predict(train_tsdf, test_tsdf)


        from tabpfn_time_series.plot import plot_pred_and_actual_ts

        plot_pred_and_actual_ts(
            train=train_tsdf,
            test=test_tsdf_ground_truth,
            pred=pred,
        )

        from autogluon.timeseries.metrics.point import MASE
        from autogluon.timeseries.utils.datetime import get_seasonality


        MASEComputer = MASE()
        MASEComputer.clear_past_metrics()

        pred["mean"] = pred["target"]


        MASEComputer.save_past_metrics(
            data_past=train_tsdf,
            seasonal_period=get_seasonality(train_tsdf.freq),
        )

        clean_mase = MASEComputer.compute_metric(
            data_future=test_tsdf_ground_truth.slice_by_timestep(-prediction_length -1, -1),
            predictions=pred,
        )

        # fig = plt.figure(figsize=(5, 5))
        
        # plot_pred_and_actual_ts(
        #     train=train_tsdf,
        #     test=test_tsdf_ground_truth,
        #     pred=pred,
        # )
        # plt.savefig(f"plots/pred_exp1_{id_}.png")
        
        print(f"ID: {id_}, PATHC: {PATHC}, clean_mase: {clean_mase}")