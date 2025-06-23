import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

import pandas as pd
from datasets import load_dataset
from typing import Optional

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics.point import MASE
from autogluon.timeseries.utils.datetime import get_seasonality

from tabpfn_time_series import (
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
)
from tabpfn_time_series.data_preparation import to_gluonts_univariate, generate_test_X
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)
from patch_features import PatchingFeature
from detrend_y import detrend_tsdf, retrend_tsdf
# from save_plot import plot_pred_and_actual_ts


import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# -- Configuration ------------------------------------------------------------
DATASET_METADATA = {
    "m4_daily": {"prediction_length": 14},
    "m4_hourly": {"prediction_length": 14},
    "m4_monthly": {"prediction_length": 13},
    "m4_quarterly": {"prediction_length": 18},
    "m4_weekly": {"prediction_length": 8},
    "solar_1h": {"prediction_length": 48},
    "taxi_1h": {"prediction_length": 48},
    "monash_tourism_monthly": {"prediction_length": 24},
    "electricity_15min": {"prediction_length": 48},
}








# -- MASE Evaluator ----------------------------------------------------------------
def quick_mase_evaluation(train_tsdf, test_tsdf_ground_truth, pred, prediction_length):
    """
    Compute MASE scores for each item_id and overall average.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['item_id', 'mase_score']
                     Last row contains average with item_id='AVERAGE'
    """
    from autogluon.timeseries.metrics.point import MASE
    from autogluon.timeseries.utils.datetime import get_seasonality
    import pandas as pd
    
    mase_results = []
    
    # Loop over each item_id and calculate MASE score
    for item_id, df_item in train_tsdf.groupby(level="item_id"):
        mase_computer = MASE()
        mase_computer.clear_past_metrics()
        
        pred["mean"] = pred["target"]
        
        mase_computer.save_past_metrics(
            data_past=train_tsdf.loc[[item_id]],
            seasonal_period=get_seasonality(train_tsdf.freq),
        )
        
        mase_score = mase_computer.compute_metric(
            data_future=test_tsdf_ground_truth.loc[[item_id]].slice_by_timestep(
                -prediction_length - 1, -1
            ),
            predictions=pred.loc[[item_id]],
        )
        
        mase_results.append({
            'item_id': item_id,
            'mase_score': mase_score
        })
    
    # Create DataFrame with individual results
    results_df = pd.DataFrame(mase_results)
    
    # Add average row
    average_mase = results_df['mase_score'].mean()
    average_row = pd.DataFrame({
        'item_id': ['AVERAGE'],
        'mase_score': [average_mase]
    })
    
    # Combine results
    final_results = pd.concat([results_df, average_row], ignore_index=True)
    
    return final_results


# -- Full Prediction Pipeline --------------------------------------------------
def get_prediction(dataset_choice: str, 
                   prediction_length: int, 
                   root_output_dir: str, 
                   DEGREE: int,
                   SAVE_RESULTS: bool):
    load_dotenv()

    print(f"--------DEGREE: {DEGREE}--------")
    print(f"--------SAVE_RESULTS: {SAVE_RESULTS}--------")
    
    # Load dataset
    dataset = load_dataset("autogluon/chronos_datasets", dataset_choice)

    # Convert to TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame(to_gluonts_univariate(dataset['train']))
    num_time_series_subset = len(tsdf.item_ids)
    print(f"--------num_time_series_subset: {num_time_series_subset}--------")
    if num_time_series_subset > 30:
        num_time_series_subset = 30
    
    # Take a subset and split into train and ground truth
    tsdf = tsdf[tsdf.index.get_level_values('item_id').isin(tsdf.item_ids[:num_time_series_subset])]
    
    # Detrending
    tsdf, trends = detrend_tsdf(tsdf, degree=DEGREE)
    
    # Split into train and ground truth
    train_tsdf, test_tsdf_ground_truth = tsdf.train_test_split(prediction_length=prediction_length)
    test_tsdf = generate_test_X(train_tsdf, prediction_length)
    
    # Feature Engineering (Patching Testing)
    selected_features = [
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ]
    
    
    feature_transformer = FeatureTransformer(selected_features)
    train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)




    # Prediction
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.LOCAL,
    )
    pred = predictor.predict(train_tsdf, test_tsdf)


    # Retrending
    plot_train_tsdf = retrend_tsdf(train_tsdf, trends, df_type="train")
    plot_test_gt_tsdf = retrend_tsdf(test_tsdf_ground_truth, trends, df_type="train")
    plot_pred_tsdf = retrend_tsdf(pred, trends, df_type="pred")

    # Plotting
    from save_plot import plot_pred_and_actual_ts
    output_dir = Path(f"{root_output_dir}/detrend_y")
    output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_RESULTS:
        plot_dir = f"{output_dir}/degree_{DEGREE}"
        os.makedirs(plot_dir, exist_ok=True)
        save_plot_path = f"{plot_dir}/D_{dataset_choice.replace('/', '-')}_degree_{DEGREE}.pdf"
        print(f"file path: {save_plot_path}")
    else:
        save_plot_path = None
    
    plot_pred_and_actual_ts(plot_pred_tsdf, plot_train_tsdf, plot_test_gt_tsdf, 
                            item_ids=list(plot_train_tsdf.item_ids), show_points=True,
                            title=f"{dataset_choice}_degree_{DEGREE}",
                            save_path=save_plot_path
                        )
    
    # Calculate MASE
    final_results = quick_mase_evaluation(train_tsdf, test_tsdf_ground_truth, 
                                      pred, prediction_length,
                                      )


    if SAVE_RESULTS:
        mase_dir = f"{output_dir}/degree_{DEGREE}"
        os.makedirs(mase_dir, exist_ok=True)
        print(f"file path: {mase_dir}/D_{dataset_choice.replace('/', '-')}_degree_{DEGREE}.csv")
        final_results.to_csv(f"{mase_dir}/D_{dataset_choice.replace('/', '-')}_degree_{DEGREE}.csv", index=False)
    print(f"--------get mase--------")
    
    return final_results




# -- Main ----------------------------------------------------------------
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--degree", type=int, default=1)
        parser.add_argument("--save_results", action="store_true")
        args = parser.parse_args()
        
        for dataset_choice in DATASET_METADATA.keys():
            try:
                print(f"--------WORK ON {dataset_choice}--------")
                get_prediction(dataset_choice, 
                             DATASET_METADATA[dataset_choice]['prediction_length'], 
                             "plots/June20", 
                             args.degree, 
                             args.save_results,
                             )
                print(f"--------{dataset_choice} DONE--------")
            except KeyError as ke:
                print(f"Error: Missing metadata for dataset {dataset_choice}: {ke}")
            except Exception as e:
                print(f"Error processing dataset {dataset_choice}: {e}")
                continue
        print(f"--------ALL DONE--------")
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    