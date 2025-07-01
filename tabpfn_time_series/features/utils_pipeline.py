import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame

def train_test_split_time_series(df: pd.DataFrame, prediction_length: int):
    """
    Splits a DataFrame into train and test sets per item_id using prediction_length.
    Args:
        df (pd.DataFrame): Input DataFrame with 'item_id' and 'timestamp'.
        prediction_length (int): Number of last time steps to use for test per item_id.
    Returns:
        train_df (pd.DataFrame): Training set (all but last prediction_length per item_id).
        test_df (pd.DataFrame): Test set (last prediction_length per item_id).
    """
    train_list = []
    test_list = []
    for item_id, group in df.groupby('item_id'):
        group_sorted = group.sort_values('timestamp')
        if len(group_sorted) <= prediction_length:
            # If not enough data, put all in train
            train_list.append(group_sorted)
            continue
        train_list.append(group_sorted.iloc[:-prediction_length])
        test_list.append(group_sorted.iloc[-prediction_length:])
    train_df = pd.concat(train_list, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).reset_index(drop=True) if test_list else pd.DataFrame(columns=df.columns)
    
    # after the train test split, make the "target" column in test_df to be NaN
    ground_truth = test_df.copy()
    test_df["target"] = np.nan
    
    return train_df, test_df, ground_truth

def from_autogluon_tsdf_to_df(tsdf):
    return tsdf.copy().to_data_frame().reset_index()

def from_df_to_autogluon_tsdf(df):
    return TimeSeriesDataFrame.from_data_frame(df.copy())