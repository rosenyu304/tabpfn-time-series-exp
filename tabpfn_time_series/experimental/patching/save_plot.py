import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame

# Rosen
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


def is_subset(tsdf_A: TimeSeriesDataFrame, tsdf_B: TimeSeriesDataFrame) -> bool:
    tsdf_index_set_A, tsdf_index_set_B = set(tsdf_A.index), set(tsdf_B.index)
    return tsdf_index_set_A.issubset(tsdf_index_set_B)


def plot_time_series(
    df: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    in_single_plot: bool = False,
    y_limit: tuple[float, float] | None = None,
    show_points: bool = False,
    target_col: str = "target",
    title: str = None,
):
    if item_ids is None:
        item_ids = df.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(df.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    if not in_single_plot:
        # create subplots
        fig, axes = plt.subplots(
            len(item_ids), 1, figsize=(10, 3 * len(item_ids)), sharex=True
        )

        if len(item_ids) == 1:
            axes = [axes]

        for ax, item_id in zip(axes, item_ids):
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col])
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
            ax.set_title(f"Item ID: {item_id}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Target")
            if y_limit is not None:
                ax.set_ylim(*y_limit)
            if title is not None:
                ax.set_title(title)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        for item_id in item_ids:
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col], label=f"Item ID: {item_id}")
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
        ax.legend()
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        if title is not None:
            ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_actual_ts(
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_points: bool = False,
):
    if item_ids is None:
        item_ids = train.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(train.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    _, ax = plt.subplots(len(item_ids), 1, figsize=(10, 3 * len(item_ids)))
    ax = [ax] if not isinstance(ax, np.ndarray) else ax

    def plot_single_item(ax, item_id):
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        if is_subset(train_item, test_item):
            ground_truth = test_item["target"]
        else:
            ground_truth = pd.concat([train_item[["target"]], test_item[["target"]]])
        ax.plot(ground_truth.index, ground_truth, label="Ground Truth")
        if show_points:
            ax.scatter(
                ground_truth.index, ground_truth, color="lightblue", s=8, alpha=0.8
            )

        train_item_length = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(
            x=train_item_length, color="r", linestyle="--", label="Train/Test Split"
        )

        ax.set_title(f"Item ID: {item_id}")
        ax.legend()

    for i, item_id in enumerate(item_ids):
        plot_single_item(ax[i], item_id)

    plt.tight_layout()
    plt.show()


def plot_pred_and_actual_ts(
    pred: TimeSeriesDataFrame,
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_quantiles: bool = True,
    show_points: bool = False,
    hide_legend: bool = False,
    title: str = None,
    save_path: str = None,
    font_weight: str = "semibold",
    font_size: int = 10,
    line_width: float = 1.2,
):
    if item_ids is None:
        item_ids = train.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(train.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    if pred.shape[0] != test.shape[0]:
        if not is_subset(pred, test):
            raise ValueError(
                "Pred and Test have different number of items and Pred is not a subset of Test"
            )

        filled_pred = test.copy()
        filled_pred["target"] = np.nan
        for col in pred.columns:
            filled_pred.loc[pred.index, col] = pred[col]
        pred = filled_pred

    assert pred.shape[0] == test.shape[0]

    _, ax = plt.subplots(len(item_ids), 1, figsize=(10, 3 * len(item_ids)))
    ax = [ax] if not isinstance(ax, np.ndarray) else ax

    def plot_single_item(ax, item_id):
        pred_item = pred.xs(item_id, level="item_id")
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        if is_subset(train_item, test_item):
            ground_truth = test_item["target"]
        else:
            ground_truth = pd.concat([train_item[["target"]], test_item[["target"]]])
        ax.plot(
            ground_truth.index,
            ground_truth,
            label="Ground Truth",
            linewidth=line_width,
            rasterized=True,
        )  # Updated spline weight and rasterized
        ax.plot(
            pred_item.index,
            pred_item["target"],
            label="Prediction",
            linewidth=line_width,
            rasterized=True,
        )  # Updated spline weight and rasterized
        if show_points:
            ax.scatter(
                ground_truth.index,
                ground_truth,
                color="lightblue",
                s=8,
                alpha=0.8,
                rasterized=True,
            )

        if show_quantiles:
            # Plot the lower and upper bound of the quantile predictions
            quantile_config = sorted(
                pred_item.columns.drop(["target"]).tolist(), key=lambda x: float(x)
            )
            # print(quantile_config)
            lower_quantile = quantile_config[0]
            upper_quantile = quantile_config[-1]
            ax.fill_between(
                pred_item.index,
                pred_item[lower_quantile],
                pred_item[upper_quantile],
                color="gray",
                alpha=0.2,
                label=f"{lower_quantile}-{upper_quantile} Quantile Range",
                rasterized=True,  # Updated border weight and rasterized
            )

        train_item_length = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(
            x=train_item_length,
            color="r",
            linestyle="--",
            label="Train/Test Split",
            linewidth=line_width,
            rasterized=True,
        )

        if title is None:
            ax.set_title(
                f"Item ID: {item_id}", fontsize=font_size, fontweight=font_weight
            )
        else:
            ax.set_title(title, fontsize=font_size, fontweight=font_weight)

        if not hide_legend:
            ax.legend(
                loc="upper left", bbox_to_anchor=(0, 1), fontsize=font_size
            )  # Font size

    for i, item_id in enumerate(item_ids):
        plot_single_item(ax[i], item_id)

    plt.tight_layout()
    if save_path is not None:
        # Save as PDF with optimized settings for large number of points
        plt.savefig(save_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.show()
