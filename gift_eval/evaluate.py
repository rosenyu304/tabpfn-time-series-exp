import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))


import csv
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

from gift_eval.data import Dataset
from gift_eval.dataset_definition import (
    MED_LONG_DATASETS,
    ALL_DATASETS,
    DATASET_PROPERTIES_MAP,
)
from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor

# Global constants
MODEL_NAME = "tabpfn-ts"

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# Set up logging
logger = logging.getLogger(__name__)


def construct_evaluation_data(
    dataset_name: str,
    dataset_storage_path: Path,
    terms: List[str] = ["short", "medium", "long"],
) -> List[Tuple[Dataset, dict]]:
    sub_datasets = []

    # Construct evaluation data
    ds_key = dataset_name.split("/")[0]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and dataset_name not in MED_LONG_DATASETS:
            continue

        if "/" in dataset_name:
            ds_key = dataset_name.split("/")[0]
            ds_freq = dataset_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = dataset_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = DATASET_PROPERTIES_MAP[ds_key]["frequency"]

        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(
                name=dataset_name,
                term=term,
                to_univariate=False,
                storage_path=dataset_storage_path,
            ).target_dim
            == 1
            else True
        )
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_path=dataset_storage_path,
        )
        season_length = get_seasonality(dataset.freq)

        dataset_metadata = {
            "full_name": f"{ds_key}/{ds_freq}/{term}",
            "key": ds_key,
            "freq": ds_freq,
            "term": term,
            "season_length": season_length,
        }
        sub_datasets.append((dataset, dataset_metadata))

    return sub_datasets


def create_csv_file(csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ]
        )


def append_results_to_csv(
    res,
    csv_file_path,
    dataset_metadata,
    model_name,
):
    with open(csv_file_path, "a", newline="") as csvfile:
        res = res.iloc[0]
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                dataset_metadata["full_name"],
                model_name,
                res["MSE[mean]"],
                res["MSE[0.5]"],
                res["MAE[0.5]"],
                res["MASE[0.5]"],
                res["MAPE[0.5]"],
                res["sMAPE[0.5]"],
                res["MSIS"],
                res["RMSE[mean]"],
                res["NRMSE[mean]"],
                res["ND[0.5]"],
                res["mean_weighted_sum_quantile_loss"],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["domain"],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["num_variates"],
            ]
        )

    logger.info(
        f"Results for {dataset_metadata['full_name']} have been written to {csv_file_path}"
    )


def evaluate_dataset(
    sub_dataset: Dataset,
    dataset_metadata: dict,
    output_csv_path: Path,
) -> None:
    """Evaluate a single dataset and log results."""
    logger.info(f"Evaluating dataset {sub_dataset.name}")
    logger.info(f"Dataset size: {len(sub_dataset.test_data)}")
    logger.info(f"Dataset freq: {sub_dataset.freq}")
    logger.info(f"Dataset term: {dataset_metadata['term']}")
    logger.info(f"Dataset prediction length: {sub_dataset.prediction_length}")
    logger.info(f"Dataset target dim: {sub_dataset.target_dim}")

    # Initialize predictor
    tabpfn_predictor = TabPFNTSPredictor(
        ds_prediction_length=sub_dataset.prediction_length,
        ds_freq=sub_dataset.freq,
    )

    # Evaluate model
    res = evaluate_model(
        tabpfn_predictor,
        test_data=sub_dataset.test_data,
        metrics=metrics,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=dataset_metadata["season_length"],
    )

    # Write results to csv
    append_results_to_csv(
        res=res,
        csv_file_path=output_csv_path,
        dataset_metadata=dataset_metadata,
        model_name=MODEL_NAME,
    )


def main(args):
    if args.dataset not in ALL_DATASETS:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    logger.info(f"Evaluating dataset {args.dataset}")

    # Check if the dataset storage path exists
    if not Path(args.dataset_storage_path).exists():
        raise ValueError(
            f"Dataset storage path {args.dataset_storage_path} does not exist"
        )

    output_dir = args.output_dir / MODEL_NAME / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "results.csv"

    # Create CSV file
    create_csv_file(output_csv_path)

    # Construct evaluation data (i.e. sub-datasets) for this dataset
    # (some datasets contain different forecasting terms, e.g. short, medium, long)
    sub_datasets = construct_evaluation_data(
        args.dataset, args.dataset_storage_path, args.terms
    )

    # Evaluate each sub-dataset
    for i, (sub_dataset, dataset_metadata) in enumerate(sub_datasets):
        logger.info(
            f"Evaluating {i + 1}/{len(sub_datasets)} dataset {sub_dataset.name}"
        )

        try:
            evaluate_dataset(
                sub_dataset=sub_dataset,
                dataset_metadata=dataset_metadata,
                output_csv_path=output_csv_path,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {sub_dataset.name}: {str(e)}")
            if args.continue_on_error:
                logger.info("Continuing with next dataset...")
                continue
            else:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tabpfn-ts")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default=str(Path(__file__).parent / "results")
    )
    parser.add_argument(
        "--terms",
        type=str,
        default="short,medium,long",
        help="Comma-separated list of terms to evaluate",
    )
    parser.add_argument(
        "--dataset_storage_path", type=str, default=str(Path(__file__).parent / "data")
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue evaluation if one dataset fails",
    )

    args = parser.parse_args()
    args.dataset_storage_path = Path(args.dataset_storage_path)
    args.output_dir = Path(args.output_dir)
    args.terms = args.terms.split(",")

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Command Line Arguments: {vars(args)}")

    main(args)
