import pandas as pd
import wandb
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


def download_wandb_runs(
    entity: str,
    project: str,
    tags: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Download runs data from Weights & Biases with parallelization.

    Args:
        entity: The wandb entity (username or team name)
        project: The wandb project name
        tags: Optional list of tags to filter runs
        max_workers: Maximum number of parallel workers

    Returns:
        pandas.DataFrame: DataFrame containing the runs data
    """
    api = wandb.Api()

    # Build the filter query
    filter_query = {}
    if tags:
        filter_query["tags"] = {"$in": tags}

    # Get runs from the specified project
    runs = api.runs(f"{entity}/{project}", filters=filter_query)

    print(f"Found {len(runs)} runs")

    def process_run(run):
        """Process a single run and extract its data"""
        run_data = {
            "name": run.name,
            "state": run.state,
        }

        # Add config parameters
        for key, value in run.config.items():
            if key not in run_data:
                run_data[f"config/{key}"] = value

        # Add summary metrics
        for key, value in run.summary.items():
            if key not in run_data:
                run_data[key] = value

        return run_data

    # Extract run data in parallel
    runs_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_run = {executor.submit(process_run, run): run for run in runs}

        # Process results as they complete with a progress bar
        for future in tqdm.tqdm(
            as_completed(future_to_run), total=len(runs), desc="Processing runs"
        ):
            try:
                run_data = future.result()
                runs_data.append(run_data)
            except Exception as exc:
                print(f"Run processing generated an exception: {exc}")

    # Convert to DataFrame
    df = pd.DataFrame(runs_data)

    return df
