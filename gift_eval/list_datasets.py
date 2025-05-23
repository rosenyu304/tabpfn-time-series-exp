import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from gift_eval.dataset_definition import (
    ALL_DATASETS,
    MED_LONG_DATASETS,
    DATASET_PROPERTIES_MAP,
)
from gift_eval.evaluate import pretty_names

def list_datasets():
    """List all available datasets and their supported forecast terms."""
    print("\nAvailable Datasets and Their Properties:")
    print("-" * 80)
    print(f"{'Dataset':<40} {'Terms':<15}")
    print("-" * 80)

    for dataset in sorted(ALL_DATASETS):
        # Determine available terms
        terms = ["short"]
        if dataset in MED_LONG_DATASETS:
            terms.extend(["medium", "long"])
        
        # Get dataset key (handle both simple and complex dataset names)
        ds_key = dataset.split("/")[0].lower()

        if "/" in dataset:
            ds_key = dataset.split("/")[0]
        ds_key = ds_key.lower()
        ds_key = pretty_names.get(ds_key, ds_key)
        
        print(f"{dataset:<40} {','.join(terms):<15}")

if __name__ == "__main__":
    list_datasets() 