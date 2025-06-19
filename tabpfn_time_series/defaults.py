TABPFN_TS_DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TABPFN_TS_DEFAULT_CONFIG = {
    "tabpfn_internal": {
        "model_path": "2noar4o2",
    },
    "tabpfn_output_selection": "median",  # mean or median
}

# Rosen: Add more ensembling and polynomial_features
TABPFN_TS_STRONG_CONFIG = {
    "tabpfn_internal": {
        "model_path": "2noar4o2",
        "fingerprint_feature": True,
        "fix_nan_borders_after_target_transform": True,
        "polynomial_features": "all",  # or an int for max number
        "preprocess_transforms": None,  # Use TabPFN's default (which is already strong)
        "subsample_samples": None,
        "use_sklearn_16_decimal_precision": False,
        # If using TabPFN Extensions for post-hoc ensembling:
        "n_estimators": 10,  # or higher for more ensembling
    },
    "tabpfn_output_selection": "median",  # or "mean"
}