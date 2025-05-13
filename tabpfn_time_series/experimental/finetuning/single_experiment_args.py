from argparse import ArgumentParser, Namespace


def common_parse_args(return_parser: bool = False) -> Namespace | ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Fine-tune TabPFN Time Series models")

    # Configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--sample_factor",
        type=int,
        default=5,
        help="Factor to sample the dataset by",
    )
    parser.add_argument(
        "--term",
        type=str,
        default="short",
        help="Term to use for the dataset",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="vanilla",
        help="Name of the base model configuration to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="default",
        help="Name of the method configuration to use (overrides experiment's method)",
    )

    # Runtime settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Logging settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-ft",
        help="Weights & Biases project name",
    )
    # parser.add_argument(
    #     "--run_name",
    #     type=str,
    #     default="dummy-run-name",
    #     help="Custom name for the W&B run",
    # )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for the W&B run",
    )

    # Training settings
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "-s",
        "--accumulate_grad_steps",
        type=int,
        help="Number of gradient accumulation steps",
    )

    parser.add_argument(
        "--no_log_model",
        action="store_true",
        help="Disable logging the model to Weights & Biases",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    parser.add_argument(
        "--also_validate_on_train",
        action="store_true",
        help="Also validate on training datasets",
    )

    if return_parser:
        return parser
    else:
        return parser.parse_args()
