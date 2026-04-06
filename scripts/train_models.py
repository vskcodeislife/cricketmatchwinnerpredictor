"""Training entry-point.

Usage
-----
  # Use synthetic data (default, always works offline)
  python scripts/train_models.py

  # Use locally-cached cricsheet data (run after the server has downloaded it,
  # or after running ``POST /predict/data/refresh`` at least once)
  python scripts/train_models.py --cricsheet

  # Download fresh cricsheet data then train
  python scripts/train_models.py --cricsheet --download
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cricket_predictor.config.settings import get_settings
from cricket_predictor.data.dataset_generator import save_synthetic_datasets
from cricket_predictor.models.training import save_artifacts, train_all
from cricket_predictor.services.data_update_service import DataUpdateService


def _train_from_cricsheet(settings, download: bool) -> bool:
    """Try to train from cricsheet data; return True on success."""
    service = DataUpdateService(settings)
    if download:
        print("Downloading cricsheet archives and retraining …")
    else:
        print("Retraining from local cricsheet data and saved feedback …")

    trained = service.retrain_from_cricsheet(download=download)
    if not trained:
        print("No usable cricsheet data found locally. Run with --download to fetch it.")
        return False

    print("Models trained on cricsheet data with feedback-aware retraining.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cricket Predictor models.")
    parser.add_argument(
        "--cricsheet",
        action="store_true",
        help="Train from cricsheet real-match data instead of synthetic data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fresh cricsheet archives before training (requires --cricsheet).",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.cricsheet:
        if _train_from_cricsheet(settings, download=args.download):
            return
        print("Falling back to synthetic data.")

    datasets = save_synthetic_datasets(settings.synthetic_data_dir)
    artifacts = train_all(datasets.matches, datasets.players)
    save_artifacts(artifacts, settings.model_artifact_dir)
    print("Synthetic datasets and model artifacts generated successfully.")


if __name__ == "__main__":
    main()
