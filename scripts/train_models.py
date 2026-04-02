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


def _train_from_cricsheet(settings, download: bool) -> bool:
    """Try to train from cricsheet data; return True on success."""
    from cricket_predictor.data.cricsheet_loader import CricsheetLoader

    loader = CricsheetLoader(settings.cricsheet_data_dir)
    urls = [settings.cricsheet_ipl_url, settings.cricsheet_t20_url, settings.cricsheet_recent_url]

    if download:
        print("Checking for cricsheet updates …")
        extracted = loader.download_and_extract(urls)
    else:
        # Use already-extracted directories if they exist on disk
        extracted = [
            Path(settings.cricsheet_data_dir) / Path(u).stem
            for u in urls
            if (Path(settings.cricsheet_data_dir) / Path(u).stem).exists()
        ]

    if not extracted:
        print("No cricsheet data found locally. Run with --download to fetch it.")
        return False

    print(f"Parsing {len(extracted)} source(s) …")
    matches_df = loader.parse_matches(extracted)
    players_df = loader.parse_player_stats(extracted)

    if matches_df.empty or players_df.empty:
        print(f"Parsed data is empty (matches={len(matches_df)}, players={len(players_df)}).")
        return False

    artifacts = train_all(matches_df, players_df)
    save_artifacts(artifacts, settings.model_artifact_dir)
    print(
        f"Models trained on cricsheet data: "
        f"{len(matches_df)} matches, {len(players_df)} players."
    )
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
