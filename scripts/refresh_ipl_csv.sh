#!/usr/bin/env sh
set -eu

target_dir="${CRICKET_PREDICTOR_IPL_CSV_DATA_DIR:-/app/data/ipl_csv}"
dataset_url="${CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL:-https://www.kaggle.com/api/v1/datasets/download/krishd123/ipl-2026-complete-dataset}"
tmp_dir="$(mktemp -d)"
archive_path="$tmp_dir/ipl_csv.zip"
extract_dir="$tmp_dir/extracted"

cleanup() {
	rm -rf "$tmp_dir"
}

trap cleanup EXIT INT TERM

mkdir -p "$target_dir" "$extract_dir"

echo "Downloading IPL CSV dataset into $tmp_dir"
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
	echo "Using Kaggle credentials for dataset download"
	curl --fail --location --silent --show-error \
		--user "$KAGGLE_USERNAME:$KAGGLE_KEY" \
		--output "$archive_path" \
		"$dataset_url"
else
	echo "Downloading without Kaggle credentials"
	curl --fail --location --silent --show-error \
		--output "$archive_path" \
		"$dataset_url"
fi

echo "Extracting IPL CSV dataset"
python - <<'PY' "$archive_path" "$extract_dir"
import sys
import zipfile

archive_path = sys.argv[1]
extract_dir = sys.argv[2]

with zipfile.ZipFile(archive_path) as archive:
		archive.extractall(extract_dir)
PY

python - <<'PY' "$extract_dir" "$target_dir"
from pathlib import Path
import shutil
import sys

extract_dir = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
required = {"matches.csv", "points_table.csv", "orange_cap.csv", "purple_cap.csv", "squads.csv"}
optional = {"deliveries.csv", "venues.csv"}

found: dict[str, Path] = {}
for path in extract_dir.rglob("*.csv"):
		found.setdefault(path.name, path)

missing = sorted(required - set(found))
if missing:
		raise SystemExit(f"Missing required IPL CSV files: {', '.join(missing)}")

for filename in required | optional:
		source = found.get(filename)
		if source is None:
				continue
		shutil.copy2(source, target_dir / filename)

print("Refreshed IPL CSV files:", ", ".join(sorted((required | optional) & set(found))))
PY

echo "IPL CSV refresh completed into $target_dir"