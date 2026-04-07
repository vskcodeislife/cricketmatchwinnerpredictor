#!/usr/bin/env sh
# ---------------------------------------------------------------------------
# Multi-source IPL CSV refresh
#
# Tries each configured Kaggle data source in order. If a source provides the
# standard IPL CSV layout (matches.csv, points_table.csv, …) it is used
# directly. If it provides only a ball-by-ball file, the normalizer script
# derives the standard files automatically.
#
# Environment variables:
#   CRICKET_PREDICTOR_IPL_CSV_DATA_DIR   – target directory (default: /app/data/ipl_csv)
#   CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL          – primary dataset URL
#   CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL_ALT      – alternate dataset URL (ball-by-ball)
#   KAGGLE_USERNAME / KAGGLE_KEY         – optional Kaggle auth
# ---------------------------------------------------------------------------
set -eu

target_dir="${CRICKET_PREDICTOR_IPL_CSV_DATA_DIR:-/app/data/ipl_csv}"
script_dir="$(cd "$(dirname "$0")" && pwd)"

# Data sources in priority order (first success wins)
url_primary="${CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL:-https://www.kaggle.com/api/v1/datasets/download/krishd123/ipl-2026-complete-dataset}"
url_alt="${CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL_ALT:-https://www.kaggle.com/api/v1/datasets/download/sujalninawe/ipl-2026-ball-by-ball-dataset-daily-updated}"

tmp_dir="$(mktemp -d)"
archive_path="$tmp_dir/ipl_csv.zip"
extract_dir="$tmp_dir/extracted"

cleanup() {
	rm -rf "$tmp_dir"
}
trap cleanup EXIT INT TERM
mkdir -p "$target_dir" "$extract_dir"

# Prefer python3 but fall back to python
PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)"

# ---------------------------------------------------------------------------
# download_url <url> <dest>
# ---------------------------------------------------------------------------
download_url() {
	local url="$1" dest="$2"
	if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
		curl --fail --location --silent --show-error \
			--user "$KAGGLE_USERNAME:$KAGGLE_KEY" \
			--output "$dest" "$url"
	else
		curl --fail --location --silent --show-error \
			--output "$dest" "$url"
	fi
}

# ---------------------------------------------------------------------------
# try_standard_layout <extract_dir> <target_dir>
# Returns 0 if the standard required files are present and copies them.
# ---------------------------------------------------------------------------
try_standard_layout() {
	$PYTHON - <<'PY' "$1" "$2"
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
    raise SystemExit(1)

for filename in required | optional:
    source = found.get(filename)
    if source is None:
        continue
    shutil.copy2(source, target_dir / filename)

print("standard-layout:" + ", ".join(sorted((required | optional) & set(found))))
PY
}

# ---------------------------------------------------------------------------
# try_ball_by_ball_normalise <extract_dir> <target_dir>
# Looks for any CSV inside extract_dir, normalises it into standard files.
# ---------------------------------------------------------------------------
try_ball_by_ball_normalise() {
	$PYTHON - <<'PY' "$1" "$2" "$script_dir"
import sys
from pathlib import Path

extract_dir = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
script_dir = Path(sys.argv[3])

# Add scripts directory to import the normaliser
sys.path.insert(0, str(script_dir.parent))
from scripts.normalize_bbb_csv import normalise

csvs = list(extract_dir.rglob("*.csv"))
if not csvs:
    raise SystemExit(1)

stats = normalise(csvs[0], target_dir)
print("ball-by-ball-normalised: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
PY
}

# ---------------------------------------------------------------------------
# try_source <label> <url>
# Downloads, extracts, and attempts both layout strategies.
# ---------------------------------------------------------------------------
try_source() {
	local label="$1" url="$2"
	echo "[$label] Downloading from $url"

	rm -f "$archive_path"
	rm -rf "$extract_dir"
	mkdir -p "$extract_dir"

	if ! download_url "$url" "$archive_path"; then
		echo "[$label] Download failed, skipping."
		return 1
	fi

	echo "[$label] Extracting archive"
	$PYTHON - <<'PY' "$archive_path" "$extract_dir"
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY

	# Strategy 1: standard layout (matches.csv + friends)
	if try_standard_layout "$extract_dir" "$target_dir"; then
		echo "[$label] Standard CSV layout detected and copied."
		return 0
	fi

	# Strategy 2: ball-by-ball normalisation
	if try_ball_by_ball_normalise "$extract_dir" "$target_dir"; then
		echo "[$label] Ball-by-ball data normalised into standard files."
		return 0
	fi

	echo "[$label] Could not process dataset, skipping."
	return 1
}

# ---------------------------------------------------------------------------
# Main: try sources in order
# ---------------------------------------------------------------------------
success=0

if try_source "primary" "$url_primary"; then
	success=1
elif try_source "alternate" "$url_alt"; then
	success=1
fi

if [ "$success" -eq 0 ]; then
	echo "ERROR: All data sources failed."
	exit 1
fi

echo "IPL CSV refresh completed into $target_dir"