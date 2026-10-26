#!/bin/bash
set -euo pipefail
analysis_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd "$analysis_directory/../.." && pwd)"
cd "$repository_directory"
uv run python "$analysis_directory/short_loss_trial.py" --phase plan "$@"
mkdir -p "$repository_directory/logs"
sbatch --parsable --chdir="$repository_directory" \
    --output="$repository_directory/logs/secondary_short_loss_%j.log" \
    --error="$repository_directory/logs/secondary_short_loss_%j.log" \
    "$analysis_directory/run_short_loss_trial.sh" "$repository_directory" "$@"
