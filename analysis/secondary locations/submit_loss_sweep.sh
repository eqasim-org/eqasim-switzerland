#!/bin/bash
set -euo pipefail
analysis_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd "$analysis_directory/../.." && pwd)"
# Validate and pin the completed baseline before requesting a compute node.
cd "$repository_directory"
uv run python "$analysis_directory/loss_sweep.py" --phase plan "$@"
mkdir -p "$repository_directory/logs"
sbatch --parsable --chdir="$repository_directory" \
    --output="$repository_directory/logs/secondary_loss_weights_%j.log" \
    --error="$repository_directory/logs/secondary_loss_weights_%j.log" \
    "$analysis_directory/run_loss_sweep.sh" "$repository_directory" "$@"
