#!/bin/bash
set -euo pipefail
analysis_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd "$analysis_directory/../.." && pwd)"
mkdir -p "$repository_directory/logs"
sbatch --chdir="$repository_directory" \
    --output="$repository_directory/logs/secondary_analysis_%j.log" \
    --error="$repository_directory/logs/secondary_analysis_%j.log" \
    "$analysis_directory/run_slurm.sh" "$repository_directory" "$@"
