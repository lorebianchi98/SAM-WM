#!/bin/bash

# Ensure bash version supports associative arrays (Bash 4+)
if ((BASH_VERSINFO[0] < 4)); then
    echo "Error: This script requires Bash 4 or higher."
    exit 1
fi

# Declare an associative array: [config_name]="cuda_devices"
declare -A job_map

# ---------------------------------------------------------
# CONFIGURE YOUR JOBS HERE
# ---------------------------------------------------------
job_map["prejepa_pusht_bs128_lr0.01"]="0,1"
job_map["prejepa_reacher_bs128_lr0.01"]="2,3"
job_map["prejepa_cube_bs128_lr0.01"]="4,5"
job_map["prejepa_tworoom_bs128_lr0.01"]="6,7"
# ---------------------------------------------------------

# Environment name to activate
CONDA_ENV="stable-wm"

# Loop through the map and launch the tmux sessions
for config in "${!job_map[@]}"; do
    cuda_devices=${job_map[$config]}
    session_name="train_${config//./_}"
    
    echo "Starting session '${session_name}' on CUDA devices: ${cuda_devices}"
    
    # 1. Start a new detached tmux session
    tmux new-session -d -s "$session_name"
    
    # 2. Activate the conda environment first
    tmux send-keys -t "$session_name" "conda activate ${CONDA_ENV}" C-m
    
    # 3. Construct and send the training command
    full_cmd="CUDA_VISIBLE_DEVICES=${cuda_devices} python scripts/train/prejepa.py --config-name ${config}"
    tmux send-keys -t "$session_name" "$full_cmd" C-m
done

echo "---"
echo "All sessions successfully initialized with conda environment '${CONDA_ENV}'!"
echo "Run 'tmux ls' to check active sessions."