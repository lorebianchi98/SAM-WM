#!/bin/bash

# Ensure bash version supports associative arrays (Bash 4+)
if ((BASH_VERSINFO[0] < 4)); then
    echo "Error: This script requires Bash 4 or higher."
    exit 1
fi

# Environment name to activate
CONDA_ENV="stable-wm"

# Declare associative arrays for policies, configs, and CUDA devices
declare -A policies
declare -A configs
declare -A cuda_map

# -----------------------------------------------------------------------------------------
# CONFIGURATION: Define tasks, policies, configs, and CUDA devices
# -----------------------------------------------------------------------------------------
# 1. pusht
cuda_map["pusht"]="0"
policies["pusht"]="pusht_sam3_psmall"
configs["pusht"]="pusht"

# 2. tworoom
cuda_map["tworoom"]="1"
policies["tworoom"]="tworoom_sam3_psmall"
configs["tworoom"]="tworoom"

# 3. reacher
cuda_map["reacher"]="2"
policies["reacher"]="reacher_sam3_psmall"
configs["reacher"]="reacher"

# 4. cube
cuda_map["cube"]="3"
policies["cube"]="cube_sam3_psmall"
configs["cube"]="cube"
# -----------------------------------------------------------------------------------------

# Loop through tasks and launch individual detached tmux sessions
for task in "${!policies[@]}"; do
    cuda_device=${cuda_map[$task]}
    policy=${policies[$task]}
    cfg=${configs[$task]}
    session_name="miou_${task}"

    echo "Launching '${session_name}' on CUDA device: ${cuda_device}"

    # 1. Start a new detached tmux session running bash
    tmux new-session -d -s "$session_name" "bash"

    # 2. Source conda and activate the target environment
    tmux send-keys -t "$session_name" "source \$(conda info --base)/etc/profile.d/conda.sh" C-m
    tmux send-keys -t "$session_name" "conda activate ${CONDA_ENV}" C-m

    # 3. Queue the calculation command and kill the tmux session when finished
    cmd="CUDA_VISIBLE_DEVICES=${cuda_device} python scripts/ablations/calculate_miou.py policy=${policy} --config-name=${cfg}"
    tmux send-keys -t "$session_name" "$cmd" C-m
    tmux send-keys -t "$session_name" "tmux kill-session -t ${session_name}" C-m
done

echo "---"
echo "All mIoU sessions initialized in parallel with conda environment '${CONDA_ENV}'!"
echo "Run 'tmux ls' to monitor running sessions."