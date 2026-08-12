#!/bin/bash

# Ensure bash version supports associative arrays (Bash 4+)
if ((BASH_VERSINFO[0] < 4)); then
    echo "Error: This script requires Bash 4 or higher."
    exit 1
fi

# Environment name to activate
CONDA_ENV="stable-wm"

# Declare associative arrays for arguments and CUDA devices
declare -A eval_args
declare -A cuda_map

# -----------------------------------------------------------------------------------------
# CONFIGURATION: Define CUDA devices and script arguments for each task
# -----------------------------------------------------------------------------------------
# 1. pusht
cuda_map["pusht"]="6"
eval_args["pusht"]="policy=pusht_dinov2_small_psmall solver.n_steps=30 solver.num_samples=300 solver.batch_size=1 eval.num_eval=50 video_path=video/rasterized_dino-wm_pusht rastered_masks_goals=true --config-name pusht"

# 2. reacher
cuda_map["reacher"]="6"
eval_args["reacher"]="policy=reacher_dinov2_small_psmall solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 eval.num_eval=50 video_path=video/rasterized_dino-wm_reacher rastered_masks_goals=true --config-name reacher"

# 3. cube
cuda_map["cube"]="6"
eval_args["cube"]="policy=cube_dinov2_small_psmall solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 eval.num_eval=50 video_path=video/rasterized_dino-wm_cube rastered_masks_goals=true --config-name cube"

# 4. tworoom
cuda_map["tworoom"]="6"
eval_args["tworoom"]="policy=tworoom_dinov2_small_psmall solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 eval.num_eval=50 video_path=video/rasterized_dino-wm_tworoom rastered_masks_goals=true --config-name tworoom"
# -----------------------------------------------------------------------------------------

# Loop through the tasks and launch tmux sessions
for task in "${!eval_args[@]}"; do
    cuda_devices=${cuda_map[$task]}
    args=${eval_args[$task]}
    session_name="eval_${task}_dinowm_rasterized"
    
    echo "Starting evaluation session '${session_name}' on CUDA device(s): ${cuda_devices}"
    
    # 1. Start a new detached tmux session
    tmux new-session -d -s "$session_name"
    
    # 2. Activate the conda environment
    tmux send-keys -t "$session_name" "conda activate ${CONDA_ENV}" C-m
    
    # 3. Construct and send the full evaluation command
    full_cmd="CUDA_VISIBLE_DEVICES=${cuda_devices} python scripts/plan/eval_wm.py ${args}"
    tmux send-keys -t "$session_name" "$full_cmd" C-m
done

echo "---"
echo "All evaluation sessions initialized with conda environment '${CONDA_ENV}'!"
echo "Run 'tmux ls' to check active sessions."