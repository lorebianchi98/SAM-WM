#!/bin/bash

# Environment name to activate
CONDA_ENV="stable-wm"

# List of commands to execute in separate tmux sessions
declare -a COMMANDS=(
    "python scripts/plan/oracle_mpc.py cost_function=mask solver.device=cpu solver.n_steps=30 solver.num_samples=300 solver.batch_size=1 video_path=video/pusht_oracle_mask --config-name pusht_best-threshs"
    "python scripts/plan/oracle_mpc.py cost_function=mask solver.device=cpu solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 video_path=video/reacher_oracle_mask --config-name reacher_best-threshs"
    "python scripts/plan/oracle_mpc.py cost_function=mask solver.device=cpu solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 video_path=video/cube_oracle_mask --config-name cube_best-threshs"
    "python scripts/plan/oracle_mpc.py cost_function=mask solver.device=cpu solver.n_steps=10 solver.num_samples=300 solver.batch_size=1 video_path=video/tworoom_oracle_mask --config-name tworoom_long_best-threshs"
)

# Loop over commands and launch a dedicated tmux session for each
for cmd in "${COMMANDS[@]}"; do
    # Extract cost_function and config-name to build a readable session name
    cost_func=$(echo "$cmd" | grep -oP 'cost_function=\K[^ ]+')
    config_name=$(echo "$cmd" | grep -oP '\-\-config-name \K[^ ]+')
    session_name="oracle_${config_name}_${cost_func}"

    echo "Launching session: '${session_name}'"

    # 1. Start a new detached tmux session running bash
    tmux new-session -d -s "$session_name" "bash"

    # 2. Source conda and activate the environment
    tmux send-keys -t "$session_name" "source \$(conda info --base)/etc/profile.d/conda.sh" C-m
    tmux send-keys -t "$session_name" "conda activate ${CONDA_ENV}" C-m

    # 3. Queue the CPU command with CUDA disabled, then auto-kill session when completed
    full_cmd="CUDA_VISIBLE_DEVICES= ${cmd}"
    tmux send-keys -t "$session_name" "$full_cmd" C-m
    tmux send-keys -t "$session_name" "tmux kill-session -t ${session_name}" C-m
done

echo "---"
echo "All oracle MPC sessions launched with environment '${CONDA_ENV}'!"
echo "Run 'tmux ls' to monitor them."