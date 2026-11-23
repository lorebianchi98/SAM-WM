# World Models on Segmentation Features enable Mask-Guided Planning

SAM-WM extends the [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) project by adding action-conditioned transition-model
training over frozen Segment Anything 3 (SAM3) features. This enables zero-shot model-predictive
control (MPC) using natural-language goals and binary destination masks instead of target RGB images.
<table align="center">
  <tr>
    <td align="center">
      <b>Push-T</b><br>
      <img src="assets/pusht_plan.gif" width="100%" />
    </td>
    <td align="center">
      <b>Two-Room</b><br>
      <img src="assets/tworoom_plan.gif" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Reacher</b><br>
      <img src="assets/reacher_plan.gif" width="100%" />
    </td>
    <td align="center">
      <b>OGBench-Cube</b><br>
      <img src="assets/cube_plan.gif" width="100%" />
    </td>
  </tr>
</table>

<!-- <p align="center">
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#quick-start"><b>Datasets</b></a> ·
  <a href="#environments"><b>Training</b></a> ·
  <a href="#solvers-and-baselines"><b>Evaluation</b></a> ·
  <a href="https://galilai-group.github.io/stable-worldmodel/"><b>Results</b></a> ·
  <!-- <a href="#citation"><b>Citation</b></a> -->
</p> 
Paper coming soon!

## Features

- Action-conditioned transition model trained on SAM3 features
- Zero-shot MPC with natural language queries and mask-based objectives
- Integrations with existing stable-worldmodel datasets and policies


## News
- 📢 Join us in Malmö for the presentation of SAM3-WM at the [UniWorld Workshop](https://uniworld-ai.github.io/) and the [How to Build Effective World Models for Embodied AI](https://eccv26wmeai.github.io/) workshop at ECCV 2026!
## Installation
From source:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
# Conda example (adjust Python version as needed)
conda create -n stable-wm python=3.12 -y
conda activate stable-wm
conda install -c conda-forge uv -y  # only if `uv` is missing
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
uv sync --extra all --group dev
```

venv workflow:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
uv venv --python=3.10 && source .venv/bin/activate
uv sync --extra all --group dev
```

Datasets and checkpoints are stored under the environment variable `STABLEWM_HOME` (default: `~/.stable_worldmodel/`). Set `STABLEWM_HOME` to change the storage location.

## Quick start

1. Install package or use the development setup above.
2. Download the datasets (see Datasets below).
3. Run a training job (example):

```bash
python scripts/train/prejepa_sam3.py --config-name CONFIG_NAME
```

To run multiple trainings (GPU selection via script):

```bash
./commands/launch_train_sam3.sh
```

## Datasets

We use the same expert datasets as LeWorldModel. You can obtain them from the Hugging Face collection:

https://huggingface.co/collections/quentinll/lewm

Place downloaded datasets under `STABLEWM_HOME` (or set `STABLEWM_HOME` to your dataset root).

## Training

- To run training for 3 seeds across every environment, use:

  ```bash
  ./commands/launch_train_sam-wm.sh
  ```

- To launch a single training job, use `scripts/train/prejepa_sam3.py` with the desired Hydra config. For example, to train Push-T:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train/prejepa_sam3.py --config-name prejepa_sam3_pusht
  ```

- Example configs are available in `scripts/train/config`.


## Evaluation & Planning

- To evaluate SAM-WM across every environment, run:

  ```bash
  ./commands/launch_eval_sam-wm.sh
  ```

- To run a single Push-T evaluation, for example:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/plan/eval_wm.py \
    policy=pusht_sam3_psmall_seed0 cost_function=mask \
    solver.n_steps=30 solver.num_samples=300 solver.batch_size=1 \
    eval.num_eval=50 video_path=video/pusht_sam3-wm \
    save_frames=true --config-name pusht
  ```



- To run oracle MPC, use `commands/launch_eval_oracle.sh`. Oracle MPC evaluates the mask-cost planner with a perfect predictor: it simulates the sampled actions in the environment and uses the real masks instead of decoding predicted masks.

Note: the configurations in `scripts/plan/config` contain SAM3 threshold values optimized on the validation set using:
  ```bash
  ./commands/launch_calculate_val_miou.sh
  ```




## Results

Comparison against JEPA world models. Task success rates (%) across 50 evaluation episodes (mean $\pm$ std over 3 seeds). Goal-image methods receive complete images of the desired final scene and use a richer goal specification. Red down arrows (🔻) denote goal-image methods that achieve equal or worse performance compared to SAM-WM (mask-based). Bold values indicate the best performance among methods using mask-based goals.

| Model | Goal Type | Two-Room* | Reacher | Push-T | Cube |
|---|:---:|:---:|:---:|:---:|:---:|
| Random | -- | $0 \pm 0$ | $10 \pm 5$ | $2 \pm 2$ | $48 \pm 3$ |
| Oracle | Masks | $74 \pm 2$ | $62 \pm 1$ | $84 \pm 2$ | $66 \pm 2$ |
| | | | | | |
| *DINO-WM* | Image | $63 \pm 1$ 🔻 | $79 \pm 6$ | $74 \pm 4$ 🔻 | $86 \pm 5$ |
| *LeWM* | Image | $16 \pm 2$ 🔻 | $86 \pm 5$ | $96 \pm 3$ | $74 \pm 3$ |
| *PLDM* | Image | $51 \pm 2$ 🔻 | $78 \pm 6$ | $78 \pm 5$ 🔻 | $65 \pm 3$ |
| | | | | | |
| DINO-WM | Masks | $45 \pm 2$ | $22 \pm 5$ | $19 \pm 1$ | $42 \pm 2$ |
| **SAM-WM (Ours)** | Masks | **$74 \pm 2$** | **$61 \pm 2$** | **$83 \pm 2$** | **$64 \pm 2$** |

<small><i>*Evaluated under an extended horizon (100-step offset, 150-step budget) and reproduced by us; all other benchmark results are taken from Maes et al. (2026).</i></small>

Notes:
- **Oracle Planner**: Evaluates mask-guided MPC using ground-truth simulator rollouts and ground-truth segmentation masks, serving as an empirical upper bound for the planning objective.
- **Image Baselines**: Rely on complete RGB goal images of the final scene. Notice that on Two-Room, SAM-WM outperforms all goal-image methods, and on Push-T it outperforms DINO-WM and PLDM without seeing a target image.
- **DINO-WM (mask-based)**: An adaptation of DINO-WM that receives a flat, binary destination mask canvas encoded via DINOv2 instead of a full RGB photo.

## Visualize Mask predictions

Use the expert-policy rollouts to visualize the world-model's predicted destination masks over expert trajectories. The helper script `scripts/ablations/predict_masks.py` produces per-episode mask-overlay videos and a trace file; a convenience launcher is available at `commands/launch_sam3-wm_predictions.sh`.

<table align="center">
  <tr>
    <td align="center">
      <b>Push-T</b><br>
      <img src="assets/pusht_preds.gif" width="100%" />
    </td>
    <td align="center">
      <b>Two-Room</b><br>
      <img src="assets/tworoom_preds.gif" width="100%" />
    </td>
    <td align="center">
      <b>Reacher</b><br>
      <img src="assets/reacher_preds.gif" width="100%" />
    </td>
    <td align="center">
      <b>OGBench-Cube</b><br>
      <img src="assets/cube_preds.gif" width="100%" />
    </td>
  </tr>
</table>

Quick examples:

Run a single environment (edit `policy` or config name as needed):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/ablations/predict_masks.py \
  policy=pusht_sam3_psmall_seed0 eval.num_eval=50 \
  video_path=video/masks_pusht --config-name pusht
```

Run all environment visualizations with the launcher:

```bash
./commands/launch_sam3-wm_predictions.sh
```

## Acknowledgements
This codebase is built upon  [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). We sincerely thank the authors and maintainers of stable-worldmodel for providing a clean, modular, and robust foundation for latent world modeling research.