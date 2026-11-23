"""Evaluate the upper bound of mask-based MPC with an environment oracle."""

import csv
import json
import os
import time
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
import stable_worldmodel as swm

from stable_worldmodel.solver.oracle import OracleMaskModel

from eval_wm import get_dataset, get_episodes_length


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run CEM while scoring candidates with the real environment state."""
    requested_indices = cfg.get('episodes', None)
    if requested_indices is not None:
        requested_indices = list(requested_indices)
        if not requested_indices:
            raise ValueError('episodes must contain at least one dataset index.')
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError('episodes must contain unique dataset indices.')
        cfg.eval.num_eval = len(requested_indices)

    if cfg.get('cost_function', 'mask') not in (
        'mask',
        'mask_iou',
        'mask_mse',
    ):
        raise ValueError(
            "Set cost_function to 'mask', 'mask_iou', or 'mask_mse'."
        )
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError('Planning horizon must fit within eval_budget.')

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    dataset.augmentations_goals = cfg.get('augmentations_goals', False)
    dataset.rastered_masks_goals = cfg.get('rastered_masks_goals', False)
    dataset.textual_query = list(cfg.textual_query)

    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
    process = {}
    for col in ('action',):
        if col not in cfg.dataset.keys_to_cache:
            continue
        scaler = preprocessing.StandardScaler()
        values = dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        scaler.fit(values)
        process[col] = scaler
        if col != 'action':
            process[f'goal_{col}'] = scaler

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_by_episode = dict(zip(ep_indices, max_start))
    valid = dataset.get_col_data('step_idx') <= np.array(
        [max_start_by_episode[x] for x in dataset.get_col_data(col_name)]
    )
    valid_indices = np.flatnonzero(valid)
    rng = np.random.default_rng(cfg.seed)
    if requested_indices is None:
        if len(valid_indices) < cfg.eval.num_eval:
            raise ValueError('Not enough valid starting points for evaluation.')
        selected = np.sort(
            rng.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
        )
    else:
        selected = np.asarray(requested_indices)
        if not np.issubdtype(selected.dtype, np.integer):
            raise ValueError('episodes must contain integer dataset indices.')
        dataset_size = len(dataset.get_col_data(col_name))
        invalid_indices = selected[
            (selected < 0) | (selected >= dataset_size)
        ]
        if len(invalid_indices) > 0:
            raise ValueError(
                f'episodes contains indices outside the dataset: '
                f'{invalid_indices.tolist()}'
            )
    print(selected)
    eval_episodes = dataset.get_row_data(selected)[col_name]
    eval_start = dataset.get_row_data(selected)['step_idx']

    oracle = OracleMaskModel(
        cfg.world.env_name,
        cfg.plan_config.action_block,
        list(cfg.textual_query),
        cfg.get('cost_function', 'mask'),
        cfg.get('weight_cost_strategy', 'area'),
        process.get('action'),
    )
    solver = hydra.utils.instantiate(cfg.solver, model=oracle)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process
    )
    oracle.set_env(world.envs)
    world.set_policy(policy)

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    start = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video=cfg.get('video_path', None),
        save_frames=cfg.get('save_frames', False),
    )
    elapsed = time.time() - start
    print(metrics)
    print(f'[eval] videos saved to {Path(cfg.video_path).resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(results_path).replace('txt', 'csv'))

    config_data = OmegaConf.to_container(cfg, resolve=True)
    config_data['eval_time'] = elapsed
    config_json = json.dumps(config_data, default=str)
    csv_row = {
        'model_name': 'oracle_mpc',
        'n_epoch': cfg.get('n_epoch', None),
        'success_rate': metrics['success_rate'],
        'cost_function': cfg.get('cost_function', 'mse'),
        'config': config_json,
        'episode_successes': json.dumps(
            np.asarray(metrics['episode_successes']).tolist()
        ),
    }

    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'model_name',
                'n_epoch',
                'success_rate',
                'cost_function',
                'config',
                'episode_successes',
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    with results_path.open('a') as file:
        file.write('\n')
        file.write('==== CONFIG ====\n')
        file.write(OmegaConf.to_yaml(cfg))
        file.write('\n')
        file.write('==== RESULTS ====\n')
        file.write(f'metrics: {metrics}\n')
        file.write(f'evaluation_time: {elapsed} seconds\n')


if __name__ == '__main__':
    run()
