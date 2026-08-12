"""Script to save per-frame world-model costs for expert episodes."""

import os

import sys
sys.path.append('..')
os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path
from tqdm import tqdm

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.masks.cube import (
    generate_ogbcube_cube_mask,
    generate_ogbcube_gripper_mask,
)
from stable_worldmodel.masks.pusht import (
    generate_pusht_ball_mask,
    generate_pusht_t_mask,
)
from stable_worldmodel.masks.reacher import generate_reacher_masks
from stable_worldmodel.masks.tworoom import generate_tworoom_circle_mask

from omegaconf import OmegaConf

def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    return swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )


def build_process(dataset, cfg):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]
    return process


def _apply_process(values, processor):
    values = np.asarray(values)
    shape = values.shape
    if values.ndim == 1:
        flattened = values.reshape(-1, 1)
    else:
        flattened = values.reshape(-1, shape[-1])
    transformed = processor.transform(flattened).astype(np.float32)
    return transformed.reshape(shape)


def _to_uint8_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (
        1,
        3,
    ):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)

    if frame.dtype.kind == 'f':
        max_value = float(np.max(frame)) if frame.size else 0.0
        if max_value <= 1.5:
            frame = frame * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _prepare_pixels(pixels, transform):
    return transform(_to_uint8_frame(pixels))


def _build_goal_masks(goal_state: dict, dataset_path: str):
    if 'pusht' in dataset_path:
        t_mask = torch.stack(
            [
                generate_pusht_t_mask(torch.tensor(x))
                for x in goal_state['goal_state']
            ],
            dim=0,
        )
        ball_mask = torch.stack(
            [
                generate_pusht_ball_mask(torch.tensor(x))
                for x in goal_state['goal_state']
            ],
            dim=0,
        )
        goal_state['goal_mask'] = torch.stack((t_mask, ball_mask), dim=1)
    elif 'tworoom' in dataset_path:
        circle_mask = torch.stack(
            [
                generate_tworoom_circle_mask(torch.tensor(x))
                for x in goal_state['goal_proprio']
            ],
            dim=0,
        )
        goal_state['goal_mask'] = circle_mask.unsqueeze(1)
    elif 'reacher' in dataset_path:
        stick_masks = []
        circle_masks = []
        for idx in range(goal_state['goal'].shape[0]):
            stick_mask, circle_mask = generate_reacher_masks(goal_state, idx)
            stick_masks.append(stick_mask)
            circle_masks.append(circle_mask)
        goal_state['goal_mask'] = torch.stack(
            (torch.stack(stick_masks, dim=0), torch.stack(circle_masks, dim=0)),
            dim=1,
        )
    elif 'cube' in dataset_path:
        cube_masks = []
        env_cube = CubeEnv(env_type='single', ob_type='pixels')
        for idx in range(goal_state['goal_qpos'].shape[0]):
            qpos_sample = goal_state['goal_qpos'][idx]
            env_cube.reset()
            cube_mask = generate_ogbcube_cube_mask(env_cube, qpos_sample)
            gripper_mask = generate_ogbcube_gripper_mask(env_cube, qpos_sample)
            cube_masks.append(cube_mask)
        cube_masks = torch.stack(cube_masks, dim=0)
        goal_state['goal_mask'] = cube_masks.unsqueeze(1)

    return goal_state


def _build_init_goal_from_chunk(chunk, dataset, start_idx, goal_offset):
    init_lists: dict[str, list] = {}
    goal_lists: dict[str, list] = {}

    for col in dataset.column_names:
        if col.startswith('goal'):
            continue
        value = chunk[col]
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            continue
        current = value[start_idx]
        goal = value[start_idx + goal_offset]
        init_lists.setdefault(col, []).append(
            current.numpy() if isinstance(current, torch.Tensor) else current
        )
        goal_lists.setdefault(col, []).append(
            goal.numpy() if isinstance(goal, torch.Tensor) else goal
        )

    init_state = {k: np.stack(v) for k, v in init_lists.items()}
    goal_state = {}
    for k, v in goal_lists.items():
        goal_state['goal' if k == 'pixels' else f'goal_{k}'] = np.stack(v)

    goal_state = _build_goal_masks(goal_state, dataset.h5_file.filename)

    if getattr(dataset, 'augmentations_goals', False):
        import albumentations as A

        transform = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5,
                    p=1.0,
                )
            ]
        )
        goal_state['goal'] = np.stack(
            [transform(image=goal)['image'] for goal in goal_state['goal']],
            axis=0,
        )

    if getattr(dataset, 'rastered_masks_goals', False) and 'goal_mask' in goal_state:
        goal_images = torch.zeros_like(torch.tensor(goal_state['goal']))
        collapsed_mask = goal_state['goal_mask'].any(dim=1)
        goal_images[collapsed_mask.bool()] = 255
        goal_state['goal'] = goal_images.numpy()

    return init_state, goal_state


def _frame_mask_input(frame_state: dict, dataset_path: str) -> dict:
    if 'pusht' in dataset_path:
        source = frame_state.get('state', frame_state.get('goal_state'))
        if source is None:
            raise KeyError('PushT masks require a state or goal_state column.')
        return {'goal_state': np.asarray(source)[None, ...]}

    if 'tworoom' in dataset_path:
        source = frame_state.get('proprio', frame_state.get('goal_proprio'))
        if source is None:
            raise KeyError('TwoRoom masks require a proprio or goal_proprio column.')
        return {'goal_proprio': np.asarray(source)[None, ...]}

    if 'reacher' in dataset_path:
        source = frame_state.get('qpos', frame_state.get('goal_qpos'))
        if source is None:
            source = frame_state.get('goal')
        if source is None:
            raise KeyError('Reacher masks require a qpos, goal_qpos, or goal column.')
        return {'goal_qpos': np.asarray(source)[None, ...]}

    if 'cube' in dataset_path:
        source = frame_state.get('qpos', frame_state.get('goal_qpos'))
        if source is None:
            raise KeyError('Cube masks require a qpos or goal_qpos column.')
        return {'goal_qpos': np.asarray(source)[None, ...]}

    raise ValueError(f'Unsupported dataset for mask extraction: {dataset_path}')


def _build_frame_mask(frame_state: dict, dataset_path: str) -> torch.Tensor:
    mask_input = _frame_mask_input(frame_state, dataset_path)
    return _build_goal_masks(mask_input, dataset_path)['goal_mask'][0]


def extract_episode_pixels_and_masks(
    dataset,
    episode_ids,
    goal_offset: int,
    transform: dict,
):
    """Return per-episode pixel and mask tensors for the first `goal_offset` frames.

    The returned tensors have shapes:
    - pixels: [n_episodes, goal_offset, 3, 224, 224]
    - masks: [n_episodes, goal_offset, n_masks, 224, 224]
    """

    pixel_episodes = []
    mask_episodes = []
    dataset_path = dataset.h5_file.filename

    for ep_id in episode_ids:
        chunk = dataset.load_chunk(
            np.array([ep_id]),
            np.array([0]),
            np.array([goal_offset]),
        )[0]

        frame_pixels = []
        frame_masks = []
        for frame_idx in range(goal_offset):
            frame_pixels.append(
                _prepare_pixels(chunk['pixels'][frame_idx], transform['pixels'])
            )

            frame_state = {
                key: value[frame_idx]
                for key, value in chunk.items()
                if isinstance(value, (torch.Tensor, np.ndarray))
                and not key.startswith('pixels')
            }
            frame_masks.append(_build_frame_mask(frame_state, dataset_path))

        pixel_episodes.append(torch.stack(frame_pixels, dim=0))
        mask_episodes.append(torch.stack(frame_masks, dim=0))

    return torch.stack(pixel_episodes, dim=0), torch.stack(mask_episodes, dim=0)


def _build_info_dict(init_state, goal_state, process, transform, device):
    info = {}
    for key, value in init_state.items():
        if key == 'pixels':
            info[key] = _prepare_pixels(value if value.shape[0] > 1 else value.squeeze(0), transform['pixels']).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        elif key in process:
            processed = _apply_process(value, process[key])
            info[key] = torch.as_tensor(processed, device=device).unsqueeze(0).unsqueeze(0)
        else:
            info[key] = torch.as_tensor(value, device=device).unsqueeze(0).unsqueeze(0)

    for key, value in goal_state.items():
        if key == 'goal':
            info[key] = _prepare_pixels(value if value.shape[0] > 1 else value.squeeze(0), transform['goal']).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        elif key == 'goal_mask':
            info[key] = torch.as_tensor(value, device=device).unsqueeze(0).unsqueeze(0)
        elif key in process:
            processed = _apply_process(value, process[key])
            info[key] = torch.as_tensor(processed, device=device).unsqueeze(0).unsqueeze(0)
        else:
            info[key] = torch.as_tensor(value, device=device).unsqueeze(0).unsqueeze(0)

    return info


def _build_action_candidates(raw_actions, action_block, device):
    raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
    if raw_actions.ndim != 2:
        raise ValueError(
            f'Expected raw actions to be (T, A), got shape={tuple(raw_actions.shape)}'
        )
    if raw_actions.shape[0] % action_block != 0:
        raise ValueError(
            f'Action window length {raw_actions.shape[0]} is not divisible by action_block={action_block}'
        )

    horizon = raw_actions.shape[0] // action_block
    chunks = []
    for idx in range(horizon):
        block = raw_actions[idx * action_block : (idx + 1) * action_block]
        chunks.append(block.reshape(-1))
    return torch.stack(chunks, dim=0).unsqueeze(0).unsqueeze(0)


@hydra.main(version_base=None, config_path='../plan/config', config_name='pusht')
def run(cfg: DictConfig):
    """Save a tensor of frame-wise costs for sampled expert episodes."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    dataset.augmentations_goals = cfg.get('augmentations_goals', False)
    dataset.rastered_masks_goals = cfg.get('rastered_masks_goals', False)
    process = build_process(dataset, cfg)

    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)

    total_action_steps = cfg.plan_config.horizon * cfg.plan_config.action_block
    required_len = max(cfg.eval.goal_offset_steps + 1, total_action_steps)
    valid_episode_mask = episode_len >= required_len
    valid_episodes = ep_indices[valid_episode_mask]
    if len(valid_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for cost evaluation.'
        )

    g = np.random.default_rng(cfg.seed)
    selected_episodes = g.choice(
        valid_episodes, size=cfg.eval.num_eval, replace=False
    )

    model = swm.wm.utils.load_pretrained(
        cfg.policy, n_epoch=cfg.get('n_epoch', None)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    if cfg.get('bf16', False):
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model.cost_function = cfg.get('cost_function', 'mse')

    if 'mask' in model.cost_function:
        print('Using mask cost function for cost extraction.')
        model.build_mask_decoder(
            inference_resolution=224,
            textual_query=list(cfg.textual_query),
            threshold=cfg.threshold,
            mask_threshold=cfg.mask_threshold,
        )
        model.weight_cost_strategy = cfg.weight_cost_strategy

    if hasattr(model.extra_encoders, 'proprio'):
        del model.extra_encoders.proprio

    transform = img_transform(cfg, img_dtype)

    frame_costs_x_eps = []
    print(f'Extracting costs for {len(selected_episodes)} episodes...')
    for ep_id in tqdm(selected_episodes.tolist()):
        chunk = dataset.load_chunk(
            np.array([ep_id]),
            np.array([0]),
            np.array([cfg.eval.goal_offset_steps]),
        )[0]

        info_dict = {}
        if 'mask' in model.cost_function:
            if 'pusht' in os.path.basename(dataset.h5_file.filename):
                t_mask = torch.stack([generate_pusht_t_mask(torch.tensor(x)) for x in chunk['state']], dim=0)
                ball_mask = torch.stack([generate_pusht_ball_mask(torch.tensor(x)) for x in chunk['state']], dim=0)
                goal_mask = torch.stack((t_mask, ball_mask), dim=1)
            elif 'tworoom' in os.path.basename(dataset.h5_file.filename):
                circle_mask = torch.stack([generate_tworoom_circle_mask(torch.tensor(x)) for x in chunk['proprio']], dim=0)
                goal_mask = circle_mask.unsqueeze(1)
            elif 'reacher' in os.path.basename(dataset.h5_file.filename):
                stick_masks = []
                circle_masks = []
                state = {}
                state['goal_qpos'] = chunk['qpos']
                for idx in range(chunk['pixels'].shape[0]):
                    stick_mask, circle_mask = generate_reacher_masks(state, idx)
                    stick_masks.append(stick_mask)
                    circle_masks.append(circle_mask)
                goal_mask = torch.stack((torch.stack(stick_masks, dim=0), torch.stack(circle_masks, dim=0)), dim=1)
            elif 'cube' in os.path.basename(dataset.h5_file.filename):
                gripper_masks = []
                cube_masks = []
                env_cube = CubeEnv(env_type='single', ob_type='pixels')
                for idx in range(chunk['pixels'].shape[0]):
                    qpos_sample = chunk['qpos'][idx] 
                    env_cube.reset()
                    cube_mask = generate_ogbcube_cube_mask(env_cube, qpos_sample)
                    gripper_mask = generate_ogbcube_gripper_mask(env_cube, qpos_sample)
        
                    cube_masks.append(cube_mask)
                    gripper_masks.append(gripper_mask)
                cube_masks = torch.stack(cube_masks, dim=0)
                goal_mask = cube_masks.unsqueeze(1)
                
            info_dict['goal_mask'] = goal_mask[-1].unsqueeze(0).unsqueeze(0).to(device)
        else:
            with torch.no_grad():
                gt_emb = model._encode_image(transform(chunk['pixels'][-1]).unsqueeze(0).unsqueeze(0).to(device))
                info_dict['pixels_goal_emb'] = gt_emb
        frame_costs = []
        for idx in range(cfg.eval.goal_offset_steps):        
            if 'mask' in model.cost_function:
                info_dict['mask_preds'] = goal_mask[idx].unsqueeze(0).unsqueeze(0).to(device)
            else:
                with torch.no_grad():
                    out = model._encode_image(transform(chunk['pixels'][idx]).unsqueeze(0).unsqueeze(0).to(device))
                info_dict[f'predicted_pixels_emb'] = out.unsqueeze(0)
                
            info_dict['pixels'] = chunk['pixels'][idx].unsqueeze(0).to(device)
            frame_costs.append(model.criterion(info_dict, None))
        frame_costs_x_eps.append(torch.cat(frame_costs, dim=0).squeeze(1))

    avg_costs = torch.stack(frame_costs_x_eps, dim = 0).mean(dim=0)

    print(f'Average costs: {avg_costs}')

    # create output directory if it doesn't exist
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save costs to a file
    costs_file = output_dir / f'costs_{cfg.policy}_{cfg.cost_function}.pt'
    print(f'Saving costs to {costs_file.resolve()}')
    torch.save(avg_costs, costs_file)

if __name__ == '__main__':
    run()