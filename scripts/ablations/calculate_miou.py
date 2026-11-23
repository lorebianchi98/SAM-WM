"""Evaluate segmentation mIoU over a grid of decoder thresholds."""

from pathlib import Path
from itertools import islice

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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


def compute_miou(predicted_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """Return mIoU for each query, averaging only non-empty target unions."""
    if predicted_masks.shape != target_masks.shape:
        raise ValueError(
            'Predicted and target masks must have the same shape, got '
            f'{tuple(predicted_masks.shape)} and {tuple(target_masks.shape)}.'
        )

    predicted_masks = predicted_masks.bool()
    target_masks = target_masks.bool()
    intersection = (predicted_masks & target_masks).sum(dim=(-2, -1)).float()
    union = (predicted_masks | target_masks).sum(dim=(-2, -1)).float()
    valid = union > 0
    iou = torch.where(valid, intersection / union.clamp_min(1), torch.zeros_like(union))
    valid_count = valid.sum(dim=0).float()
    return torch.where(
        valid_count > 0,
        iou.sum(dim=0) / valid_count.clamp_min(1),
        torch.full_like(valid_count, float('nan')),
    )



def _episode_masks(episode: dict, dataset_name: str) -> torch.Tensor:
    """Build [frames, queries, height, width] masks for one episode."""
    if 'pusht' in dataset_name:
        states = episode['state']
        t_mask = torch.stack([generate_pusht_t_mask(torch.as_tensor(x)) for x in states])
        ball_mask = torch.stack(
            [generate_pusht_ball_mask(torch.as_tensor(x)) for x in states]
        )
        return torch.stack((t_mask, ball_mask), dim=1)

    if 'tworoom' in dataset_name:
        masks = [
            generate_tworoom_circle_mask(torch.as_tensor(x))
            for x in episode['proprio']
        ]
        return torch.stack(masks, dim=0).unsqueeze(1)

    if 'reacher' in dataset_name:
        state = {'goal_qpos': episode['qpos']}
        masks = [generate_reacher_masks(state, idx) for idx in range(len(episode['qpos']))]
        return torch.stack(
            (torch.stack([mask[0] for mask in masks]), torch.stack([mask[1] for mask in masks])),
            dim=1,
        )

    if 'cube' in dataset_name:
        env = CubeEnv(env_type='single', ob_type='pixels')
        cube_masks = []
        for qpos in episode['qpos']:
            env.reset()
            cube_masks.append(generate_ogbcube_cube_mask(env, qpos))
        return torch.stack(cube_masks, dim=0).unsqueeze(1)

    raise ValueError(f'Unsupported dataset for mask extraction: {dataset_name}')


def _load_episodes(cfg: DictConfig):
    # Load the training config to get the exact validation split.
    if 'pusht' in cfg.eval.dataset_name:
        train_config = 'scripts/train/config/prejepa_pusht.yaml'
    elif 'tworoom' in cfg.eval.dataset_name:
        train_config = 'scripts/train/config/prejepa_tworoom.yaml'
    elif 'reacher' in cfg.eval.dataset_name:
        train_config = 'scripts/train/config/prejepa_reacher.yaml'
    elif 'cube' in cfg.eval.dataset_name:
        train_config = 'scripts/train/config/prejepa_cube.yaml'
    else:
        raise ValueError(f'Unsupported dataset for miou extraction: {cfg.eval.dataset_name}')
    train_cfg_path = hydra.utils.to_absolute_path(train_config)
    train_cfg = OmegaConf.load(train_cfg_path)
    dataset = swm.data.load_dataset(
        train_cfg.dataset_name,
        num_steps=train_cfg.n_steps,
        frameskip=train_cfg.frameskip,
    )
    dataset_name = cfg.eval.dataset_name
    split_generator = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(
        dataset,
        [train_cfg.train_split, 1 - train_cfg.train_split],
        generator=split_generator,
    )
    num_samples = int(cfg.get('miou_num_samples', 500))
    if len(val_set) < num_samples:
        raise ValueError(
            f'Found {len(val_set)} validation episodes, but '
            f'miou_num_samples={num_samples}.'
        )

    episodes = [val_set[idx] for idx in range(num_samples)]
    return episodes, dataset_name


def _build_model(cfg: DictConfig):
    if cfg.get('policy', 'random') == 'random':
        raise ValueError('This script requires a pretrained world-model checkpoint.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = swm.wm.utils.load_pretrained(
        cfg.policy, n_epoch=cfg.get('n_epoch', None)
    ).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    mask_type = cfg.get('mask_type', 'sam3')
    if mask_type == 'talk2dino':
        talk2dino = cfg.get('talk2dino_config', {})
        model.build_mask_decoder(
            mask_type='talk2dino',
            textual_query=list(cfg.textual_query),
            talk2dino_config_path=talk2dino.get('config_path'),
            talk2dino_template_set=talk2dino.get(
                'template_set', 'sub_imagenet_template'
            ),
            talk2dino_bg_thresh=talk2dino.get('bg_thresh'),
            talk2dino_proj_asset_root=talk2dino.get('proj_asset_root'),
        )
    else:
        model.build_mask_decoder(
            inference_resolution=224,
            textual_query=list(cfg.textual_query),
            threshold=cfg.threshold,
            mask_threshold=cfg.mask_threshold,
        )
    return model, device



def _threshold_grid(cfg: DictConfig
                    ) -> torch.Tensor:
    if cfg.mask_type == 'talk2dino':
        step = 0.01
        thresh_min = 0.0
        thresh_max = 0.0
        mask_thresh_min = -0.2
        mask_thresh_max = 0.3
    else:
        step = 0.02
        thresh_min = 0.0
        thresh_max = 0.2
        mask_thresh_min = 0.36
        mask_thresh_max = 0.8
    thresholds = np.arange(
        float(cfg.get('miou_threshold_min', thresh_min)),
        float(cfg.get('miou_threshold_max', thresh_max)) + step / 2,
        step,
    )
    mask_thresholds = np.arange(
        float(cfg.get('miou_mask_threshold_min', mask_thresh_min)),
        float(cfg.get('miou_mask_threshold_max', mask_thresh_max)) + step / 2,
        step,
    )
    return torch.tensor(
        [(mask, threshold) for mask in mask_thresholds for threshold in thresholds],
        dtype=torch.float32,
    )


@hydra.main(version_base=None, config_path='../plan/config', config_name='pusht')
def run(cfg: DictConfig):
    """Calculate threshold-wise mask mIoU for sampled dataset episodes."""
    episodes, dataset_name = _load_episodes(cfg)
    print("Building target masks for episodes...")
    target_masks = torch.stack(
        [_episode_masks(ep, dataset_name)[0] for ep in tqdm(episodes)]
    ).unsqueeze(1)
    model, device = _build_model(cfg)
    
    mask_type = cfg.get('mask_type', 'sam3')
    print(f"Calculating mIoU for {len(episodes)} episodes with mask type '{mask_type}'...")
    threshold_grid = _threshold_grid(cfg)
    batch_size = int(cfg.get('miou_batch_size', 32))
    if batch_size < 1:
        raise ValueError(f'miou_batch_size must be positive, got {batch_size}.')
    predictions_by_threshold = [[] for _ in threshold_grid]
    episode_iterator = iter(episodes)
    autocast_enabled = cfg.get('bf16', False) and device.type == 'cuda'
    autocast_ctx = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    )

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(episodes), batch_size), desc='Episode batches'
        ):
            episode_batch = list(islice(episode_iterator, batch_size))
            if not episode_batch:
                break

            current_batch_size = len(episode_batch)
            batch_pixels = torch.stack(
                [ep['pixels'][:1] for ep in episode_batch], dim=0
            ).to(device)
            with autocast_ctx:
                encoded = model._encode_image(batch_pixels)
                mask_info = {
                    'pixels': batch_pixels,
                    'predicted_pixels_emb': encoded.unsqueeze(1),
                }
                for pair_idx, (mask_threshold, threshold) in enumerate(threshold_grid):
                    if mask_type == 'talk2dino':
                        model.mask_decoder.binarize_threshold = float(mask_threshold)
                    model.mask_threshold = float(mask_threshold)
                    model.threshold = float(threshold)
                    predicted = model.get_masks(
                        mask_info,
                        target='predicted_pixels_emb',
                        discretize_mask=True,
                    )
                    predicted = predicted.reshape(
                        current_batch_size, 1, *predicted.shape[1:]
                    )
                    predictions_by_threshold[pair_idx].append(
                        predicted.squeeze(2).detach().cpu()
                    )

    scores = torch.stack(
        [
            compute_miou(torch.cat(predictions), target_masks).mean()
            for predictions in predictions_by_threshold
        ]
    )
    output_path = Path(
        cfg.get(
            'miou_results_path',
            f"checkpoints/{Path(cfg.eval.dataset_name).stem}_mious_x_thresh_{mask_type}.pt",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, output_path)

    best_idx = int(torch.argmax(scores).item())
    best_mask, best_threshold = threshold_grid[best_idx].tolist()
    best_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if mask_type == 'talk2dino':
        best_config.talk2dino_config.bg_thresh = round(float(best_mask), 2)
    else:
        best_config.mask_threshold = round(float(best_mask), 2)
        best_config.threshold = round(float(best_threshold), 2)
    used_config_name = HydraConfig.get().job.config_name
    used_config_path = Path(
        hydra.utils.get_original_cwd(),
        'scripts',
        'plan',
        'config',
        f'{used_config_name}.yaml',
    )
    best_config_path = Path(
        cfg.get(
            'best_config_path',
            str(
                used_config_path.with_name(
                    f'{used_config_path.stem}{used_config_path.suffix}'
                )
            ),
        )
    )
    best_config_path = Path(hydra.utils.to_absolute_path(best_config_path))
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(best_config, best_config_path)

    print(f'Best mask_threshold: {best_mask:.4f}')
    print(f'Best threshold: {best_threshold:.4f}')
    print(f'Best mIoU: {scores[best_idx].item():.6f}')
    print(f'Saved scores to: {output_path}')
    print(f'Saved config to: {best_config_path}')


if __name__ == '__main__':
    run()