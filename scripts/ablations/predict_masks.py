"""Trace world-model mask predictions along expert dataset actions."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
from PIL import Image
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


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


def _stack_episode_batch(chunk, key, history_size, total_size):
    items = []
    for ep in chunk:
        values = ep[key][:total_size] if key == 'action' else ep[key][:history_size]
        items.append(torch.as_tensor(values))
    stacked = torch.stack(items, dim=0)
    return stacked.unsqueeze(1)


def _expand_actions(actions, frameskip):
    """Expand (T, A) actions into (T, A*frameskip) by concatenating future actions."""

    if actions.ndim != 2:
        raise ValueError(
            f'Expected actions to be (T, A), got shape={tuple(actions.shape)}'
        )
    if frameskip <= 1:
        return actions

    time_steps, _ = actions.shape
    chunks = []
    for offset in range(frameskip):
        indices = torch.clamp(
            torch.arange(time_steps, device=actions.device) + offset,
            max=time_steps - 1,
        )
        chunks.append(actions.index_select(0, indices))
    return torch.cat(chunks, dim=-1)


def _average_action_blocks(actions, frameskip):
    """Average contiguous action blocks of length frameskip along the time axis."""

    if actions.ndim != 4:
        raise ValueError(
            f'Expected actions to be (B, C, T, A), got shape={tuple(actions.shape)}'
        )
    if frameskip <= 1:
        return actions

    batch_size, channels, time_steps, action_dim = actions.shape
    n_blocks = (time_steps + frameskip - 1) // frameskip
    padded = torch.zeros(
        (batch_size, channels, n_blocks * frameskip, action_dim),
        device=actions.device,
        dtype=actions.dtype,
    )
    padded[:, :, :time_steps, :] = actions
    reshaped = padded.reshape(batch_size, channels, n_blocks, frameskip, action_dim)
    return reshaped.mean(dim=3)


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


def _overlay_masks(frame, masks):
    frame = _to_uint8_frame(frame)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or frame.shape[:2] != masks.shape[-2:]:
        return frame

    out = frame.astype(np.float32)
    palette = np.array(
        [
            [255, 92, 92],
            [92, 168, 255],
            [101, 204, 122],
            [255, 196, 92],
            [203, 109, 255],
            [92, 228, 222],
        ],
        dtype=np.float32,
    )
    alpha = 0.65

    for mask_idx, mask in enumerate(masks):
        mask = mask.astype(bool)
        if not mask.any():
            continue
        color = palette[mask_idx % len(palette)]
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_masks_heatmap(frame, masks, alpha=0.7):
    """Overlay continuous (non-binarized) mask/similarity maps as a heatmap.

    Each query's map is min-max normalized per frame and blended with a per-query
    color, with opacity proportional to the (normalized) value. Used for soft
    Talk2DINO similarity maps, where a hard boolean overlay would color nearly the
    whole frame. Binary masks are handled by _overlay_masks instead.
    """
    frame = _to_uint8_frame(frame)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or frame.shape[:2] != masks.shape[-2:]:
        return frame

    out = frame.astype(np.float32)
    palette = np.array(
        [
            [255, 92, 92],
            [92, 168, 255],
            [101, 204, 122],
            [255, 196, 92],
            [203, 109, 255],
            [92, 228, 222],
        ],
        dtype=np.float32,
    )

    for mask_idx, mask in enumerate(masks):
        mask = mask.astype(np.float32)
        lo, hi = float(mask.min()), float(mask.max())
        if hi - lo < 1e-6:
            continue
        weight = ((mask - lo) / (hi - lo))[..., None] * alpha  # (H, W, 1)
        color = palette[mask_idx % len(palette)]
        out = (1.0 - weight) * out + weight * color

    return np.clip(out, 0, 255).astype(np.uint8)


def _prepare_pixels(pixels, transform):
    tensor = torch.as_tensor(pixels)
    if tensor.ndim == 5:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 6:
        raise ValueError(
            f'Expected pixels to be (B, 1, T, H, W, C) or (1, T, H, W, C), got shape={tuple(tensor.shape)}'
        )

    shape = tensor.shape
    tensor = tensor.reshape(-1, *shape[3:])
    tensor = torch.stack([transform(frame) for frame in tensor], dim=0)
    tensor = tensor.reshape(*shape[:3], *tensor.shape[1:])
    return tensor


def _apply_process(values, processor):
    values = np.asarray(values)
    shape = values.shape
    if values.ndim == 1:
        flattened = values.reshape(-1, 1)
    else:
        flattened = values.reshape(-1, shape[-1])
    transformed = processor.transform(flattened).astype(np.float32)
    return transformed.reshape(shape)


@hydra.main(version_base=None, config_path='../plan/config', config_name='pusht')
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = build_process(dataset, cfg)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for tracing.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for tracing.'
        )

    policy_name = cfg.get('policy', 'random')
    if policy_name == 'random':
        raise ValueError(
            'This script requires a pretrained world-model checkpoint.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = swm.wm.utils.load_pretrained(
        cfg.policy, n_epoch=cfg.get('n_epoch', None)
    )
    if cfg.get('bf16', False):
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model.cost_function = cfg.get('cost_function', 'mse')

    if not hasattr(model, 'build_mask_decoder'):
        raise ValueError('Loaded model does not support mask decoding.')

    mask_type = cfg.get('mask_type', 'sam3')
    if mask_type == 'talk2dino':
        talk2dino_config = cfg.get('talk2dino_config', {})
        model.build_mask_decoder(
            mask_type='talk2dino',
            textual_query=list(cfg.textual_query),
            talk2dino_config_path=talk2dino_config.get('config_path'),
            talk2dino_template_set=talk2dino_config.get('template_set', 'sub_imagenet_template'),
            talk2dino_bg_thresh=talk2dino_config.get('bg_thresh', None),
            talk2dino_proj_asset_root=talk2dino_config.get('proj_asset_root', None),
        )
    else:
        model.build_mask_decoder(
            inference_resolution=224,
            textual_query=list(cfg.textual_query),
            threshold=cfg.threshold,
            mask_threshold=cfg.mask_threshold,
        )
    model.weight_cost_strategy = cfg.weight_cost_strategy

    if cfg.get('compile', False):
        encoder_attr = 'backbone' if hasattr(model, 'backbone') else 'encoder'
        setattr(
            model,
            encoder_attr,
            torch.compile(getattr(model, encoder_attr)),
        )
        model.predictor = torch.compile(model.predictor)

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )
    video_path = Path(cfg.get('video_path', results_path / 'mask_rollouts'))
    video_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    history_size = model.history_size
    trace_horizon = history_size + int(cfg.eval.goal_offset_steps)
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    pixel_transform = img_transform(cfg, img_dtype)

    chunk = dataset.load_chunk(
        eval_episodes,
        eval_start_idx,
        eval_start_idx + trace_horizon,
    )

    raw_action_dim = int(dataset.get_dim('action'))
    stacked_action_dim = int(model.extra_encoders['action'].in_chans)
    if stacked_action_dim % raw_action_dim != 0:
        raise ValueError(
            'Model action encoder width is not a multiple of the dataset action dimension.'
        )
    frameskip = stacked_action_dim // raw_action_dim

    model_keys = ['pixels'] + [
        key for key in getattr(model, 'extra_encoders', {}).keys() if key != 'action'
    ]
    missing_keys = [key for key in model_keys if key not in dataset.column_names]
    if missing_keys:
        raise ValueError(
            f'Missing dataset columns required by the model: {missing_keys}'
        )

    info = {}
    for key in model_keys:
        info[key] = _stack_episode_batch(
            chunk, key, history_size=history_size, total_size=trace_horizon
        ).to(device)

    info['pixels'] = _prepare_pixels(info['pixels'].cpu(), pixel_transform).to(device)

    for key, processor in process.items():
        if key in info and key != 'pixels':
            processed = _apply_process(info[key].cpu().numpy(), processor)
            info[key] = torch.as_tensor(processed, device=device)

    action_sequence = _stack_episode_batch(
        chunk, 'action', history_size=history_size, total_size=trace_horizon
    )
    if 'action' in process:
        processed_actions = _apply_process(action_sequence.squeeze(1).cpu().numpy(), process['action'])
        action_sequence = torch.as_tensor(processed_actions, dtype=torch.float32).unsqueeze(1)
    else:
        action_sequence = action_sequence.float()
    action_sequence = torch.stack(
        [
            _expand_actions(ep.squeeze(0), frameskip)
            for ep in action_sequence
        ],
        dim=0,
    ).unsqueeze(1).to(device=device, dtype=torch.float32)

    action_sequence = action_sequence[:, :,torch.arange((action_sequence.shape[2] // frameskip) + 1) * frameskip]

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False) and device.type == 'cuda',
    )
    if hasattr(model, '_init_cached_info'):
        del model._init_cached_info
    start_time = time.time()
    with torch.no_grad(), autocast_ctx:
        rollout_info = model.rollout(info, action_sequence)

        predicted = rollout_info['predicted_pixels_emb']
        predicted_mask_steps = []
        for step_idx in range(history_size, predicted.shape[2]):
            mask_info = {
                'pixels': info['pixels'],
                'predicted_pixels_emb': predicted[:, :, : step_idx + 1],
            }
            step_masks = model.get_masks(mask_info, target='predicted_pixels_emb', discretize_mask=True)
            predicted_mask_steps.append(step_masks[:, 0].detach().cpu())

    end_time = time.time()

    # replicate the predicted masks to match the frameskip
    x = torch.stack(predicted_mask_steps, dim=1)
    y = x.unsqueeze(2)
    y = y.expand(-1, -1, frameskip, -1, -1, -1)
    predicted_masks = y.reshape(x.shape[0], frameskip * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

    trace = {
        'episodes_idx': torch.as_tensor(eval_episodes),
        'start_idx': torch.as_tensor(eval_start_idx),
        'history_size': history_size,
        'action_sequence': action_sequence.detach().cpu(),
        'predicted_pixels_emb': rollout_info['predicted_pixels_emb'].detach().cpu(),
        'predicted_masks': predicted_masks,
        'textual_query': list(cfg.textual_query),
    }
    trace_path = video_path / 'mask_trace.pt'

    from stable_worldmodel.plot import save_video

    # Talk2DINO similarity maps are continuous when bg_thresh is unset -> heatmap;
    # binarized (bg_thresh set) and Sam3 masks -> hard boolean overlay.
    soft_overlay = (
        mask_type == 'talk2dino'
        and cfg.get('talk2dino_config', {}).get('bg_thresh', None) is None
    )
    overlay_fn = _overlay_masks_heatmap if soft_overlay else _overlay_masks

    episode_chunks = [ep['pixels'] for ep in chunk]
    episode_masks = trace['predicted_masks']
    for idx, (ep_pixels, ep_masks) in enumerate(zip(episode_chunks, episode_masks)):
        frames = []
        ep_pixels = np.asarray(ep_pixels)
        total_frames = history_size + ep_masks.shape[0]
        for frame_idx in range(total_frames):
            bg_idx = min(frame_idx, ep_pixels.shape[0] - 1)
            frame = _to_uint8_frame(ep_pixels[bg_idx])
            if frame_idx >= history_size:
                mask_idx = frame_idx - history_size
                frame = overlay_fn(frame, ep_masks[mask_idx].numpy())
            frames.append(frame)

        ep_path = video_path / f'episode_{idx}.mp4'
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(ep_path, frames, fps=15)

        ep_frames_path = video_path / f'episode_{idx}'
        ep_frames_path.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            frame_path = ep_frames_path / f'frame_{frame_idx}.png'
            Image.fromarray(frame).save(frame_path)

    print(f'[trace] saved mask trace to {trace_path.resolve()}')
    print(f'[trace] saved videos to {video_path.resolve()}')

    results_file = results_path / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open('a') as f:
        f.write('\n')
        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')
        f.write('==== RESULTS ====\n')
        f.write(f'trace_path: {trace_path}\n')
        f.write(f'video_path: {video_path}\n')
        f.write(f'tracing_time: {end_time - start_time} seconds\n')

# python scripts/plan/eval_wm_masks.py policy=pusht_sam3_psmall eval.num_eval=50 video_path=video/pusht cost_function=mask --config-name pusht
if __name__ == '__main__':
    run()