from __future__ import annotations

from pathlib import Path

import numpy as np


def save_video(path: Path, frames: list[np.ndarray], fps: int = 15) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    out = imageio.get_writer(str(path), fps=fps, codec='libx264')
    for f in frames:
        out.append_data(f)
    out.close()


def _overlay_goal_masks(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        return frame

    if frame.shape[:2] != masks.shape[-2:]:
        return frame

    if frame.dtype.kind in {'u', 'i'}:
        out = frame.astype(np.float32)
        scale = float(np.iinfo(frame.dtype).max)
    else:
        out = frame.astype(np.float32)
        scale = 1.0 if float(np.max(frame)) <= 1.5 else 255.0

    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=2)

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
    ) / 255.0 * scale
    alpha = 0.7

    for mask_idx, mask in enumerate(masks):
        mask = mask.astype(bool)
        if not mask.any():
            continue
        color = palette[mask_idx % len(palette)]
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    if frame.dtype.kind in {'u', 'i'}:
        out = np.clip(out, 0, scale).astype(frame.dtype)
    else:
        out = np.clip(out, 0.0, scale).astype(frame.dtype)
    return out


def _frame_to_png_array(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype.kind in {'u', 'i'}:
        if frame.dtype == np.uint8:
            return frame
        scale = float(np.iinfo(frame.dtype).max)
        frame = frame.astype(np.float32) / scale * 255.0
    else:
        frame = frame.astype(np.float32)
        scale = 1.0 if float(np.max(frame)) <= 1.5 else 255.0
        if scale == 1.0:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_panel_videos(
    video_dir,
    panels,
    fps: int = 15,
    goal_mask=None,
    episode_successes=None,
    save_frames: bool = False,
) -> None:
    """Save one mp4 per env with labeled panels side-by-side.

    ``panels`` maps a label to per-env data indexable by env index. Each
    per-env entry is either a ``(T, H, W, C)`` sequence or a ``(H, W, C)``
    still that is repeated for every frame. When ``goal_mask`` is provided,
    each env's masks are overlaid on every panel frame before composition. If
    ``save_frames`` is True, the raw ``agent`` and ``dataset`` panel frames are
    also written as PNGs under ``env_{i}_{status}/``.
    """
    from PIL import Image, ImageDraw, ImageFont

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    for video_path in video_dir.glob('*.mp4'):
        video_path.unlink()

    labels = list(panels)
    n_envs = len(panels[labels[0]])

    sample = np.asarray(panels[labels[0]][0])
    h, w = sample.shape[1:3] if sample.ndim == 4 else sample.shape[:2]
    n = len(labels)
    pad, gap, lh = max(12, w // 14), max(10, w // 16), max(22, w // 9)
    cw = (2 * pad + n * w + (n - 1) * gap + 15) // 16 * 16
    ch = (2 * pad + h + lh + 15) // 16 * 16
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(12, w // 14))
    except OSError:
        font = ImageFont.load_default()
    y_text = pad + h + max(8, lh // 4)

    for i in range(n_envs):
        env_panels = [np.asarray(panels[label][i]) for label in labels]
        env_goal_mask = None if goal_mask is None else np.asarray(goal_mask[i])
        T = max((len(p) for p in env_panels if p.ndim == 4), default=1)
        frame_dir = None
        if save_frames:
            status = 'unknown'
            if episode_successes is not None:
                status = 'success' if episode_successes[i] else 'failure'
            frame_dir = video_dir / f'env_{i}_{status}'
            frame_dir.mkdir(parents=True, exist_ok=True)
        composed = []
        for t in range(T):
            c = np.full((ch, cw, 3), 250, dtype=np.uint8)
            for j, p in enumerate(env_panels):
                frame = p[min(t, len(p) - 1)] if p.ndim == 4 else p
                display_frame = frame
                if env_goal_mask is not None and labels[j] != 'goal':
                    display_frame = _overlay_goal_masks(frame, env_goal_mask)
                if save_frames and labels[j] in {'agent', 'dataset'}:
                    assert frame_dir is not None
                    Image.fromarray(_frame_to_png_array(display_frame)).save(
                        frame_dir / f'frame{t}_{labels[j]}.png'
                    )
                x = pad + j * (w + gap)
                c[pad : pad + h, x : x + w] = display_frame
            img = Image.fromarray(c)
            draw = ImageDraw.Draw(img)
            for j, label in enumerate(labels):
                b = draw.textbbox((0, 0), label, font=font)
                x = pad + j * (w + gap) + w // 2 - (b[2] - b[0]) // 2
                draw.text((x, y_text), label, fill=(130, 130, 130), font=font)
            composed.append(np.array(img))
        if episode_successes is None:
            save_video(video_dir / f'env_{i}.mp4', composed, fps=fps)
        else:
            success = episode_successes[i]
            status = 'success' if success else 'failure'
            save_video(video_dir / f'env_{i}_{status}.mp4', composed, fps=fps)