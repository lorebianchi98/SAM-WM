import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from transformers import Sam3VisionModel
from stable_worldmodel.masks import Sam3Wrapper, calculate_giou, calculate_iou
from stable_worldmodel.masks.sam3 import post_process_instance_segmentation_nothresh
class PreJEPA(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        decoder=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
    ):
        super().__init__()

        self.backbone = encoder
        self.predictor = predictor
        self.extra_encoders = extra_encoders or {}
        self.decoder = decoder
        self.history_size = history_size
        self.num_pred = num_pred

        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.use_masks = False
    
    def build_mask_decoder(self, inference_resolution=224, textual_query=['grey block', 'ball'], threshold=0.08, mask_threshold=0.4,
                          mask_type='sam3', talk2dino_config_path=None, talk2dino_template_set='sub_imagenet_template',
                          talk2dino_bg_thresh=None, talk2dino_proj_asset_root=None):
        device = next(self.parameters()).device
        self.mask_type = mask_type

        if mask_type == 'sam3':
            self.mask_decoder = Sam3Wrapper(
                inference_resolution=inference_resolution
            ).to(device)
            self.mask_decoder.register_text_embeddings(textual_query)
            self.threshold = threshold
            self.mask_threshold = mask_threshold

        elif mask_type == 'talk2dino':
            # Talk2DINO projects CLIP text embeddings into DINO space; masks are the
            # cosine similarity with the predicted DINO embeddings (see _get_masks_talk2dino).
            import os
            import yaml
            from stable_worldmodel.masks import Talk2DINOWrapper
            if talk2dino_config_path is None:
                raise ValueError("talk2dino_config_path must be provided when mask_type='talk2dino'")

            with open(talk2dino_config_path, 'r') as f:
                t2d_cfg = yaml.safe_load(f)['model']
            asset_root = os.path.dirname(talk2dino_config_path)
            proj_config_path = os.path.join(asset_root, 'configs', f"{t2d_cfg['proj_class']}.yaml")
            proj_weights_path = os.path.join(asset_root, 'weights', f"{t2d_cfg['proj_name']}.pth")

            self.mask_decoder = Talk2DINOWrapper(
                proj_config_path=proj_config_path,
                proj_weights_path=proj_weights_path,
                clip_model_name=t2d_cfg.get('clip_model_name', 'ViT-B/32'),
                template_set=talk2dino_template_set,
                binarize_threshold=talk2dino_bg_thresh,
                device=device,
            ).to(device)
            self.mask_decoder.register_text_embeddings(textual_query)

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        self.use_masks = True
        
    def get_masks(self, info, target='emb', discretize_mask=False):
        if getattr(self, 'mask_type', 'sam3') == 'talk2dino':
            return self._get_masks_talk2dino(info, target=target, discretize_mask=discretize_mask)
        # mask_decoder_outs = self.mask_decoder(info[target].squeeze(dim=1))
        b, t, _, hw, c = info[target].shape
        mask_decoder_outs = self.mask_decoder(info[target][:, :, -1:].reshape(b*t, hw, c))
        merged_masks = []
        for mask_decoder_out in mask_decoder_outs:
            if discretize_mask:
                # Apply thresholding and instance merging to get the final masks
                postprocessed_results = self.mask_decoder.processor.post_process_instance_segmentation(
                    mask_decoder_out,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=[list(info['pixels'].shape[-2:])] * b*t # replicate per batch size the image dimensions
                )
                
                merged_mask_x_text = []
                for postprocessed_result in postprocessed_results:
                    merged_mask = torch.zeros(info['pixels'].shape[-2:], dtype=torch.uint8).to(info['pixels'].device)
                    for mask in postprocessed_result["masks"]:
                        merged_mask = torch.logical_or(merged_mask, mask)
                    merged_mask_x_text.append(merged_mask)
                merged_mask_x_text = torch.stack(merged_mask_x_text, dim = 0).unsqueeze(dim=1)
                merged_masks.append(merged_mask_x_text)
            else:
                postprocessed_results = post_process_instance_segmentation_nothresh(
                    mask_decoder_out,
                    threshold=0,
                    target_sizes=[list(info['pixels'].shape[-2:])] * b*t # replicate per batch size the image dimensions
                )
                merged_mask_x_text = []
                for postprocessed_result in postprocessed_results:
                    nothresh_mask = torch.einsum('n, nhw -> hw', postprocessed_result["scores"].float(), postprocessed_result["masks"].float()) / postprocessed_result["scores"].sum()
                    merged_mask_x_text.append(nothresh_mask)
                merged_masks.append(torch.stack(merged_mask_x_text, dim=0).unsqueeze(dim=1))
        merged_masks = torch.stack(merged_masks, dim=2)
        return merged_masks

    def _get_masks_talk2dino(self, info, target='emb', discretize_mask=False):
        """Generate masks from predicted DINO embeddings via Talk2DINO.

        Talk2DINO projects CLIP text embeddings into DINOv2 space, so a mask is
        the cosine similarity between the predicted DINO patch embeddings and the
        projected text embeddings -- no pixel decoding needed. Similarity maps are
        upsampled to the image resolution and returned as (E, 1, num_texts, H, W),
        matching the Sam3 contract. Binarized when a threshold is set: `bg_thresh`
        (the wrapper's binarize_threshold) takes precedence
        """
        b, t, _, hw, c = info[target].shape

        # Predicted DINO patch embeddings for the last step: (E, hw, c)
        feats = info[target][:, :, -1:].reshape(b * t, hw, c)

        # Cosine similarity with the projected text embeddings: (E, num_texts, hw)
        sims = self.mask_decoder.similarity(feats)
        e, num_texts, _ = sims.shape
        grid = int(hw ** 0.5)
        sims = sims.reshape(e, num_texts, grid, grid)

        # Upsample to the image / goal-mask resolution
        out_hw = tuple(info['pixels'].shape[-2:])
        sims = F.interpolate(sims, size=out_hw, mode='bilinear', align_corners=False)

        thresh = self.mask_decoder.binarize_threshold
        if thresh is None and discretize_mask:
            thresh = 0.5
        masks = (sims >= thresh).to(sims.dtype) if thresh is not None else sims

        return masks.unsqueeze(1)  # (E, 1, num_texts, H, W)

    def encode(
        self,
        info,
        pixels_key='pixels',
        emb_keys=None,
        prefix=None,
        target='emb',
        is_video=False,
    ):
        assert target not in info, f'{target} key already in info_dict'
        emb_keys = self.extra_encoders.keys() if emb_keys is None else emb_keys
        prefix = prefix or ''

        encode_fn = self._encode_video if is_video else self._encode_image
        pixels_embed = encode_fn(info[pixels_key].float())  # (B, T, 3, H, W)

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f'pixels_{target}'] = pixels_embed

        for key in emb_keys:
            extr_enc = self.extra_encoders[key]
            extra_input = info[f'{prefix}{key}'].float()  # (B, T, dim)
            extra_embed = extr_enc(
                extra_input
            )  # (B, T, dim) -> (B, T, emb_dim)
            info[f'{key}_{target}'] = extra_embed

            # copy extra embedding across patches for each time step
            extra_tiled = repeat(
                extra_embed.unsqueeze(2), 'b t 1 d -> b t p d', p=n_patches
            )

            # concatenate along feature dimension
            embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)
        return info

    def _encode_image(self, pixels):
        # == pixels embedding
        B = pixels.shape[0]
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )
        pixels_embed = self.backbone(pixels, **kwargs)

        if self.backbone.__class__ is Sam3VisionModel:
            pixels_embed = pixels_embed['last_hidden_state']

        elif hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            num_additional_tokens = 5 if 'with_registers' in str(self.backbone.__class__) else 1
            pixels_embed = pixels_embed[:, num_additional_tokens:, :]  # drop cls token and additional tokens
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(
            pixels_embed.detach(), '(b t) p d -> b t p d', b=B
        )

        return pixels_embed

    def _encode_video(self, pixels):
        B, T, C, H, W = pixels.shape
        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )

        pixels_embeddings = []

        # roll the embedding computation over time
        for t in range(T):
            padding = max(T - (t + 1), 0)  # number of frames to pad
            past_frames = pixels[:, : t + 1, :, :, :]  # (B, t+1, C, H, W)

            # repeat last frame to pad
            pad_frames = past_frames[:, -1:, :, :, :].repeat(
                1, padding, 1, 1, 1
            )  # (B, padding, C, H, W)
            frames = torch.cat(
                [past_frames, pad_frames], dim=1
            )  # (B, T, C, H, W)

            frame_embed = self.backbone(frames, **kwargs)  # (B, 1, P, emb_dim)
            frame_embed = frame_embed.last_hidden_state
            pixels_embeddings.append(frame_embed)

        pixels_embed = torch.stack(
            pixels_embeddings, dim=1
        )  # (B, T, P, emb_dim)

        return pixels_embed

    def predict(self, embedding):
        """predict next latent state
        Args:
            embedding: (B, T, P, d)
        Returns:
            preds: (B, T, P, d)
        """

        T = embedding.shape[1]
        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        preds = self.predictor(embedding)
        preds = rearrange(preds, 'b (t p) d -> b t p d', t=T)

        return preds

    def decode(self, info):
        assert 'pixels_emb' in info, 'pixels_emb not in info_dict'
        pixels_embed = info['pixels_emb']
        num_frames = pixels_embed.shape[1]

        pixels, diff = self.decoder(
            pixels_embed
        )  # (b*num_frames, 3, 224, 224)
        pixels = rearrange(pixels, '(b t) c h w -> b t c h w', t=num_frames)

        info['reconstructed_pixels'] = pixels
        info['reconstruction_diff'] = diff

        return info

    def split_embedding(
        self, embedding, extra_dims
    ):  # action_dim, proprio_dim):
        split_embed = {}
        pixel_dim = embedding.shape[-1] - sum(extra_dims)

        # == pixels embedding
        split_embed['pixels_emb'] = embedding[..., :pixel_dim]

        # == extra embeddings
        start_dim = pixel_dim
        for i, (key, _) in enumerate(self.extra_encoders.items()):
            dim = extra_dims[i]
            extra_emb = embedding[..., start_dim : start_dim + dim]
            split_embed[f'{key}_emb'] = extra_emb[
                :, :, :, 0
            ]  # all patches are the same
            start_dim += dim

        return split_embed

    def replace_action_in_embedding(self, embedding, act):
        """Replace the action embeddings in the latent state z with the provided actions."""
        assert 'action' in self.extra_encoders, (
            'No action encoder defined in the model.'
        )
        n_patches = embedding.shape[3]
        B, N = act.shape[:2]
        act_flat = rearrange(act, 'b n ... -> (b n) ...')
        z_act = self.extra_encoders['action'](act_flat)  # (B, T, action_dim)
        action_dim = z_act.shape[-1]
        act_tiled = repeat(
            z_act.unsqueeze(2),
            '(b n) t 1 a -> b n t p a',
            b=B,
            n=N,
            p=n_patches,
        )
        # z (B, N, T, P, d) with d = dim + extra_dims
        # determine where action starts in the embedding
        extra_dim = sum(
            encoder.emb_dim for encoder in self.extra_encoders.values()
        )
        pixel_dim = embedding.shape[-1] - extra_dim

        start = pixel_dim
        for key, encoder in self.extra_encoders.items():
            if key == 'action':
                break
            start += encoder.emb_dim

        prefix = embedding[..., :start]
        suffix = embedding[..., start + action_dim :]

        new_embedding = torch.cat([prefix, act_tiled, suffix], dim=-1)

        # embedding[..., start : start + action_dim] = act_tiled
        # return embedding
        return new_embedding

    def rollout(self, info, action_sequence):
        """Rollout the world model given an initial observation and a sequence of actions.

        Params:
        obs_start: n current observations (B, n, C, H, W)
        actions: current and predicted actions (B, n+t, action_dim)

        Returns:
        z_obs: dict with latent observations (B, n+t+1, n_patches, D)
        z: predicted latent states (B, n+t+1, n_patches, D)
        """

        assert 'pixels' in info, 'pixels not in info_dict'
        n_obs = info['pixels'].shape[2]
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == add action to info dict
        act_0 = action_sequence[:, :, :n_obs]
        info['action'] = act_0

        # check if we have already computed the initial embedding for this state
        if (
            hasattr(self, '_init_cached_info')
            and torch.equal(self._init_cached_info['id'], info['id'][:, 0])
            and torch.equal(
                self._init_cached_info['step_idx'], info['step_idx'][:, 0]
            )
        ):
            init_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._init_cached_info.items()
            }
        else:
            # prepare init_info_dict
            init_info_dict = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    init_info_dict[k] = info[k][:, 0]  # (B, 1, ...)

            init_info_dict = self.encode(
                init_info_dict,
                pixels_key='pixels',
                target='emb',
            )
            # repeat copy for each action candidate
            init_info_dict['emb'] = (
                init_info_dict['emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['emb'].ndim - 1)),
                )
                .clone()
            )
            init_info_dict['pixels_emb'] = (
                init_info_dict['pixels_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['pixels_emb'].ndim - 1)),
                )
            )

            for key in emb_keys:
                init_info_dict[f'{key}_emb'] = (
                    init_info_dict[f'{key}_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_sequence.shape[1],
                        *([-1] * (init_info_dict[f'{key}_emb'].ndim - 1)),
                    )
                )

            init_info_dict = {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in init_info_dict.items()
            }
            self._init_cached_info = init_info_dict

        info['emb'] = init_info_dict['emb']
        info['pixels_emb'] = init_info_dict['pixels_emb']

        for key in emb_keys:
            info[f'{key}_emb'] = init_info_dict[f'{key}_emb']

        # actually compute the embedding of action for each candidate
        info['emb'] = self.replace_action_in_embedding(
            info['emb'], action_sequence[:, :, :n_obs]
        )

        action_dim = init_info_dict['action_emb'].shape[-1]
        info['action_emb'] = info['emb'][:, :, :n_obs, 0, -action_dim:]

        # number of step to predict
        act_pred = action_sequence[:, :, n_obs:]
        n_steps = act_pred.shape[2]

        # == initial embedding
        z = info['emb']
        B, N = z.shape[:2]

        # we flatten B and N to process all candidates in a single batch in the predictor
        z_flat = rearrange(z, 'b n ... -> (b n) ...').clone()
        act_pred_flat = rearrange(act_pred, 'b n ... -> (b n) ...')

        for t in range(n_steps):
            # predict the next state
            pred_embed = self.predict(z_flat[:, -self.history_size :])[
                :, -1:
            ]  # (B*N, 1, P, D)

            # add corresponding action to new embedding
            new_action = act_pred_flat[
                None, :, t : t + 1, :
            ]  # (1, B*N, 1, action_dim)
            new_embed = self.replace_action_in_embedding(
                pred_embed.unsqueeze(0), new_action
            )[0]

            # append new embedding to the sequence
            z_flat = torch.cat([z_flat, new_embed], dim=1)

        # predict the last state (n+t+1)
        pred_embed = self.predict(z_flat[:, -self.history_size :])[
            :, -1:
        ]  # (B, 1, P, D)
        z_flat = torch.cat([z_flat, pred_embed], dim=1)
        z = rearrange(z_flat, '(b n) ... -> b n ...', b=B, n=N)
        # == update info dict with predicted embeddings
        info['predicted_embedding'] = z

        extra_dims = []
        for key in self.extra_encoders:
            if f'{key}_emb' not in info:
                raise ValueError(f'{key}_emb not in info dict')
            extra_dims.append(info[f'{key}_emb'].shape[-1])

        splitted_embed = self.split_embedding(z, extra_dims)
        info.update({f'predicted_{k}': v for k, v in splitted_embed.items()})

        return info

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost for planning. Should be overridden for custom costs."""
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']
        cost = 0.0

        if self.use_masks:
            if self.cost_function in ('mask', 'mask_iou'):
                if 'mask_preds' not in info_dict:
                    mask_preds = self.get_masks(info_dict, target='predicted_pixels_emb', discretize_mask=True)
                else:
                    mask_preds = info_dict['mask_preds']
                gious = []

                goal_masks = info_dict['goal_mask']
                *_, n_texts, h, w = goal_masks.shape 
                flattened_goals = goal_masks.reshape(-1, n_texts, h, w)

                # calculate weights for each goal mask based on the specified strategy
                if self.weight_cost_strategy == 'area':
                    episodes_weights = flattened_goals.sum(dim=(2, 3)) / flattened_goals.sum(dim=(1, 2, 3)).unsqueeze(1)
                else:
                    episodes_weights = torch.ones(flattened_goals.shape[:2])

                for mask_pred, goal_mask, episode_weight in zip(mask_preds.squeeze(1), flattened_goals, episodes_weights):
                    # Calculate the selected overlap cost for each text query and sum the results.
                    gious_x_text = []
                    for mask_pred_text, goal_mask_text, weight_text in zip(mask_pred, goal_mask, episode_weight):
                        overlap = (
                            calculate_iou(mask_pred_text, goal_mask_text)
                            if self.cost_function == 'mask_iou'
                            else calculate_giou(mask_pred_text, goal_mask_text)
                        )
                        gious_x_text.append((1 - overlap) * weight_text)
                    gious.append(sum(gious_x_text))
                cost = torch.tensor(gious).reshape(info_dict['goal_mask'].shape[0], info_dict['goal_mask'].shape[1])
            elif self.cost_function in ('mask_mse', 'mask_mse-thresh'):
                if 'mask_preds' not in info_dict:
                    mask_preds = self.get_masks(
                        info_dict,
                        target='predicted_pixels_emb',
                        discretize_mask=self.cost_function == 'mask_mse-thresh',
                    )
                    if self.cost_function == 'mask_mse-thresh':
                        mask_preds = mask_preds.float()
                else:
                    mask_preds = info_dict['mask_preds']
                cost = F.mse_loss(mask_preds.unsqueeze(0), info_dict['goal_mask'], reduction='none').mean(dim=tuple(range(2, len(mask_preds.shape) + 1)))
        else:
            for key in emb_keys + ['pixels']:
                preds = info_dict[f'predicted_{key}_emb']
                goal = info_dict[f'{key}_goal_emb']
                cost = cost + F.mse_loss(
                    preds[:, :, -1:], goal, reduction='none'
                ).mean(dim=tuple(range(2, preds.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert 'action' in info_dict, 'action key must be in info_dict'
        assert 'pixels' in info_dict, 'pixels key must be in info_dict'

        # == non action embeddings keys
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == get the goal embedding

        # check if we have already computed the goal embedding for this goal
        if (
            hasattr(self, '_goal_cached_info')
            and torch.equal(
                self._goal_cached_info['id'], info_dict['id'][:, 0]
            )
            and torch.equal(
                self._goal_cached_info['step_idx'], info_dict['step_idx'][:, 0]
            )
        ):
            goal_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._goal_cached_info.items()
            }

        else:
            # prepare goal_info_dict
            goal_info_dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    goal_info_dict[k] = info_dict[k][:, 0]  # (B, ...)
            if not self.use_masks:
                goal_info_dict = self.encode(
                    goal_info_dict,
                    target='goal_emb',
                    pixels_key='goal',
                    prefix='goal_',
                    emb_keys=emb_keys,
                )
                goal_info_dict['goal_emb'] = (
                    goal_info_dict['goal_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_candidates.shape[1],
                        *([-1] * (goal_info_dict['goal_emb'].ndim - 1)),
                    )
                )

                goal_info_dict['pixels_goal_emb'] = (
                    goal_info_dict['pixels_goal_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_candidates.shape[1],
                        *([-1] * (goal_info_dict['pixels_goal_emb'].ndim - 1)),
                    )
                )

                for key in emb_keys:
                    goal_info_dict[f'{key}_goal_emb'] = (
                        goal_info_dict[f'{key}_goal_emb']
                        .unsqueeze(1)
                        .expand(
                            -1,
                            action_candidates.shape[1],
                            *([-1] * (goal_info_dict[f'{key}_goal_emb'].ndim - 1)),
                        )
                    )

                goal_info_dict = {
                    k: v.detach() if torch.is_tensor(v) else v
                    for k, v in goal_info_dict.items()
                }
                self._goal_cached_info = goal_info_dict

                info_dict['goal_emb'] = goal_info_dict['goal_emb']
                info_dict['pixels_goal_emb'] = goal_info_dict['pixels_goal_emb']

                for key in emb_keys:
                    info_dict[f'{key}_goal_emb'] = goal_info_dict[f'{key}_goal_emb']

        # == run world model
        info_dict = self.rollout(info_dict, action_candidates)

        # cost = 0.0

        # for key in emb_keys + ["pixels"]:
        #     preds = info_dict[f"predicted_{key}_embed"]
        #     goal = info_dict[f"{key}_goal_embed"]
        #     cost = cost + F.mse_loss(preds[:, :, -1:], goal, reduction="none").mean(dim=tuple(range(2, preds.ndim)))

        return self.criterion(info_dict, action_candidates)


__all_ = ['PreJEPA']
