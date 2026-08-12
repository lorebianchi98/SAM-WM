import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# The 'sub_imagenet_template' prompt set from Talk2DINO (datasets/templates.py).
SUB_IMAGENET_TEMPLATE = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

_ACTS = {None: None, 'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}


class _TextProjection(nn.Module):
    """Talk2DINO projection MLP: maps CLIP text embeddings into DINOv2 space.

    Mirrors Talk2DINO's ``ProjectionLayer.project_clip_txt``: a linear layer
    (clip_dim -> dino_dim) followed, for each hidden layer, by ``act`` then a
    linear (dino_dim -> dino_dim). State-dict keys match Talk2DINO's
    (``linear_layer.*``, ``hidden_layers.N.*``).
    """

    def __init__(self, clip_dim, dino_dim, num_hidden=1, act=nn.Tanh()):
        super().__init__()
        self.linear_layer = nn.Linear(clip_dim, dino_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dino_dim, dino_dim) for _ in range(num_hidden)]
        )
        self.act = act

    def forward(self, x):
        x = self.linear_layer(x.float())
        for hidden in self.hidden_layers:
            if self.act is not None:
                x = self.act(x)
            x = hidden(x)
        return x


class Talk2DINOWrapper(torch.nn.Module):
    """Text-guided masks via Talk2DINO's projection in DINOv2 embedding space.

    Talk2DINO learns to project CLIP text embeddings into the DINOv2 patch
    embedding space. Since the world model predicts *future DINO patch
    embeddings* directly, a mask for a text query is just the cosine similarity
    between those predicted embeddings and the projected text embedding -- no
    pixel decoding, image re-encoding, or the Talk2DINO segmentation framework
    needed.

    Usage:
      - ``register_text_embeddings(queries)`` once to cache the projected,
        L2-normalized text embeddings (in DINO space).
      - ``similarity(dino_embeds)`` to get per-query cosine similarity maps.
    """

    def __init__(self, proj_config_path, proj_weights_path, clip_model_name='ViT-B/32',
                 template_set='sub_imagenet_template', binarize_threshold=None,
                 device='cuda'):
        super().__init__()
        import clip

        self.template_set = template_set
        # If set, similarity maps are binarized as (sim >= binarize_threshold),
        # mirroring `evaluate.bg_thresh` in the original repo. If None, the raw
        # similarity map is returned.
        self.binarize_threshold = binarize_threshold

        # -- projection MLP from Talk2DINO's config + weights
        with open(proj_config_path, 'r') as f:
            proj_cfg = yaml.safe_load(f)['model']
        hidden = proj_cfg.get('hidden_layer', False)
        num_hidden = 1 if hidden is True else int(hidden or 0)
        self.proj = _TextProjection(
            clip_dim=proj_cfg.get('clip_embed_dim', 512),
            dino_dim=proj_cfg.get('dino_embed_dim', 384),
            num_hidden=num_hidden,
            act=_ACTS[proj_cfg.get('act', None)],
        )
        state = torch.load(proj_weights_path, map_location='cpu')
        if 'linear_layer2.weight' in state:  # Talk2DINO old-code naming
            state['hidden_layers.0.weight'] = state.pop('linear_layer2.weight')
            state['hidden_layers.0.bias'] = state.pop('linear_layer2.bias')
        self.proj.load_state_dict(state)
        self.proj.eval().requires_grad_(False)

        # -- CLIP text encoder (used only to embed the text queries once)
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval().requires_grad_(False)

        self.text_embeds = None  # (num_texts, dino_dim), L2-normalized
        self.classnames = None
        self.to(device)

    @property
    def device(self):
        return self.proj.linear_layer.weight.device

    def _templates(self):
        if self.template_set in ('sub_imagenet_template', 'subset'):
            return SUB_IMAGENET_TEMPLATE
        raise ValueError(f"Unsupported template_set: {self.template_set}")

    @torch.no_grad()
    def register_text_embeddings(self, text):
        """Encode + project the text queries into DINO space (cached)."""
        import clip

        templates = self._templates()
        embs = []
        for classname in text:
            prompts = [t.format(classname) for t in templates]
            tokens = clip.tokenize(prompts).to(self.device)
            clip_emb = self.clip_model.encode_text(tokens)  # (num_templates, clip_dim)
            embs.append(clip_emb.mean(dim=0))  # average over templates
        clip_embs = torch.stack(embs, dim=0)  # (num_texts, clip_dim)
        dino_embs = self.proj(clip_embs)  # (num_texts, dino_dim)
        self.text_embeds = F.normalize(dino_embs, dim=-1)
        self.classnames = list(text)

    @torch.no_grad()
    def similarity(self, dino_embeds):
        """Cosine similarity between DINO patch embeddings and each text query.

        Args:
            dino_embeds: (N, P, C) DINO patch embeddings (e.g. predicted by the WM).
        Returns:
            (N, num_texts, P) cosine similarities in [-1, 1].
        """
        if self.text_embeds is None:
            raise RuntimeError("Call register_text_embeddings() before similarity()")
        feats = F.normalize(dino_embeds.float(), dim=-1)  # (N, P, C)
        sims = feats @ self.text_embeds.t().to(feats.dtype)  # (N, P, num_texts)
        return sims.permute(0, 2, 1).contiguous()  # (N, num_texts, P)
