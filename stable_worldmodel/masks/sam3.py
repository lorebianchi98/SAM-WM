import torch
from transformers import Sam3Processor, Sam3Model, Sam3Config
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

class Sam3Wrapper(torch.nn.Module):
    def __init__(self, model=None, processor=None, inference_resolution=224):
        super().__init__()
        self.model = model

        
        self.processor = processor
        self.inference_resolution = inference_resolution

        if self.model is None:
            config = Sam3Config.from_pretrained("facebook/sam3")
            config.vision_config.backbone_config.image_size = self.inference_resolution
            self.model = Sam3Model.from_pretrained("facebook/sam3")
        if self.processor is None:
            self.processor = Sam3Processor.from_pretrained("facebook/sam3")
            self.processor.image_processor.size = {"height": self.inference_resolution, "width": self.inference_resolution}
        del self.model.vision_encoder.backbone # We use only the Nect, the Mask Decoder and text encoder here

    def register_text_embeddings(self, text):
        device = next(self.model.parameters()).device
    
        inputs_text = self.processor(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs_text)
        self.text_embeds = text_embeds
        self.attention_mask = inputs_text['attention_mask']
    
    def forward(self, vision_embeds):
        hidden_states = vision_embeds
        n_patches_x_row = int(hidden_states.shape[1] ** 0.5) 
        # Reshape for FPN neck: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, height, width]
        batch_size = hidden_states.shape[0]
        height = n_patches_x_row
        width = n_patches_x_row
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        fpn_hidden_states, fpn_position_encoding = self.model.vision_encoder.neck(hidden_states_spatial)

        vision_embeds = Sam3VisionEncoderOutput(
            last_hidden_state=hidden_states,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )

        bs = vision_embeds.last_hidden_state.shape[0]
        outputs = []
        for i in range(self.text_embeds.pooler_output.shape[0]):
            text_embed = CLIPTextModelOutput()
            text_embed.pooler_output = self.text_embeds.pooler_output[i].unsqueeze(0).expand(bs, -1, -1)

            inputs_text = {}

            inputs_text['text_embeds'] = text_embed
            inputs_text['attention_mask'] = self.attention_mask[i].unsqueeze(0).expand(bs, -1)
            inputs_text['vision_embeds'] = vision_embeds
            output = self.model(**inputs_text)
            outputs.append(output)
        return outputs

def post_process_instance_segmentation_nothresh(
    outputs,
    threshold: float = 0.3,
    target_sizes: list[tuple] | None = None,
):
    """
    Converts raw output into instance segmentation predictions returning soft masks and scores.

    Args:
        outputs ([`Sam3ImageSegmentationOutput`]):
            Raw outputs of the model containing pred_logits, pred_masks, and optionally presence_logits.
        threshold (`float`, *optional*, defaults to 0.3):
            Score threshold to keep instance predictions.
        target_sizes (`list[tuple[int, int]]`, *optional*):
            List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
            batch. If unset, predictions will not be resized.

    Returns:
        `list[dict]`: A list of dictionaries, each dictionary containing:
            - **scores** (`torch.Tensor`): Confidence scores for each predicted instance.
            - **masks** (`torch.Tensor`): Soft probability masks for each instance [0.0, 1.0], 
              shape (num_instances, height, width).
    """
    pred_logits = outputs.pred_logits  # (batch_size, num_queries)
    pred_masks = outputs.pred_masks  # (batch_size, num_queries, height, width)
    presence_logits = outputs.presence_logits  # (batch_size, 1) or None

    batch_size = pred_logits.shape[0]

    if target_sizes is not None and len(target_sizes) != batch_size:
        raise ValueError("Make sure that you pass in as many target sizes as images")

    # Compute scores: combine pred_logits with presence_logits if available
    batch_scores = pred_logits.sigmoid()
    if presence_logits is not None:
        presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
        batch_scores = batch_scores * presence_scores  # Broadcast multiplication

    # Apply sigmoid to convert mask logits to probabilities [0.0, 1.0]
    batch_masks = pred_masks.sigmoid()

    results = []
    for idx, (scores, masks) in enumerate(zip(batch_scores, batch_masks)):
        # Filter by score threshold
        keep = scores > threshold
        scores = scores[keep]
        masks = masks[keep]  # (num_keep, height, width)

        # Resize masks to target size if provided
        if target_sizes is not None:
            target_size = target_sizes[idx]
            if len(masks) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0),  # (1, num_keep, height, width)
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (num_keep, target_height, target_width)

        results.append({"scores": scores, "masks": masks})

    return results