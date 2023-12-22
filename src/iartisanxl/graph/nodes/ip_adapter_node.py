import torch
from transformers import CLIPImageProcessor
from diffusers.models import ImageProjection

from iartisanxl.graph.nodes.node import Node


class IPAdapterNode(Node):
    REQUIRED_INPUTS = ["unet", "image_encoder", "image"]
    OUTPUTS = ["image_embeds", "negative_image_embeds"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = CLIPImageProcessor()

    def __call__(self) -> dict:
        super().__call__()

        image_embeds, negative_image_embeds = self.encode_image(self.image)

        self.values["image_embeds"] = image_embeds
        self.values["negative_image_embeds"] = negative_image_embeds

        return self.values

    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=self.torch_dtype)

        output_hidden_states = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True

        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(1, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(torch.zeros_like(image), output_hidden_states=True).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(1, dim=0)
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(1, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds
