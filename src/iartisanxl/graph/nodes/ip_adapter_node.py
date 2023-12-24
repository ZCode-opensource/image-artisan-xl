import torch
from transformers import CLIPImageProcessor
from diffusers.models import ImageProjection
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0

from iartisanxl.graph.nodes.node import Node


class IPAdapterNode(Node):
    REQUIRED_INPUTS = ["unet", "ip_adapter_model", "image_encoder", "image"]
    OUTPUTS = ["image_embeds", "negative_image_embeds"]

    def __init__(self, adapter_scale: float = None, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = CLIPImageProcessor()
        self.adapter_scale = adapter_scale

    def update_adapter(self, adapter_scale: float, enabled: bool):
        self.adapter_scale = adapter_scale
        self.enabled = enabled
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["adapter_scale"] = self.adapter_scale
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(IPAdapterNode, cls).from_dict(node_dict)
        node.adapter_scale = node_dict["adapter_scale"]
        return node

    def update_inputs(self, node_dict):
        self.adapter_scale = node_dict["adapter_scale"]

    def __call__(self) -> dict:
        super().__call__()

        self.unet._load_ip_adapter_weights(self.ip_adapter_model)

        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = self.adapter_scale

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

    def unload(self):
        self.unet.encoder_hid_proj = None
        self.unet.set_default_attn_processor()
