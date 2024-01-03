import torch

from transformers import CLIPImageProcessor
from diffusers.models import ImageProjection
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, AttnProcessor
from torchvision import transforms
import numpy as np

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

        image = self.image

        if isinstance(image, dict):
            image = [image]

        image_embeds, negative_image_embeds = self.encode_image(image)

        self.values["image_embeds"] = image_embeds
        self.values["negative_image_embeds"] = negative_image_embeds

        return self.values

    def encode_image(self, images):
        transform = transforms.ToTensor()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        tensor_images = []
        weights = []

        for image_dict in images:
            image = image_dict.get("image")
            weight = image_dict.get("weight")

            tensor_image = transform(image)

            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                alpha = torch.from_numpy(np.array(image.convert("RGBA").split()[-1])).float()
                mask = (alpha != 0).float()

                # Apply the mask to the RGB channels of the tensor
                tensor_image[:3, :, :] *= mask.unsqueeze(0)

            tensor_image = tensor_image[:3, :, :]
            tensor_image = tensor_image.unsqueeze(0)
            tensor_image = normalize(tensor_image)

            tensor_image = tensor_image.to(device=self.device, dtype=self.torch_dtype)
            tensor_images.append(tensor_image)
            weights.append(weight)

        tensor_images = torch.cat(tensor_images, dim=0)
        weights = torch.tensor(weights).to(self.device, dtype=self.torch_dtype).view(-1, 1)

        output_hidden_states = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True

        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(tensor_images, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(len(images), dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(torch.zeros_like(tensor_images), output_hidden_states=True).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(len(images), dim=0)
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(tensor_images).image_embeds
            image_embeds = image_embeds * weights
            image_embeds = image_embeds.repeat_interleave(len(images), dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def unload(self):
        self.unet.encoder_hid_proj = None

        processor = AttnProcessor()
        self.unet.set_attn_processor(processor, _remove_lora=True)
