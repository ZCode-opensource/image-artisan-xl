import torch

from transformers import CLIPImageProcessor
from diffusers.models import ImageProjection
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, AttnProcessor2_0
from torchvision import transforms
import numpy as np

from iartisanxl.graph.nodes.node import Node


class IPAdapterNode(Node):
    REQUIRED_INPUTS = ["unet", "ip_adapter_model", "image_encoder", "image"]
    OUTPUTS = ["image_embeds", "negative_image_embeds"]

    def __init__(self, type_index: int, adapter_type: str, adapter_scale: float = None, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = CLIPImageProcessor()
        self.type_index = type_index
        self.adapter_type = adapter_type
        self.adapter_scale = adapter_scale

    def update_adapter(self, type_index: int, adapter_type: str, enabled: bool, adapter_scale: float = None):
        self.type_index = type_index
        self.adapter_type = adapter_type
        self.enabled = enabled
        self.adapter_scale = adapter_scale
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["type_index"] = self.type_index
        node_dict["adapter_type"] = self.adapter_type
        node_dict["adapter_scale"] = self.adapter_scale
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(IPAdapterNode, cls).from_dict(node_dict)
        node.type_index = node_dict["type_index"]
        node.adapter_type = node_dict["adapter_type"]
        node.adapter_scale = node_dict["adapter_scale"]
        return node

    def update_inputs(self, node_dict):
        self.type_index = node_dict["type_index"]
        self.adapter_type = node_dict["adapter_type"]
        self.adapter_scale = node_dict["adapter_scale"]

    def __call__(self) -> dict:
        if self.enabled:
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
        else:
            self.unload()
            self.values["image_embeds"] = None
            self.values["negative_image_embeds"] = None

        return self.values

    def encode_image(self, images):
        transform = transforms.ToTensor()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        tensor_images = []
        neg_tensor_images = []
        weights = []

        for image_dict in images:
            image = image_dict.get("image")
            weight = image_dict.get("weight")

            # formula taken from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/IPAdapterPlus.py
            weight *= 0.1 + (weight - 0.1)
            weight = 1.19e-05 if weight <= 1.19e-05 else weight

            noise = image_dict.get("noise")

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

            if noise == 0:
                neg_tensor_image = torch.zeros_like(tensor_image)
            else:
                neg_tensor_image = self.image_add_noise(tensor_image, noise)

            neg_tensor_image = neg_tensor_image.to(device=self.device, dtype=self.torch_dtype)
            neg_tensor_images.append(neg_tensor_image)

        tensor_images = torch.cat(tensor_images, dim=0)
        neg_tensor_images = torch.cat(neg_tensor_images, dim=0)
        weights = torch.tensor(weights).to(self.device, dtype=self.torch_dtype).view(-1, 1, 1, 1)
        weighted_tensor_images = tensor_images * weights
        weighted_neg_tensor_images = neg_tensor_images * weights

        output_hidden_states = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True

        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(weighted_tensor_images, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(len(images), dim=0)

            uncond_image_enc_hidden_states = self.image_encoder(weighted_neg_tensor_images, output_hidden_states=True).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(len(images), dim=0)

            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(weighted_tensor_images).image_embeds
            image_embeds = image_embeds.repeat_interleave(len(images), dim=0)

            uncond_image_embeds = self.image_encoder(weighted_neg_tensor_images).image_embeds
            uncond_image_embeds = uncond_image_embeds.repeat_interleave(len(images), dim=0)

            return image_embeds, uncond_image_embeds

    def unload(self):
        if self.unet is not None:
            self.unet.encoder_hid_proj = None

            processor = AttnProcessor2_0()
            self.unet.set_attn_processor(processor)

    # formula taken from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/IPAdapterPlus.py
    def image_add_noise(self, source_image, noise):
        image = source_image.clone()
        torch.manual_seed(0)
        transformations = transforms.Compose(
            [
                transforms.ElasticTransform(alpha=75.0, sigma=noise * 3.5),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomHorizontalFlip(p=1.0),
            ]
        )
        image = transformations(image.cpu())
        image = image + ((0.25 * (1 - noise) + 0.05) * torch.randn_like(image))
        return image
