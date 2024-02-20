import numpy as np
import torch
from diffusers.models.embeddings import ImageProjection, IPAdapterFullImageProjection, IPAdapterPlusImageProjection
from torchvision import transforms
from transformers import CLIPImageProcessor

from iartisanxl.graph.nodes.node import Node
from iartisanxl.utilities.image.noise import add_torch_noise, create_mandelbrot_tensor, create_noise_tensor


class IPAdapterNode(Node):
    REQUIRED_INPUTS = ["ip_adapter_model", "image_encoder", "image"]
    OPTIONAL_INPUTS = ["mask_alpha_image"]
    OUTPUTS = ["ip_adapter"]

    def __init__(self, type_index: int, adapter_type: str, adapter_scale: float = None, **kwargs):
        super().__init__(**kwargs)

        self.type_index = type_index
        self.adapter_type = adapter_type
        self.adapter_scale = adapter_scale
        self.ip_image_prompt_embeds = None

        self.clip_image_processor = CLIPImageProcessor()

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
        image_projection = self.convert_ip_adapter_image_proj_to_diffusers(self.ip_adapter_model["image_proj"])
        image_projection.to(device=self.device, dtype=self.torch_dtype)

        image = self.image
        if isinstance(image, dict):
            image = [image]

        output_hidden_states = True
        if self.type_index == 0:
            output_hidden_states = False

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(image, output_hidden_states)

        # save_embeds = torch.cat([uncond_image_prompt_embeds, image_prompt_embeds])
        # torch.save(save_embeds, "C:/Users/Ozzy/Desktop/iartisanxl_style_test.ipadpt")

        tensor_mask = None
        if self.mask_alpha_image is not None:
            np_image = np.array(self.mask_alpha_image).astype(np.float32) / 255.0
            image = torch.from_numpy(np_image.transpose(2, 0, 1))

            alpha_channel = image[3, :, :]
            tensor_mask = torch.where(alpha_channel < 0.5, 1, 0).float()
            tensor_mask = tensor_mask.unsqueeze(0)
            tensor_mask = 1 - tensor_mask

        self.values["ip_adapter"] = {
            "weights": self.ip_adapter_model,
            "image_prompt_embeds": image_prompt_embeds,
            "uncond_image_prompt_embeds": uncond_image_prompt_embeds,
            "scale": self.adapter_scale,
            "tensor_mask": tensor_mask,
            "image_projection": image_projection,
        }

        return self.values

    def get_image_embeds(self, images, output_hidden_states):
        image_prompt_embeds = None
        uncond_image_prompt_embeds = None

        for image in images:
            weight = image["weight"]
            weight *= 0.1 + (weight - 0.1)
            weight = 1.19e-05 if weight <= 1.19e-05 else weight

            pil_image = image["image"]
            tensor_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            tensor_image = tensor_image.to(self.device, dtype=self.torch_dtype)

            if output_hidden_states:
                image_embeds = self.image_encoder(
                    tensor_image, output_hidden_states=output_hidden_states
                ).hidden_states[-2]
            else:
                image_embeds = self.image_encoder(tensor_image).image_embeds

            image_embeds = image_embeds * weight
            image_prompt_embeds = (
                torch.cat((image_prompt_embeds, image_embeds), dim=0)
                if image_prompt_embeds is not None
                else image_embeds
            )

            if image["noise"] > 0:
                if image["noise_index"] == 0:
                    uncond_tensor_image = self.image_add_noise(tensor_image, image["noise"])
                elif image["noise_index"] == 1:
                    uncond_tensor_image = create_mandelbrot_tensor(image["noise"], 224, 224)
                elif image["noise_index"] == 2:
                    uncond_tensor_image = create_noise_tensor("perlin", image["noise"], 224, 224)
                elif image["noise_index"] == 3:
                    uncond_tensor_image = create_noise_tensor("simplex", image["noise"], 224, 224)
                elif image["noise_index"] == 4:
                    uncond_tensor_image = add_torch_noise(tensor_image, "uniform", image["noise"])
                else:
                    uncond_tensor_image = add_torch_noise(tensor_image, "gaussian", image["noise"])

                uncond_tensor_image = uncond_tensor_image.to(self.device, dtype=self.torch_dtype)

                if output_hidden_states:
                    uncond_image_embeds = self.image_encoder(
                        uncond_tensor_image, output_hidden_states=output_hidden_states
                    ).hidden_states[-2]
                else:
                    uncond_image_embeds = self.image_encoder(uncond_tensor_image).image_embeds
            else:
                if output_hidden_states:
                    uncond_image_embeds = self.image_encoder(
                        torch.zeros_like(tensor_image), output_hidden_states=output_hidden_states
                    ).hidden_states[-2]
                else:
                    uncond_image_embeds = torch.zeros_like(image_embeds)

            uncond_image_prompt_embeds = (
                torch.cat((uncond_image_prompt_embeds, uncond_image_embeds), dim=0)
                if uncond_image_prompt_embeds is not None
                else uncond_image_embeds
            )

        return image_prompt_embeds, uncond_image_prompt_embeds

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

    def convert_ip_adapter_image_proj_to_diffusers(self, state_dict):
        updated_state_dict = {}
        image_projection = None

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            image_projection = ImageProjection(
                cross_attention_dim=cross_attention_dim,
                image_embed_dim=clip_embeddings_dim,
                num_image_text_embeds=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            image_projection = IPAdapterFullImageProjection(
                cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

            image_projection = IPAdapterPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                heads=heads,
                num_queries=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")
                diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
                diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
                diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    updated_state_dict[diffusers_name] = value

        image_projection.load_state_dict(updated_state_dict)
        return image_projection
