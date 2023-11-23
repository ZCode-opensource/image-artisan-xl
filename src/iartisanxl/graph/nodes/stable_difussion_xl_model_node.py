import os

import accelerate
import torch

from transformers import (
    CLIPTextModel,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import UNet2DConditionModel

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
)
from omegaconf import OmegaConf
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load

from iartisanxl.graph.nodes.node import Node


class StableDiffusionXLModelNode(Node):
    OUTPUTS = [
        "text_encoder_1",
        "text_encoder_2",
        "tokenizer_1",
        "tokenizer_2",
        "unet",
        "num_channels_latents",
    ]

    def __init__(self, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.single_checkpoint = kwargs.get("single_checkpoint", False)
        self.path = path

    def update_path(self, path: str):
        self.clear_models()
        self.path = path
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(StableDiffusionXLModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        return node

    def update_inputs(self, node_dict):
        self.clear_models()
        self.path = node_dict["path"]

    def __call__(self):
        super().__call__()

        if not self.single_checkpoint and os.path.isdir(self.path):
            device = (
                "cpu" if self.sequential_offload or self.cpu_offload else self.device
            )

            # simple check to ensure that at least could be a diffusers model
            if os.path.isfile(os.path.join(self.path, "model_index.json")):
                # we need to load the text encoders, the tokenizers and the unet
                self.values["text_encoder_1"] = CLIPTextModel.from_pretrained(
                    os.path.join(self.path, "text_encoder"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                ).to(device)

                if self.sequential_offload:
                    text_encoder_1 = accelerate.cpu_offload(text_encoder_1, "cuda:0")

                self.values[
                    "text_encoder_2"
                ] = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.path, "text_encoder_2"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                ).to(
                    device
                )

                if self.sequential_offload:
                    text_encoder_2 = accelerate.cpu_offload(text_encoder_2, "cuda:0")

                self.values["tokenizer_1"] = CLIPTokenizer.from_pretrained(
                    os.path.join(self.path, "tokenizer")
                )

                self.values["tokenizer_2"] = CLIPTokenizer.from_pretrained(
                    os.path.join(self.path, "tokenizer_2")
                )

                self.values["unet"] = UNet2DConditionModel.from_pretrained(
                    os.path.join(self.path, "unet"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                ).to(device)

                if self.sequential_offload:
                    self.values["unet"] = accelerate.cpu_offload(
                        self.values["unet"], "cuda:0"
                    )
        else:
            if os.path.isfile(self.path) and self.path.endswith(".safetensors"):
                original_config = OmegaConf.load(
                    os.path.join("./configs/", "sd_xl_base.yaml")
                )
                unet_config = create_unet_diffusers_config(
                    original_config, image_size=1024
                )

                try:
                    checkpoint = safe_load(self.path, device="cpu")
                except FileNotFoundError as exc:
                    raise FileNotFoundError("Model file not found.") from exc

                converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                    checkpoint, unet_config
                )

                ctx = init_empty_weights
                with ctx():
                    self.values["unet"] = UNet2DConditionModel(**unet_config)

                self.values["tokenizer_1"] = CLIPTokenizer.from_pretrained(
                    "./configs/clip-vit-large-patch14",
                    local_files_only=True,
                )

                config = CLIPTextConfig.from_pretrained(
                    "./configs/clip-vit-large-patch14", local_files_only=True
                )
                ctx = init_empty_weights
                with ctx():
                    self.values["text_encoder_1"] = CLIPTextModel(config)

                self.values["text_encoder_1"] = convert_ldm_clip_checkpoint(
                    checkpoint,
                    local_files_only=True,
                    text_encoder=self.values["text_encoder_1"],
                )
                self.values["text_encoder_1"].to(
                    device=self.device, dtype=self.torch_dtype
                )

                self.values["tokenizer_2"] = CLIPTokenizer.from_pretrained(
                    "./configs/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                    pad_token="!",
                    local_files_only=True,
                )

                config_name = "./configs/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                config_kwargs = {"projection_dim": 1280}
                self.values["text_encoder_2"] = convert_open_clip_checkpoint(
                    checkpoint,
                    config_name,
                    prefix="conditioner.embedders.1.model.",
                    has_projection=True,
                    **config_kwargs,
                )
                self.values["text_encoder_2"].to(
                    device=self.device, dtype=self.torch_dtype
                )

                for param_name, param in converted_unet_checkpoint.items():
                    set_module_tensor_to_device(
                        self.values["unet"],
                        param_name,
                        self.device,
                        value=param,
                        dtype=self.torch_dtype,
                    )

        self.values["num_channels_latents"] = self.values["unet"].config.in_channels

        return self.values

    def delete(self):
        self.clear_models()
        super().delete()

    def clear_models(self):
        self.values["unet"] = None
        self.values["text_encoder_1"] = None
        self.values["text_encoder_2"] = None
        self.values["tokenizer_1"] = None
        self.values["tokenizer_2"] = None
        self.values["num_channels_latents"] = None
        torch.cuda.empty_cache()
