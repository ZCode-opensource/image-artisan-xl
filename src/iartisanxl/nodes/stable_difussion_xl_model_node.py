# pylint: disable=no-member
import os

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

from iartisanxl.nodes.node import Node


class StableDiffusionXLModelNode(Node):
    REQUIRED_ARGS = [
        "path",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.float16)
        self.single_checkpoint = kwargs.get("single_checkpoint", False)

    def __call__(
        self,
    ) -> tuple[
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPTokenizer,
        UNet2DConditionModel,
    ]:
        text_encoder_1 = None
        text_encoder_2 = None
        tokenizer_1 = None
        tokenizer_2 = None
        unet = None

        if not self.single_checkpoint and os.path.isdir(self.path):
            # simple check to ensure that at least could be a diffusers model
            if os.path.isfile(os.path.join(self.path, "model_index.json")):
                # we need to load the text encoders, the tokenizers and the unet
                text_encoder_1 = CLIPTextModel.from_pretrained(
                    os.path.join(self.path, "text_encoder"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )

                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.path, "text_encoder_2"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )

                tokenizer_1 = CLIPTokenizer.from_pretrained(
                    os.path.join(self.path, "tokenizer")
                )

                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    os.path.join(self.path, "tokenizer_2")
                )

                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(self.path, "unet"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
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
                    unet = UNet2DConditionModel(**unet_config)

                tokenizer_1 = CLIPTokenizer.from_pretrained(
                    "./configs/clip-vit-large-patch14",
                    local_files_only=True,
                )
                config = CLIPTextConfig.from_pretrained(
                    "./configs/clip-vit-large-patch14", local_files_only=True
                )
                ctx = init_empty_weights
                with ctx():
                    text_encoder_1 = CLIPTextModel(config)
                text_encoder_1 = convert_ldm_clip_checkpoint(
                    checkpoint, local_files_only=True, text_encoder=text_encoder_1
                )
                text_encoder_1.to(dtype=self.torch_dtype)
                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "./configs/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                    pad_token="!",
                    local_files_only=True,
                )

                config_name = "./configs/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                config_kwargs = {"projection_dim": 1280}
                text_encoder_2 = convert_open_clip_checkpoint(
                    checkpoint,
                    config_name,
                    prefix="conditioner.embedders.1.model.",
                    has_projection=True,
                    **config_kwargs,
                )
                text_encoder_2.to(dtype=self.torch_dtype)

                for param_name, param in converted_unet_checkpoint.items():
                    set_module_tensor_to_device(
                        unet, param_name, "cpu", value=param, dtype=self.torch_dtype
                    )

        return text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet
