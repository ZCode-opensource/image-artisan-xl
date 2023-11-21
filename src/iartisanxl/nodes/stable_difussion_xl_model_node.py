import os

import accelerate

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
    OUTPUTS = [
        "text_encoder_1",
        "text_encoder_2",
        "tokenizer_1",
        "tokenizer_2",
        "unet",
        "num_channels_latents",
    ]

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.single_checkpoint = kwargs.get("single_checkpoint", False)
        self.can_offload = True
        self.path = path

    def __call__(self):
        text_encoder_1 = None
        text_encoder_2 = None
        tokenizer_1 = None
        tokenizer_2 = None
        unet = None

        if not self.single_checkpoint and os.path.isdir(self.path):
            device = "cpu" if self.sequential_offload else self.device

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
                ).to(device)

                if self.sequential_offload:
                    text_encoder_1 = accelerate.cpu_offload(text_encoder_1, "cuda:0")

                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.path, "text_encoder_2"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                ).to(device)

                if self.sequential_offload:
                    text_encoder_2 = accelerate.cpu_offload(text_encoder_2, "cuda:0")

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
                ).to(device)

                if self.sequential_offload:
                    unet = accelerate.cpu_offload(unet, "cuda:0")
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

        num_channels_latents = unet.config.in_channels

        self.values["text_encoder_1"] = text_encoder_1
        self.values["text_encoder_2"] = text_encoder_2
        self.values["tokenizer_1"] = tokenizer_1
        self.values["tokenizer_2"] = tokenizer_2
        self.values["unet"] = unet
        self.values["num_channels_latents"] = num_channels_latents

        return self.values
