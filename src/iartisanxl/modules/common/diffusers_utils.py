import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load
from omegaconf import OmegaConf

from iartisanxl.convert_model.convert_functions import (
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)

diffusers_models = [
    {
        "name": "text_encoder",
        "loading_info": "first text encoder",
        "model": CLIPTextModel,
        "args": {
            "subfolder": "text_encoder",
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_safetensors": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        },
    },
    {
        "name": "text_encoder_2",
        "loading_info": "second text encoder",
        "model": CLIPTextModelWithProjection,
        "args": {
            "subfolder": "text_encoder_2",
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_safetensors": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        },
    },
    {
        "name": "tokenizer",
        "loading_info": "first tokenizer",
        "model": CLIPTokenizer,
        "args": {
            "subfolder": "tokenizer",
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        },
    },
    {
        "name": "tokenizer_2",
        "loading_info": "first tokenizer",
        "model": CLIPTokenizer,
        "args": {
            "subfolder": "tokenizer_2",
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        },
    },
    {
        "name": "unet",
        "loading_info": "first tokenizer",
        "model": UNet2DConditionModel,
        "args": {
            "subfolder": "unet",
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_safetensors": True,
            "local_files_only": True,
        },
    },
    {
        "name": "vae",
        "loading_info": "vae",
        "model": AutoencoderKL,
        "args": {
            "subfolder": "vae",
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_safetensors": True,
        },
    },
]


def load_vae_from_safetensors(safetensors_path, original_config_file):
    checkpoint = safe_load(safetensors_path, device="cpu")
    original_config = OmegaConf.load(original_config_file)

    vae_config = create_vae_diffusers_config(original_config, image_size=1024)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
    vae_scaling_factor = original_config.model.params.scale_factor
    vae_config["scaling_factor"] = vae_scaling_factor
    ctx = init_empty_weights
    with ctx():
        vae = AutoencoderKL(**vae_config)

    for param_name, param in converted_vae_checkpoint.items():
        set_module_tensor_to_device(vae, param_name, "cpu", value=param)

    return vae
