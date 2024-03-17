import os

import accelerate
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_open_clip_checkpoint,
    create_unet_diffusers_config,
)
from diffusers.utils.peft_utils import delete_adapter_layers, recurse_remove_peft_layers, set_adapter_layers
from omegaconf import OmegaConf
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file as safe_load
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

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

    def __init__(
        self,
        path: str = None,
        model_name: str = None,
        version: str = None,
        model_type: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.single_checkpoint = kwargs.get("single_checkpoint", False)
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type

    def update_model(self, path: str, model_name: str, version: str, model_type: str):
        self.clear_models()
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["model_name"] = self.model_name
        node_dict["version"] = self.version
        node_dict["model_type"] = self.model_type
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(StableDiffusionXLModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.model_name = node_dict["model_name"]
        node.version = node_dict["version"]
        node.model_type = node_dict["model_type"]
        return node

    def update_inputs(self, node_dict):
        self.clear_models()
        self.path = node_dict["path"]
        self.model_name = node_dict["model_name"]
        self.version = node_dict["version"]
        self.model_type = node_dict["model_type"]

    def __call__(self):
        device = "cpu" if self.sequential_offload or self.cpu_offload else self.device

        if not self.single_checkpoint and os.path.isdir(self.path):
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
                    self.values["text_encoder_1"] = accelerate.cpu_offload(self.values["text_encoder_1"], "cuda:0")

                self.values["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.path, "text_encoder_2"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                ).to(device)

                if self.sequential_offload:
                    self.values["text_encoder_2"] = accelerate.cpu_offload(self.values["text_encoder_2"], "cuda:0")

                self.values["tokenizer_1"] = CLIPTokenizer.from_pretrained(os.path.join(self.path, "tokenizer"))

                self.values["tokenizer_2"] = CLIPTokenizer.from_pretrained(os.path.join(self.path, "tokenizer_2"))

                self.values["unet"] = UNet2DConditionModel.from_pretrained(
                    os.path.join(self.path, "unet"),
                    use_safetensors=True,
                    variant="fp16",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                ).to(device)

                if self.sequential_offload:
                    self.values["unet"] = accelerate.cpu_offload(self.values["unet"], "cuda:0")
        else:
            if os.path.isfile(self.path) and self.path.endswith(".safetensors"):
                original_config = OmegaConf.load(os.path.join("./configs/", "sd_xl_base.yaml"))
                unet_config = create_unet_diffusers_config(original_config, image_size=1024)

                try:
                    checkpoint = safe_load(self.path, device="cpu")
                except FileNotFoundError as exc:
                    raise FileNotFoundError("Model file not found.") from exc

                converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)

                ctx = init_empty_weights
                with ctx():
                    self.values["unet"] = UNet2DConditionModel(**unet_config)

                self.values["tokenizer_1"] = CLIPTokenizer.from_pretrained(
                    "./configs/clip-vit-large-patch14",
                    local_files_only=True,
                )

                config = CLIPTextConfig.from_pretrained("./configs/clip-vit-large-patch14", local_files_only=True)
                ctx = init_empty_weights
                with ctx():
                    self.values["text_encoder_1"] = CLIPTextModel(config)

                self.values["text_encoder_1"] = convert_ldm_clip_checkpoint(
                    checkpoint,
                    local_files_only=True,
                    text_encoder=self.values["text_encoder_1"],
                )
                self.values["text_encoder_1"].to(device=device, dtype=self.torch_dtype)

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
                self.values["text_encoder_2"].to(device=device, dtype=self.torch_dtype)

                for param_name, param in converted_unet_checkpoint.items():
                    set_module_tensor_to_device(
                        self.values["unet"],
                        param_name,
                        device,
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

    def get_list_adapters(self) -> dict[str, list[str]]:
        set_adapters = {}

        if hasattr(self.values["text_encoder_1"], "peft_config"):
            set_adapters["text_encoder"] = list(self.values["text_encoder_1"].peft_config.keys())

        if hasattr(self.values["text_encoder_2"], "peft_config"):
            set_adapters["text_encoder_2"] = list(self.values["text_encoder_2"].peft_config.keys())

        if hasattr(self.values["unet"], "peft_config"):
            set_adapters["unet"] = list(self.values["unet"].peft_config.keys())

        return set_adapters

    def get_active_adapters(self) -> list[str]:
        active_adapters = []

        for module in self.values["unet"].modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapters
                break

        return active_adapters

    def disable_lora(self):
        self.values["unet"].disable_lora()
        set_adapter_layers(self.values["text_encoder_1"], enabled=False)
        set_adapter_layers(self.values["text_encoder_2"], enabled=False)

    def enable_lora(self):
        self.values["unet"].enable_lora()
        set_adapter_layers(self.values["text_encoder_1"], enabled=True)
        set_adapter_layers(self.values["text_encoder_2"], enabled=True)

    def delete_adapters(self, adapter_names: list[str]):
        if self.values["unet"] is not None:
            self.values["unet"].delete_adapters(adapter_names)

            for adapter_name in adapter_names:
                delete_adapter_layers(self.values["text_encoder_1"], adapter_name)
                delete_adapter_layers(self.values["text_encoder_2"], adapter_name)

    def unload_lora_weights(self):
        unet = self.values.get("unet")

        if unet is not None:
            recurse_remove_peft_layers(unet)
            if hasattr(unet, "peft_config"):
                del unet.peft_config

            recurse_remove_peft_layers(self.values["text_encoder_1"])
            if hasattr(self.values["text_encoder_1"], "peft_config"):
                del self.values["text_encoder_1"].peft_config
                # pylint: disable=protected-access
                self.values["text_encoder_1"]._hf_peft_config_loaded = None

            recurse_remove_peft_layers(self.values["text_encoder_2"])
            if hasattr(self.values["text_encoder_2"], "peft_config"):
                del self.values["text_encoder_2"].peft_config
                # pylint: disable=protected-access
                self.values["text_encoder_2"]._hf_peft_config_loaded = None
