import os
import sys
import inspect
import logging
import importlib
from typing import Optional, Union

import torch
from omegaconf import OmegaConf
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load
from transformers import (
    CLIPTextModel,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, EulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


class ImageArtisanConvertPipeline(
    DiffusionPipeline,
):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )

        self.logger = logging.getLogger()

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, vae: AutoencoderKL = None, **kwargs):
        original_config_file = kwargs.pop("original_config_file", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        pipeline_class = ImageArtisanConvertPipeline

        try:
            checkpoint = safe_load(pretrained_model_link_or_path, device="cpu")
        except FileNotFoundError as exc:
            raise FileNotFoundError("Model file not found.") from exc

        original_config = OmegaConf.load(original_config_file)

        image_size = 1024

        scheduler_config_dict = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "interpolation_type": "linear",
            "skip_prk_steps": True,
        }

        scheduler = EulerDiscreteScheduler.from_config(scheduler_config_dict)
        scheduler.register_to_config(clip_sample=False)

        unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
        path = pretrained_model_link_or_path
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, path=path, extract_ema=False)

        ctx = init_empty_weights
        with ctx():
            unet = UNet2DConditionModel(**unet_config)

        if vae is None:
            vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
            vae_scaling_factor = original_config.model.params.scale_factor
            vae_config["scaling_factor"] = vae_scaling_factor
            ctx = init_empty_weights
            with ctx():
                vae = AutoencoderKL(**vae_config)

            for param_name, param in converted_vae_checkpoint.items():
                set_module_tensor_to_device(vae, param_name, "cpu", value=param, dtype=torch.float16)

        tokenizer = CLIPTokenizer.from_pretrained(
            "./configs/clip-vit-large-patch14",
            local_files_only=True,
        )
        config = CLIPTextConfig.from_pretrained("./configs/clip-vit-large-patch14", local_files_only=True)
        ctx = init_empty_weights
        with ctx():
            text_encoder = CLIPTextModel(config)
        text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=True, text_encoder=text_encoder)

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

        for param_name, param in converted_unet_checkpoint.items():
            set_module_tensor_to_device(unet, param_name, "cpu", value=param, dtype=torch.float16)

        pipe = pipeline_class(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )

        pipe.to(dtype=torch_dtype)
        return pipe

    def save_to_diffusers(
        self,
        save_directory: Union[str, os.PathLike],
        status_update: callable,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        steps = 1
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        expected_modules, _optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            status_update(f"Converting {pipeline_component_name}...", steps)

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = sub_model._orig_mod  # pylint: disable=protected-access
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)  # pylint: disable=redefined-outer-name
                else:
                    self.logger.error(
                        "%s is not installed. Cannot save %s as %s from %s",
                        library_name,
                        pipeline_component_name,
                        library_classes,
                        library_name,
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                self.logger.error(
                    "self.%s=%s of type %s cannot be saved.",
                    pipeline_component_name,
                    sub_model,
                    type(sub_model),
                )
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

            steps = steps + 1
            status_update(f"{pipeline_component_name} saved...", steps)

        # finally save the config
        self.save_config(save_directory)
