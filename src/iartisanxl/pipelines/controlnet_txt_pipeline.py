import inspect
import json
import logging
import gc
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers, EulerDiscreteScheduler
from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from iartisanxl.convert_model.convert_functions import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)
from iartisanxl.nodes.latents_node import LatentsNode
from iartisanxl.nodes.encode_prompts_node import EncodePromptsNode


class ImageArtisanControlNetTextPipeline(
    DiffusionPipeline,
    StableDiffusionXLLoraLoaderMixin,
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            Tuple[ControlNetModel],
            MultiControlNetModel,
        ],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.logger = logging.getLogger()
        self.abort = False
        self._lora_scale = 1.0

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

        self.model_cpu_offloaded = False
        self.sequential_cpu_offloaded = False

    @classmethod
    def from_single_file(
        cls, pretrained_model_link_or_path, vae: AutoencoderKL = None, **kwargs
    ):
        original_config_file = kwargs.pop("original_config_file", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        scheduler = kwargs.pop("scheduler", None)
        pipeline_class = ImageArtisanControlNetTextPipeline

        try:
            checkpoint = safe_load(pretrained_model_link_or_path, device="cpu")
        except FileNotFoundError as exc:
            raise FileNotFoundError("Model file not found.") from exc

        original_config = OmegaConf.load(original_config_file)

        image_size = 1024

        if scheduler is None:
            scheduler_config = None
            with open(
                "./configs/scheduler_config.json", "r", encoding="utf-8"
            ) as config_file:
                scheduler_config = json.load(config_file)

            scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
            scheduler.register_to_config(clip_sample=False)

        unet_config = create_unet_diffusers_config(
            original_config, image_size=image_size
        )
        path = pretrained_model_link_or_path
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=path, extract_ema=False
        )

        ctx = init_empty_weights
        with ctx():
            unet = UNet2DConditionModel(**unet_config)

        if vae is None:
            vae_config = create_vae_diffusers_config(
                original_config, image_size=image_size
            )
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                checkpoint, vae_config
            )
            vae_scaling_factor = original_config.model.params.scale_factor
            vae_config["scaling_factor"] = vae_scaling_factor
            ctx = init_empty_weights
            with ctx():
                vae = AutoencoderKL(**vae_config)

            for param_name, param in converted_vae_checkpoint.items():
                set_module_tensor_to_device(
                    vae, param_name, "cpu", value=param, dtype=torch.float16
                )

        tokenizer = CLIPTokenizer.from_pretrained(
            "./configs/clip-vit-large-patch14",
            local_files_only=True,
        )
        text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=True)
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
            set_module_tensor_to_device(
                unet, param_name, "cpu", value=param, dtype=torch.float16
            )

        pipe = pipeline_class(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=None,
            scheduler=scheduler,
        )

        pipe.to(dtype=torch_dtype)
        return pipe

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(
            image, height=height, width=width
        ).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))  # pylint: disable=not-callable
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        prompt: str,
        seed: int,
        height: int,
        width: int,
        on_aborted_function: Callable,
        status_update: Callable = None,
        prompt_2: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
    ):
        # device = self._execution_device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.debug("Using device: %s", device)

        # align format for control guidance
        mult = (
            len(self.controlnet.nets)
            if isinstance(self.controlnet, MultiControlNetModel)
            else 1
        )
        control_guidance_start, control_guidance_end = mult * [
            control_guidance_start
        ], mult * [control_guidance_end]

        do_classifier_free_guidance = False
        if guidance_scale > 1:
            do_classifier_free_guidance = True

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                self.controlnet.nets
            )

        status_update("Encoding the prompts...")
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )

        if text_encoder_lora_scale is not None:
            self._lora_scale = text_encoder_lora_scale
            scale_lora_layers(self.text_encoder, text_encoder_lora_scale)
            scale_lora_layers(self.text_encoder_2, text_encoder_lora_scale)

        if self.abort:
            on_aborted_function()
            return

        encode_prompts_node = EncodePromptsNode(
            prompt1=prompt,
            prompt2=prompt_2,
            negative_prompt1=negative_prompt,
            negative_prompt2=negative_prompt_2,
            tokenizer1=self.tokenizer,
            tokenizer2=self.tokenizer_2,
            text_encoder1=self.text_encoder,
            text_encoder2=self.text_encoder_2,
            clip_skip=clip_skip,
            device=device,
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encode_prompts_node.process()

        if self.abort:
            on_aborted_function()
            return

        unscale_lora_layers(self.text_encoder, text_encoder_lora_scale)
        unscale_lora_layers(self.text_encoder_2, text_encoder_lora_scale)

        if isinstance(self.controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=1 * 1,
                num_images_per_prompt=1,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(self.controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        status_update("Preparing timesteps...")
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        status_update("Generating latents...")
        latents_node = LatentsNode(
            width=width,
            height=height,
            seed=seed,
            num_channels_latents=self.unet.config.in_channels,
            scale_factor=self.vae_scale_factor,
            device=device,
            dtype=prompt_embeds.dtype,
        )

        latents, generator = latents_node.process()
        latents = latents * self.scheduler.init_noise_sigma

        status_update("Preparing extra steps kwargs...")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps
            )

        status_update("Preparing added time...")
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        status_update("Preparing emdeddings...")

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(1)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        status_update("Generating image...")
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        latents = latents.to("cuda")

        for i, t in enumerate(timesteps):
            if self.abort:
                on_aborted_function()
                return

            # expand the latents
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }

            if guess_mode and do_classifier_free_guidance:
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t
                )
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                controlnet_added_cond_kwargs = {
                    "text_embeds": add_text_embeds.chunk(2)[1],
                    "time_ids": add_time_ids.chunk(2)[1],
                }
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs

            if isinstance(controlnet_keep[i], list):
                cond_scale = [
                    c * s
                    for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
                ]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                added_cond_kwargs=controlnet_added_cond_kwargs,
                return_dict=False,
            )

            if guess_mode and do_classifier_free_guidance:
                down_block_res_samples = [
                    torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                ]
                mid_block_res_sample = torch.cat(
                    [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                )

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.abort:
                on_aborted_function()
                return

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        if self.abort:
            on_aborted_function()
            return

        needs_upcasting = self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)

        status_update("Decoding latents...")
        if self.vae.device != latents.device and str(self.vae.device) != "meta":
            self.vae.to(latents.device)

        decoded = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        if self.abort:
            on_aborted_function()
            return

        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image.cpu().permute(1, 2, 0).float().numpy()
        image_pil = Image.fromarray(np.uint8(image_np * 255))

        del image, image_np
        gc.collect()

        # Offload all models
        self.maybe_free_model_hooks()

        return image_pil
