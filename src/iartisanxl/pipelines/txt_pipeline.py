import inspect
import logging
import gc
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from iartisanxl.nodes.latents_node import LatentsNode
from iartisanxl.nodes.encode_prompts_node import EncodePromptsNode


class ImageArtisanTextPipeline(
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
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

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
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.model_cpu_offloaded = False
        self.sequential_cpu_offloaded = False

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
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
    ):
        # device = self._execution_device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.debug("Using device: %s", device)

        do_classifier_free_guidance = False
        if guidance_scale > 1:
            do_classifier_free_guidance = True

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

        latents, generator = latents_node()
        latents = latents * self.scheduler.init_noise_sigma

        status_update("Preparing extra steps kwargs...")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

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

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
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
