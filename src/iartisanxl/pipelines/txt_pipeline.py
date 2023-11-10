import inspect
import json
import logging
import gc
from typing import Any, Callable, Dict, Optional, Tuple

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
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, EulerDiscreteScheduler
from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from iartisanxl.convert_model.convert_functions import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)
from iartisanxl.pipelines.prompt_utils import (
    get_prompts_tokens_with_weights,
    pad_and_group_tokens_and_weights,
)


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

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None:
            self._lora_scale = lora_scale
            scale_lora_layers(self.text_encoder, lora_scale)
            scale_lora_layers(self.text_encoder_2, lora_scale)

        # Get tokens with first tokenizer
        (
            prompt_tokens,
            prompt_weights,
            neg_prompt_tokens,
            neg_prompt_weights,
        ) = get_prompts_tokens_with_weights(self.tokenizer, prompt, negative_prompt)

        # Check prompt2 and negative_prompt_2
        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt

        # Get tokens with second tokenizer
        (
            prompt_tokens_2,
            prompt_weights_2,
            neg_prompt_tokens_2,
            neg_prompt_weights_2,
        ) = get_prompts_tokens_with_weights(
            self.tokenizer_2, prompt_2, negative_prompt_2
        )

        # pylint: disable=unbalanced-tuple-unpacking
        (
            (prompt_tokens, prompt_weights),
            (neg_prompt_tokens, neg_prompt_weights),
            (prompt_tokens_2, prompt_weights_2),
            (neg_prompt_tokens_2, neg_prompt_weights_2),
        ) = pad_and_group_tokens_and_weights(
            (prompt_tokens, prompt_weights),
            (neg_prompt_tokens, neg_prompt_weights),
            (prompt_tokens_2, prompt_weights_2),
            (neg_prompt_tokens_2, neg_prompt_weights_2),
        )

        token_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_weights, dtype=torch.float16, device=device)

        token_tensor_2 = torch.tensor(
            [prompt_tokens_2], dtype=torch.long, device=device
        )

        embeds = []
        neg_embeds = []

        # use first text encoder
        prompt_embeds_1 = self.text_encoder(token_tensor, output_hidden_states=True)

        if clip_skip is None:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
        else:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[
                -(clip_skip + 2)
            ]

        # use second text encoder
        prompt_embeds_2 = self.text_encoder_2(token_tensor_2, output_hidden_states=True)
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [
            prompt_embeds_1_hidden_states,
            prompt_embeds_2_hidden_states,
        ]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

        for j, weight in enumerate(weight_tensor):
            if weight != 1.0:
                token_embedding[j] = (
                    token_embedding[-1]
                    + (token_embedding[j] - token_embedding[-1]) * weight
                )

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)

        neg_token_tensor = torch.tensor(
            [neg_prompt_tokens], dtype=torch.long, device=device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_tokens_2], dtype=torch.long, device=device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weights, dtype=torch.float16, device=device
        )

        # use first text encoder
        neg_prompt_embeds_1 = self.text_encoder(
            neg_token_tensor.to(device), output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        neg_prompt_embeds_2 = self.text_encoder_2(
            neg_token_tensor_2.to(device), output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [
            neg_prompt_embeds_1_hidden_states,
            neg_prompt_embeds_2_hidden_states,
        ]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

        for z, weight in enumerate(neg_weight_tensor):
            if weight != 1.0:
                neg_token_embedding[z] = (
                    neg_token_embedding[-1]
                    + (neg_token_embedding[z] - neg_token_embedding[-1]) * weight
                )

        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)

        prompt_embeds = torch.cat(embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        unscale_lora_layers(self.text_encoder, lora_scale)
        unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    @classmethod
    def from_single_file(
        cls, pretrained_model_link_or_path, vae: AutoencoderKL = None, **kwargs
    ):
        original_config_file = kwargs.pop("original_config_file", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        pipeline_class = ImageArtisanTextPipeline

        try:
            checkpoint = safe_load(pretrained_model_link_or_path, device="cpu")
        except FileNotFoundError as exc:
            raise FileNotFoundError("Model file not found.") from exc

        original_config = OmegaConf.load(original_config_file)

        image_size = 1024

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
            scheduler=scheduler,
        )

        pipe.to(dtype=torch_dtype)
        return pipe

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

        status_update("Encoding the prompt...")
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )

        if self.abort:
            on_aborted_function()
            return
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        if self.abort:
            on_aborted_function()
            return

        status_update("Preparing timesteps...")
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        status_update("Setting up generator...")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        status_update("Generating latents...")
        num_channels_latents = self.unet.config.in_channels

        shape = (
            1,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=prompt_embeds.dtype
        )
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
