import logging
import gc
from typing import Any, Callable, Dict, Optional, Tuple

import torch
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
from iartisanxl.nodes.image_generation_node import ImageGenerationNode
from iartisanxl.nodes.latents_decoder_node import LatentsDecoderNode


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

        status_update("Generating embeddings..")
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

        if self.abort:
            on_aborted_function()
            return

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encode_prompts_node()

        if self.abort:
            on_aborted_function()
            return

        unscale_lora_layers(self.text_encoder, text_encoder_lora_scale)
        unscale_lora_layers(self.text_encoder_2, text_encoder_lora_scale)

        if self.abort:
            on_aborted_function()
            return

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

        if self.abort:
            on_aborted_function()
            return

        latents, generator = latents_node()

        if self.abort:
            on_aborted_function()
            return

        status_update("Generating image...")
        image_generation_node = ImageGenerationNode(
            unet=self.unet, device=device, abort=lambda: self.abort
        )

        if self.abort:
            on_aborted_function()
            return

        latents = image_generation_node(
            width=width,
            height=height,
            scheduler=self.scheduler,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            callback=callback,
        )

        if self.abort:
            on_aborted_function()
            return

        status_update("Decoding latents...")
        latents_decoder_node = LatentsDecoderNode(vae=self.vae, device=device)
        image = latents_decoder_node(latents)

        gc.collect()
        # Offload all models
        self.maybe_free_model_hooks()

        return image
