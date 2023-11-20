# pylint: disable=no-member
import inspect
from typing import Optional, Callable

import torch

from iartisanxl.nodes.node import Node


class ImageGenerationNode(Node):
    REQUIRED_ARGS = [
        "unet",
        "device",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.float16)
        self.original_size = kwargs.get("original_size", None)
        self.target_size = kwargs.get("target_size", None)
        self.crops_coords_top_left = kwargs.get("crops_coords_top_left", (0, 0))
        self.negative_original_size = kwargs.get("negative_original_size", None)
        self.negative_target_size = kwargs.get("negative_target_size", None)
        self.negative_crops_coords_top_left = kwargs.get(
            "negative_crops_coords_top_left", (0, 0)
        )
        self.abort = kwargs.get("abort", lambda: False)

    def __call__(
        self,
        width: int,
        height: int,
        scheduler,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        generator,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        cross_attention_kwargs: Optional[dict[str, any]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    ):
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        latents = latents * scheduler.init_noise_sigma

        add_text_embeds = pooled_prompt_embeds

        if self.original_size is None:
            self.original_size = (height, width)

        if self.target_size is None:
            self.target_size = (height, width)

        add_time_ids = list(
            self.original_size + self.crops_coords_top_left + self.target_size
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        if (
            self.negative_original_size is not None
            and self.negative_target_size is not None
        ):
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                self.negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        do_classifier_free_guidance = False
        if guidance_scale > 1:
            do_classifier_free_guidance = True

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(1)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        # scheduler generator
        scheduler_kwargs = {}
        accepts_generator = "generator" in set(
            inspect.signature(scheduler.step).parameters.keys()
        )
        if accepts_generator:
            scheduler_kwargs["generator"] = generator

        for i, t in enumerate(timesteps):
            # expand the latents if doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }

            if self.abort():
                return latents

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.abort():
                return latents

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(
                noise_pred, t, latents, **scheduler_kwargs, return_dict=False
            )[0]

            if self.abort():
                return latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
            ):
                if callback is not None:
                    step_idx = i // getattr(scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if self.abort():
                return latents

        return latents

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
