import inspect

import torch

from iartisanxl.nodes.node import Node


class ImageGenerationNode(Node):
    REQUIRED_INPUTS = [
        "unet",
        "width",
        "height",
        "scheduler",
        "latents",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "generator",
        "guidance_scale",
        "num_inference_steps",
    ]
    OPTIONAL_INPUTS = [
        "original_size",
        "target_size",
        "negative_original_size",
        "negative_target_size",
        "crops_coords_top_left",
        "negative_crops_coords_top_left",
        "active_loras",
        "cross_attention_kwargs",
    ]
    OUTPUTS = ["latents"]

    def __init__(self, abort: callable = None, callback: callable = None, **kwargs):
        super().__init__(**kwargs)
        self.abort = abort if abort is not None else lambda: False
        self.callback = callback

    def __call__(self):
        crops_coords_top_left = (
            self.crops_coords_top_left
            if self.crops_coords_top_left is not None
            else (0, 0)
        )

        negative_crops_coords_top_left = (
            self.negative_crops_coords_top_left
            if self.negative_crops_coords_top_left is not None
            else (0, 0)
        )

        if self.active_loras:
            self.unet.set_adapters(*self.active_loras)

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = self.latents * self.scheduler.init_noise_sigma

        add_text_embeds = self.pooled_prompt_embeds

        original_size = (
            self.original_size
            if self.original_size is not None
            else (self.height, self.width)
        )

        target_size = (
            self.target_size
            if self.target_size is not None
            else (self.height, self.width)
        )

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.prompt_embeds.dtype)

        if (
            self.negative_original_size is not None
            and self.negative_target_size is not None
        ):
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=self.prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        do_classifier_free_guidance = False
        if self.guidance_scale > 1:
            do_classifier_free_guidance = True

            prompt_embeds = torch.cat(
                [self.negative_prompt_embeds, self.prompt_embeds], dim=0
            )
            add_text_embeds = torch.cat(
                [self.negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(1)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)

        num_warmup_steps = max(
            len(timesteps) - self.num_inference_steps * self.scheduler.order, 0
        )

        # scheduler generator
        scheduler_kwargs = {}
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            scheduler_kwargs["generator"] = self.generator

        for i, t in enumerate(timesteps):
            # expand the latents if doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

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
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.abort():
                return latents

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **scheduler_kwargs, return_dict=False
            )[0]

            if self.abort():
                return latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                if self.callback is not None:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    self.callback(step_idx, t, latents)

            if self.abort():
                return latents

        self.values["latents"] = latents

        return self.values

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
