import inspect
import gc
from typing import Optional, Union, List

import torch
import numpy as np
from PIL import Image

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import PIL_INTERPOLATION

from iartisanxl.graph.nodes.node import Node
from iartisanxl.graph.additional.controlnets_wrapper import ControlnetsWrapper
from iartisanxl.graph.additional.t2i_adapters_wrapper import T2IAdaptersWrapper


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
        "vae_scale_factor",
    ]
    OPTIONAL_INPUTS = [
        "original_size",
        "target_size",
        "negative_original_size",
        "negative_target_size",
        "crops_coords_top_left",
        "negative_crops_coords_top_left",
        "lora",
        "cross_attention_kwargs",
        "controlnet",
        "t2i_adapter",
        "image_embeds",
        "negative_image_embeds",
    ]
    OUTPUTS = ["latents"]

    def __init__(self, callback: callable = None, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.control_image_processor = None

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["callback"] = self.callback.__name__ if self.callback else None
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, callbacks=None):
        node = super(ImageGenerationNode, cls).from_dict(node_dict)
        node.callback = callbacks.get(node_dict["callback"]) if callbacks else None
        return node

    @torch.inference_mode()
    def __call__(self):
        super().__call__()

        crops_coords_top_left = self.crops_coords_top_left if self.crops_coords_top_left is not None else (0, 0)
        negative_crops_coords_top_left = self.negative_crops_coords_top_left if self.negative_crops_coords_top_left is not None else (0, 0)

        if hasattr(self.unet, "peft_config"):
            if len(self.unet.peft_config) == 0:
                del self.unet.peft_config

        if self.lora:
            if isinstance(self.lora, list):
                unzipped_list = zip(*self.lora)
                reordered_list = [list(item) for item in unzipped_list]
                self.unet.set_adapters(*reordered_list)
            else:
                self.unet.set_adapters([self.lora[0]], [self.lora[1]])

        controlnets_models = None
        controlnet_images = None
        controlnet_conditioning_scale = None
        guess_mode = False

        if self.controlnet:
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor,
                do_convert_rgb=True,
                do_normalize=False,
            )

            if isinstance(self.controlnet, dict):
                controlnets = [self.controlnet]
            else:
                controlnets = self.controlnet

            models = [net["model"] for net in controlnets]
            controlnet_images = [net["image"] for net in controlnets]
            controlnet_conditioning_scale = [net["conditioning_scale"] for net in controlnets]
            controlnets_models = ControlnetsWrapper(models)

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, self.num_inference_steps, self.device)

        latents = self.latents * self.scheduler.init_noise_sigma

        add_text_embeds = self.pooled_prompt_embeds

        height = self.height
        width = self.width
        original_size = self.original_size if self.original_size is not None else (height, width)
        target_size = self.target_size if self.target_size is not None else (height, width)

        t2i_adapter_models = None
        t2i_adapter_images = []
        t2i_conditioning_factor = None
        t2i_conditioning_scale = None

        if self.t2i_adapter:
            if isinstance(self.t2i_adapter, dict):
                t2i_adapters = [self.t2i_adapter]
            else:
                t2i_adapters = self.t2i_adapter

            models = [net["model"] for net in t2i_adapters]
            t2i_conditioning_scale = [net["conditioning_scale"] for net in t2i_adapters]
            t2i_conditioning_factor = [net["conditioning_factor"] for net in t2i_adapters]
            t2i_adapter_models = T2IAdaptersWrapper(models)

            for adapter_image in [net["image"] for net in t2i_adapters]:
                adapter_image = _preprocess_adapter_image(adapter_image, height, width)
                adapter_image = adapter_image.to(device=self.device, dtype=self.torch_dtype)
                t2i_adapter_images.append(adapter_image)

        prompt_embeds = self.prompt_embeds

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        if self.negative_original_size is not None and self.negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        image_embeds = self.image_embeds

        do_classifier_free_guidance = False
        if self.guidance_scale > 1:
            do_classifier_free_guidance = True

            prompt_embeds = torch.cat([self.negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([self.negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            if image_embeds is not None:
                image_embeds = torch.cat([self.negative_image_embeds, self.image_embeds])
                image_embeds = image_embeds.to(self.device)

        if t2i_adapter_models is not None:
            adapter_states = t2i_adapter_models(t2i_adapter_images, t2i_conditioning_scale)

            if do_classifier_free_guidance:
                for k, state in enumerate(adapter_states):
                    adapter_states[k] = [torch.cat([v] * 2, dim=0) for v in state]

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(1)
            timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim).to(
                device=self.device, dtype=latents.dtype
            )

        num_warmup_steps = max(len(timesteps) - self.num_inference_steps * self.scheduler.order, 0)

        if controlnets_models is not None:
            control_images = []

            for image_ in controlnet_images:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=self.device,
                    dtype=self.torch_dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(image_)

            height, width = control_images[0].shape[-2:]

            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < net_dict["guidance_start"] or (i + 1) / len(timesteps) > net_dict["guidance_end"])
                    for net_dict in controlnets
                ]
                controlnet_keep.append(keeps)

        # scheduler generator
        scheduler_kwargs = {}
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            scheduler_kwargs["generator"] = self.generator

        if self.cpu_offload:
            self.unet.to("cuda:0")

        down_block_res_samples = None
        mid_block_res_sample = None
        down_intrablock_additional_residuals = None

        for i, t in enumerate(timesteps):
            # expand the latents if doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            if image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds

            if self.abort:
                return

            if controlnets_models is not None:
                if guess_mode and do_classifier_free_guidance:
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                cond_scale = [c * k for c, k in zip(controlnet_conditioning_scale, controlnet_keep[i])]

                down_block_res_samples, mid_block_res_sample = controlnets_models(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_images,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            if t2i_adapter_models is not None:
                down_intrablock_additional_residuals = []
                for _j, (states, factor) in enumerate(zip(adapter_states, t2i_conditioning_factor)):
                    if i < int(num_inference_steps * factor):
                        cloned_states = [state.clone() for state in states]
                        down_intrablock_additional_residuals.extend(cloned_states)
                    else:
                        zero_states = [torch.zeros_like(state) for state in states]
                        down_intrablock_additional_residuals.extend(zero_states)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.abort:
                return

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **scheduler_kwargs, return_dict=False)[0]

            if self.abort:
                return

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if self.callback is not None:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    self.callback(step_idx, t, latents)

            if self.abort:
                return

        if self.cpu_offload:
            self.unet.to("cpu")

        self.values["latents"] = latents

        del latents, noise_pred
        gc.collect()

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
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
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


def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}")
    return image


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
