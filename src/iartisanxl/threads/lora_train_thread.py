import os
import itertools
import math

import torch
import torch.nn.functional as F

from typing import Dict
from PyQt6.QtCore import QThread, pyqtSignal
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.optimization import get_scheduler

from iartisanxl.train.local_image_dataset import LocalImageTextDataset
from iartisanxl.train.lora_train_args import LoraTrainArgs


# pylint: disable=no-member
class LoraTrainThread(QThread):
    output = pyqtSignal(str)
    output_done = pyqtSignal()
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    update_epoch = pyqtSignal(int, float)

    def __init__(self, lora_train_args: LoraTrainArgs):
        super().__init__()

        self.has_error = False
        self.error_text = ""
        self.lora_train_args = lora_train_args

        if len(self.lora_train_args.output_dir) == 0:
            self.has_error = True
            self.error_text = "No output directory selected"
            return

        if not os.path.isdir(self.lora_train_args.output_dir):
            os.makedirs(self.lora_train_args.output_dir, exist_ok=False)

        if len(self.lora_train_args.model_path) == 0:
            self.error_text = "No model selected"
            self.has_error = True
            return

        if len(self.lora_train_args.dataset_path) == 0:
            self.error_text = "No dataset selected"
            self.has_error = True
            return

        self.image_size = 1024
        self.weight_dtype = torch.bfloat16
        self.accelerator = None
        self.unet = None
        self.text_encoder_one = None
        self.text_encoder_two = None

    def run(self):
        if not self.has_error:
            self.output.emit(f"Output directory: {self.lora_train_args.output_dir}")
            self.output.emit(f"Model: {self.lora_train_args.model_path}")
            self.output.emit(f"Dataset: {self.lora_train_args.dataset_path}")

            self.output.emit("Setting up accelerator...")
            accelerator_project_config = ProjectConfiguration(
                project_dir=self.lora_train_args.output_dir
            )
            self.accelerator = Accelerator(
                gradient_accumulation_steps=4,
                mixed_precision="bf16",
                project_config=accelerator_project_config,
            )
            self.output_done.emit()

            # Load the tokenizers
            self.output.emit("Setting tokenizer one...")
            tokenizer_one = AutoTokenizer.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="tokenizer",
                use_fast=False,
            )
            self.output_done.emit()

            self.output.emit("Setting tokenizer two...")
            tokenizer_two = AutoTokenizer.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="tokenizer_2",
                use_fast=False,
            )
            self.output_done.emit()

            self.output.emit("Setting the noise scheduler...")
            noise_scheduler = DDPMScheduler.from_pretrained(
                self.lora_train_args.model_path, subfolder="scheduler"
            )
            self.output_done.emit()

            self.output.emit("Loading text encoders...")
            self.text_encoder_one = CLIPTextModel.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="text_encoder",
            )
            self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="text_encoder_2",
            )
            self.output_done.emit()

            self.output.emit("Loading vae...")
            vae = AutoencoderKL.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="vae",
            )
            self.output_done.emit()

            self.output.emit("Loading unet...")
            self.unet = UNet2DConditionModel.from_pretrained(
                self.lora_train_args.model_path,
                subfolder="unet",
            )
            self.output_done.emit()

            # We only train the additional adapter LoRA layers
            vae.requires_grad_(False)
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            self.unet.requires_grad_(False)

            self.warning.emit("Using bfloat16 to train LoRA (only NVIDIA)")

            self.output.emit("Moving unet to device...")
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            self.output_done.emit()

            self.output.emit("Moving vae to device...")
            vae.to(self.accelerator.device, dtype=torch.float32)
            self.output_done.emit()

            self.output.emit("Moving text encoders to device...")
            self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
            self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
            self.output_done.emit()

            self.output.emit("Adding new LoRA weights to the attention layers...")
            unet_lora_attn_procs = {}
            unet_lora_parameters = []

            for name, _attn_processor in self.unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet.config.cross_attention_dim
                )

                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else LoRAAttnProcessor
                )

                module = lora_attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_train_args.rank,
                )
                unet_lora_attn_procs[name] = module
                unet_lora_parameters.extend(module.parameters())

            self.unet.set_attn_processor(unet_lora_attn_procs)
            self.output_done.emit()

            self.output.emit(
                "Monkey-patching the forward calls of the text encoders attention-blocks..."
            )

            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            # pylint: disable=protected-access
            text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
                self.text_encoder_one,
                dtype=torch.float32,
                rank=self.lora_train_args.rank,
            )
            text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
                self.text_encoder_two,
                dtype=torch.float32,
                rank=self.lora_train_args.rank,
            )
            self.output_done.emit()

            self.accelerator.register_save_state_pre_hook(self.save_model_hook)
            self.accelerator.register_load_state_pre_hook(self.load_model_hook)

            self.warning.emit("Enabling TF32 for faster training on Ampere GPUs")
            torch.backends.cuda.matmul.allow_tf32 = True

            self.output.emit("Creating the optimizer...")
            optimizer_class = torch.optim.AdamW
            params_to_optimize = itertools.chain(
                unet_lora_parameters,
                text_lora_parameters_one,
                text_lora_parameters_two,
            )

            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.lora_train_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
            self.output_done.emit()

            self.output.emit("Setting up the dataset and the data loader...")
            dataset = LocalImageTextDataset(
                self.lora_train_args.dataset_path,
                tokenizer_one,
                tokenizer_two,
                self.image_size,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                batch_size=self.lora_train_args.batch_size,
                num_workers=self.lora_train_args.workers,
            )
            self.output_done.emit()

            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / self.lora_train_args.accumulation_steps
            )
            max_train_steps = self.lora_train_args.epochs * num_update_steps_per_epoch
            num_warmup_steps = max_train_steps * 0.05
            print(f"Max train steps: {max_train_steps}")
            print(f"Warmup steps: {num_warmup_steps}")

            lr_scheduler = get_scheduler(
                "constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps
                * self.lora_train_args.accumulation_steps,
                num_training_steps=max_train_steps
                * self.lora_train_args.accumulation_steps,
            )

            (
                self.unet,
                self.text_encoder_one,
                self.text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = self.accelerator.prepare(
                self.unet,
                self.text_encoder_one,
                self.text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )

            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / self.lora_train_args.accumulation_steps
            )
            max_train_steps = self.lora_train_args.epochs * num_update_steps_per_epoch
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

            total_batch_size = (
                self.lora_train_args.batch_size
                * self.accelerator.num_processes
                * self.lora_train_args.accumulation_steps
            )

            self.output.emit(f"Number of images: {len(dataset)}")
            self.output.emit(f"Number of epochs: {num_train_epochs}")
            self.output.emit(f"Number of steps per epoch: {num_update_steps_per_epoch}")
            self.output.emit(f"Batch size: {self.lora_train_args.batch_size}")
            self.output.emit(
                f"Batch size with parallel and accumulation: {total_batch_size}"
            )
            self.output.emit(
                f"Gradient accumulation steps: {self.lora_train_args.accumulation_steps}"
            )
            self.output.emit(f"Total training steps: {max_train_steps}")

            global_step = 0
            first_epoch = 0

            self.warning.emit("Starting training loop")
            epoch_losses = []
            for epoch in range(first_epoch, self.lora_train_args.epochs):
                print(epoch)
                epoch_loss = 0
                self.unet.train()
                self.text_encoder_one.train()
                self.text_encoder_two.train()
                train_loss = 0.0

                for _step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(self.unet):
                        pixel_values = batch["pixel_values"]

                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(model_input)
                        bsz = model_input.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=model_input.device,
                        )
                        timesteps = timesteps.long()

                        # Add noise to the model input according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_model_input = noise_scheduler.add_noise(
                            model_input, noise, timesteps
                        )

                        add_time_ids = torch.cat(
                            [
                                self.compute_time_ids(s, c)
                                for s, c in zip(
                                    batch["original_sizes"], batch["crop_top_lefts"]
                                )
                            ]
                        )

                        # Predict the noise residual
                        unet_added_conditions = {"time_ids": add_time_ids}
                        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                            text_encoders=[
                                self.text_encoder_one,
                                self.text_encoder_two,
                            ],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[
                                batch["input_ids_one"],
                                batch["input_ids_two"],
                            ],
                        )
                        unet_added_conditions.update(
                            {"text_embeds": pooled_prompt_embeds}
                        )
                        model_pred = self.unet(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs=unet_added_conditions,
                        ).sample

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(
                                model_input, noise, timesteps
                            )

                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                        avg_loss = loss.mean()
                        train_loss += (
                            avg_loss.item() / self.lora_train_args.accumulation_steps
                        )
                        epoch_loss += avg_loss.item()

                        # Backpropagate
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            params_to_clip = itertools.chain(
                                unet_lora_parameters,
                                text_lora_parameters_one,
                                text_lora_parameters_two,
                            )
                            self.accelerator.clip_grad_norm_(params_to_clip, 1.0)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        global_step += 1
                        print(f"Train loss: {train_loss}")
                        print(f"Step: {global_step}")
                        train_loss = 0.0

                        if global_step % self.lora_train_args.save_steps == 0:
                            save_path = os.path.join(
                                self.lora_train_args.output_dir,
                                f"checkpoint-{global_step}",
                            )
                            self.accelerator.save_state(save_path)

                avg_epoch_loss = epoch_loss / len(train_dataloader)
                epoch_losses.append(avg_epoch_loss)
                self.update_epoch.emit(epoch, avg_epoch_loss)

        else:
            self.error.emit(self.error_text)

    def save_model_hook(self, models, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(self.accelerator.unwrap_model(self.unet))):
                unet_lora_layers_to_save = self.unet_attn_processors_state_dict(model)
            elif isinstance(
                model, type(self.accelerator.unwrap_model(self.text_encoder_one))
            ):
                text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(
                    model
                )
            elif isinstance(
                model, type(self.accelerator.unwrap_model(self.text_encoder_two))
            ):
                text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                    model
                )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )

    def load_model_hook(self, models, input_dir):
        unet = None
        text_encoder_one = None
        text_encoder_two = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(self.accelerator.unwrap_model(self.unet))):
                unet = model
            elif isinstance(
                model, type(self.accelerator.unwrap_model(self.text_encoder_one))
            ):
                text_encoder_one = model
            elif isinstance(
                model, type(self.accelerator.unwrap_model(self.text_encoder_two))
            ):
                text_encoder_two = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet
        )

        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one,
        )

        text_encoder_2_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_two,
        )

    def unet_attn_processors_state_dict(self, unet) -> Dict[str, torch.tensor]:
        """
        Returns:
            a state dict containing just the attention processor parameters.
        """
        attn_processors = unet.attn_processors

        attn_processors_state_dict = {}

        for attn_processor_key, attn_processor in attn_processors.items():
            for parameter_key, parameter in attn_processor.state_dict().items():
                attn_processors_state_dict[
                    f"{attn_processor_key}.{parameter_key}"
                ] = parameter

        return attn_processors_state_dict

    def collate_fn(self, images):
        pixel_values = torch.stack([image["pixel_values"] for image in images])
        original_sizes = [image["original_size"] for image in images]
        crop_top_lefts = [image["crop_top_left"] for image in images]
        input_ids_one = torch.stack([image["input_ids_one"] for image in images])
        input_ids_two = torch.stack([image["input_ids_two"] for image in images])
        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    def compute_time_ids(self, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (self.image_size, self.image_size)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
        return add_time_ids

    def tokenize_prompt(self, tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self, text_encoders, tokenizers, prompt, text_input_ids_list=None
    ):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = self.tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
