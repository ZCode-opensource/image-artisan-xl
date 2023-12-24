import os
import math
import itertools
import gc

import torch
import torch.nn.functional as F
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from peft import LoraConfig
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict

from iartisanxl.train.lora_train_args import LoraTrainArgs
from iartisanxl.train.local_image_dataset import LocalImageTextDataset


class DreamboothLoraTrainThread(QThread):
    output = pyqtSignal(str)
    output_done = pyqtSignal()
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    ready_to_start = pyqtSignal(int)
    update_step = pyqtSignal(int)
    update_epoch = pyqtSignal(int, float, float, str)
    training_finished = pyqtSignal(int, float, str)
    aborted = pyqtSignal()

    def __init__(self, lora_train_args: LoraTrainArgs):
        super().__init__()

        self.lora_train_args = lora_train_args
        self.abort = False

        if not os.path.isdir(self.lora_train_args.output_dir):
            os.makedirs(self.lora_train_args.output_dir, exist_ok=False)

        self.image_size = 1024
        self.weight_dtype = torch.bfloat16
        self.accelerator = None
        self.unet = None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.vae = None
        self.scheduler = None

    def run(self):
        self.output.emit(f"Output directory: {self.lora_train_args.output_dir}")
        self.output.emit(f"Model: {self.lora_train_args.model_path}")
        self.output.emit(f"Dataset: {self.lora_train_args.dataset_path}")

        self.output.emit("Setting up accelerator...")
        accelerator_project_config = ProjectConfiguration(project_dir=self.lora_train_args.output_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.lora_train_args.accumulation_steps,
            mixed_precision="bf16",
            project_config=accelerator_project_config,
        )
        self.output_done.emit()

        # Load the tokenizers
        self.output.emit("Loading the tokenizers...")
        try:
            self.tokenizer_one = CLIPTokenizer.from_pretrained(self.lora_train_args.model_path, subfolder="tokenizer", use_fast=False)
            self.tokenizer_two = CLIPTokenizer.from_pretrained(self.lora_train_args.model_path, subfolder="tokenizer_2", use_fast=False)
        except OSError:
            self.error.emit("Couldn't load the tokenizers.")
            return
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Setting the noise scheduler...")
        try:
            self.scheduler = DDPMScheduler.from_pretrained(self.lora_train_args.model_path, subfolder="scheduler")
        except OSError:
            self.error.emit("Couldn't load the scheduler.")
            return
        self.output_done.emit()

        self.output.emit("Loading text encoders...")
        try:
            self.text_encoder_one = CLIPTextModel.from_pretrained(self.lora_train_args.model_path, subfolder="text_encoder", variant="fp16")
            self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                self.lora_train_args.model_path, subfolder="text_encoder_2", variant="fp16"
            )
        except OSError:
            self.error.emit("Couldn't load the text encoders.")
            return
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Loading vae...")
        try:
            if len(self.lora_train_args.vae_path) > 0:
                self.vae = AutoencoderKL.from_pretrained(self.lora_train_args.vae_path, variant="fp16")
            else:
                self.vae = AutoencoderKL.from_pretrained(self.lora_train_args.model_path, subfolder="vae", variant="fp16")
        except OSError:
            self.error.emit("Couldn't load the VAE.")
            return
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Loading unet...")
        try:
            self.unet = UNet2DConditionModel.from_pretrained(self.lora_train_args.model_path, subfolder="unet", variant="fp16")
        except OSError:
            self.error.emit("Couldn't load the unet.")
            return
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Setting up the LoRA...")
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)

        self.unet.enable_gradient_checkpointing()
        self.text_encoder_one.gradient_checkpointing_enable()
        self.text_encoder_two.gradient_checkpointing_enable()

        # now we will add new LoRA weights to the attention layers
        unet_lora_config = LoraConfig(
            r=self.lora_train_args.rank,
            lora_alpha=self.lora_train_args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config)
        self.output_done.emit()

        # The text encoder comes from transformers, so we cannot directly modify it.
        # So, instead, we monkey-patch the forward calls of its attention-blocks.
        text_lora_config = LoraConfig(
            r=self.lora_train_args.rank,
            lora_alpha=self.lora_train_args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        self.text_encoder_one.add_adapter(text_lora_config)
        self.text_encoder_two.add_adapter(text_lora_config)

        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)

        torch.backends.cuda.matmul.allow_tf32 = True

        unet_lora_parameters = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, self.text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, self.text_encoder_two.parameters()))
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Creating the optimizer...")
        # Optimization parameters
        unet_lora_parameters_with_lr = {
            "params": unet_lora_parameters,
            "lr": self.lora_train_args.learning_rate,
        }
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": self.lora_train_args.adam_weight_decay_text_encoder,
            "lr": self.lora_train_args.text_encoder_learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": self.lora_train_args.adam_weight_decay_text_encoder,
            "lr": self.lora_train_args.text_encoder_learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]

        # Optimizer creation
        if self.lora_train_args.optimizer == "prodigy":
            import prodigyopt

            optimizer_class = prodigyopt.Prodigy

            params_to_optimize[1]["lr"] = self.lora_train_args.learning_rate
            params_to_optimize[2]["lr"] = self.lora_train_args.learning_rate

            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.lora_train_args.learning_rate,
                betas=(self.lora_train_args.adam_beta1, self.lora_train_args.adam_beta2),
                beta3=self.lora_train_args.prodigy_beta3,
                weight_decay=self.lora_train_args.adam_weight_decay,
                eps=self.lora_train_args.adam_epsilon,
                decouple=self.lora_train_args.prodigy_decouple,
                use_bias_correction=self.lora_train_args.prodigy_use_bias_correction,
                safeguard_warmup=self.lora_train_args.prodigy_safeguard_warmup,
            )
        else:
            if self.lora_train_args.optimizer == "adamw8bit":
                import bitsandbytes as bnb

                optimizer_class = bnb.optim.AdamW8bit
            elif self.lora_train_args.optimizer == "adamw":
                optimizer_class = torch.optim.AdamW

            optimizer = optimizer_class(
                params_to_optimize,
                betas=(self.lora_train_args.adam_beta1, self.lora_train_args.adam_beta2),
                weight_decay=self.lora_train_args.adam_weight_decay,
                eps=self.lora_train_args.adam_epsilon,
            )

        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        self.output.emit("Loading the dataset...")
        dataset = LocalImageTextDataset(
            self.lora_train_args.dataset_path,
            self.tokenizer_one,
            self.tokenizer_two,
            self.image_size,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.lora_train_args.batch_size,
            num_workers=0,
        )
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        # Handle instance prompt.
        instance_time_ids = self.compute_time_ids()
        add_time_ids = instance_time_ids

        self.output.emit("Setting up the learning scheduler...")
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.lora_train_args.accumulation_steps)
        max_train_steps = self.lora_train_args.epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            self.lora_train_args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lora_train_args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=max_train_steps * self.accelerator.num_processes,
            num_cycles=self.lora_train_args.lr_num_cycles,
            power=self.lora_train_args.lr_power,
        )
        self.output_done.emit()

        self.output.emit("Preparing the optimized paramaters with accelerator...")
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
        self.output_done.emit()

        if self.abort:
            self.aborted.emit()
            return

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.lora_train_args.accumulation_steps)

        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        self.accelerator.init_trackers("dreambooth-lora-sd-xl", config=vars(self.lora_train_args))

        # Train!
        self.ready_to_start.emit(max_train_steps)

        total_batch_size = self.lora_train_args.batch_size * self.accelerator.num_processes * self.lora_train_args.accumulation_steps

        self.output.emit(f"Number of images: {len(dataset)}")
        self.output.emit(f"Num batches each epoch = {len(train_dataloader)}")
        self.output.emit(f"Number of epochs: {num_train_epochs}")
        self.output.emit(f"Number of steps per epoch: {num_update_steps_per_epoch}")
        self.output.emit(f"Batch size: {self.lora_train_args.batch_size}")
        self.output.emit(f"Batch size with parallel and accumulation: {total_batch_size}")
        self.output.emit(f"Gradient accumulation steps: {self.lora_train_args.accumulation_steps}")
        self.output.emit(f"Warmp up steps: {self.lora_train_args.lr_warmup_steps}")
        self.output.emit(f"Total training steps: {max_train_steps}")

        self.output.emit("")
        self.output.emit("Starting training loop")

        global_step = 0
        first_epoch = 0

        if self.lora_train_args.resume_checkpoint is not None:
            path = os.path.basename(self.lora_train_args.resume_checkpoint)
            self.output.emit(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.lora_train_args.output_dir, path))
            first_epoch = int(path.split("-")[1])
            global_step = first_epoch * num_update_steps_per_epoch

        total_avg_loss = 0
        total_steps = 0

        if self.abort:
            self.aborted.emit()
            return

        for epoch in range(first_epoch, num_train_epochs):
            self.unet.train()
            self.text_encoder_one.train()
            self.text_encoder_two.train()

            self.text_encoder_one.text_model.embeddings.requires_grad_(True)
            self.text_encoder_two.text_model.embeddings.requires_grad_(True)

            avg_epoch_loss = 0
            epoch_steps = 0

            for _step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
                    tokens_one = batch["input_ids_one"]
                    tokens_two = batch["input_ids_two"]

                    # Convert images to latent space
                    model_input = self.vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=model_input.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

                    # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                    elems_to_repeat_text_embeds = 1
                    elems_to_repeat_time_ids = bsz

                    unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                        text_encoders=[self.text_encoder_one, self.text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)})
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)

                    if self.abort:
                        self.aborted.emit()
                        return

                    model_pred = self.unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample

                    if self.abort:
                        self.aborted.emit()
                        return

                    # Get the target for loss depending on the prediction type
                    if self.scheduler.config.prediction_type == "epsilon":  # pylint: disable=no-member
                        target = noise
                    elif self.scheduler.config.prediction_type == "v_prediction":  # pylint: disable=no-member
                        target = self.scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")  # pylint: disable=no-member

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    avg_epoch_loss += loss.item()
                    epoch_steps += 1
                    total_steps += 1

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = itertools.chain(
                            unet_lora_parameters,
                            text_lora_parameters_one,
                            text_lora_parameters_two,
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.lora_train_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    self.update_step.emit(total_steps)

                if self.abort:
                    self.aborted.emit()
                    return

                if self.accelerator.sync_gradients:
                    global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                self.accelerator.log(logs, step=global_step)

            image_path = ""

            if self.abort:
                self.aborted.emit()
                return

            if (epoch + 1) % self.lora_train_args.save_epochs == 0 and (epoch + 1) < self.lora_train_args.epochs:
                self.output.emit(f"Saving checkpoint: checkpoint-{epoch + 1}...")
                save_path = os.path.join(self.lora_train_args.output_dir, f"checkpoint-{epoch + 1}")
                self.output_done.emit()
                self.output.emit("Saving state...")
                self.accelerator.save_state(save_path)
                self.output_done.emit()

                if self.lora_train_args.validation_prompt:
                    self.output.emit("Generating validation image...")
                    self.unet.eval()
                    self.text_encoder_one.eval()
                    self.text_encoder_two.eval()

                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        self.lora_train_args.model_path,
                        vae=self.vae,
                        text_encoder=self.accelerator.unwrap_model(self.text_encoder_one),
                        text_encoder_2=self.accelerator.unwrap_model(self.text_encoder_two),
                        unet=self.accelerator.unwrap_model(self.unet),
                        variant="fp16",
                        torch_dtype=self.weight_dtype,
                    )

                    scheduler_args = {"use_karras_sigmas": True}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                    pipeline = pipeline.to(self.accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    generator = (
                        torch.Generator(device=self.accelerator.device).manual_seed(self.lora_train_args.seed) if self.lora_train_args.seed else None
                    )
                    pipeline_args = {"prompt": self.lora_train_args.validation_prompt}

                    if self.abort:
                        self.aborted.emit()
                        return

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            image = pipeline(
                                **pipeline_args,
                                guidance_scale=7.5,
                                num_inference_steps=20,
                                generator=generator,
                            ).images[0]

                    filename = f"checkpoint-{epoch + 1}.png"
                    image_path = os.path.join(self.lora_train_args.output_dir, filename)
                    image.save(image_path)

                    if self.abort:
                        self.aborted.emit()
                        return

                    del pipeline
                    torch.cuda.empty_cache()
                    self.output_done.emit()

            if epoch_steps > 0:
                avg_epoch_loss /= epoch_steps

            self.update_epoch.emit(epoch + 1, lr_scheduler.get_last_lr()[0], avg_epoch_loss, image_path)
            total_avg_loss += avg_epoch_loss

        # Save the lora layers
        self.output.emit("Saving final LoRA...")
        self.unet = self.accelerator.unwrap_model(self.unet)
        self.unet = self.unet.to(torch.float32)
        unet_lora_layers = get_peft_model_state_dict(self.unet)
        unet_lora_config = self.unet.peft_config["default"]

        self.text_encoder_one = self.accelerator.unwrap_model(self.text_encoder_one)
        text_encoder_lora_layers = get_peft_model_state_dict(self.text_encoder_one.to(torch.float32))
        self.text_encoder_two = self.accelerator.unwrap_model(self.text_encoder_two)
        text_encoder_2_lora_layers = get_peft_model_state_dict(self.text_encoder_two.to(torch.float32))

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=self.lora_train_args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        if self.abort:
            self.aborted.emit()
            return

        vae = AutoencoderKL.from_pretrained(
            self.lora_train_args.vae_path,
            variant="fp16",
            torch_dtype=self.weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.lora_train_args.model_path,
            vae=vae,
            variant="fp16",
            torch_dtype=self.weight_dtype,
        )

        scheduler_args = {"use_karras_sigmas": True}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.load_lora_weights(self.lora_train_args.output_dir)
        self.output_done.emit()

        final_image_path = ""

        if self.lora_train_args.validation_prompt:
            self.output.emit("Generating final image...")
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.lora_train_args.seed) if self.lora_train_args.seed else None

            if self.abort:
                self.aborted.emit()
                return

            with torch.no_grad():
                image = pipeline(
                    self.lora_train_args.validation_prompt,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]

            filename = "final.png"
            final_image_path = os.path.join(self.lora_train_args.output_dir, filename)
            image.save(final_image_path)
            self.output_done.emit()

        self.accelerator.end_training()
        total_avg_loss /= num_train_epochs
        self.training_finished.emit(num_train_epochs, total_avg_loss, final_image_path)

        gc.collect()
        torch.cuda.empty_cache()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(self.accelerator.unwrap_model(self.unet))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = get_peft_model_state_dict(model)
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
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(self.accelerator.unwrap_model(self.unet))):
                unet_ = model
            elif isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(self.accelerator.unwrap_model(self.text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

        text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_)

        text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_)

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

    def compute_time_ids(self):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (1024, 1024)
        target_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
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
    def encode_prompt(self, text_encoders, tokenizers, prompt, text_input_ids_list=None):
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
