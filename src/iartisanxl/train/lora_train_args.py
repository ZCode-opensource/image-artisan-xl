import attr


@attr.s
class LoraTrainArgs:
    output_dir = attr.ib()
    model_path = attr.ib()
    vae_path = attr.ib()
    dataset_path = attr.ib()
    rank = attr.ib(4)
    learning_rate = attr.ib(default=1e-4)
    optimizer = attr.ib(default="adamw8bit")
    adam_beta1 = attr.ib(default=0.9)
    adam_beta2 = attr.ib(default=0.999)
    adam_weight_decay = attr.ib(1e-04)
    adam_epsilon = attr.ib(1e-08)
    text_encoder_learning_rate = attr.ib(1e-4)
    adam_weight_decay_text_encoder = attr.ib(1e-03)
    lr_scheduler = attr.ib(default="constant")  # "linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"
    lr_warmup_steps = attr.ib(default=0)
    lr_num_cycles = attr.ib(default=1)
    lr_power = attr.ib(default=1.0)
    max_grad_norm = attr.ib(default=1.0)
    batch_size = attr.ib(1)
    workers = attr.ib(8)
    accumulation_steps = attr.ib(1)
    epochs = attr.ib(12)
    save_epochs = attr.ib(1)
    validation_prompt = attr.ib(default=None)
    seed = attr.ib(default=None)
    resume_checkpoint = attr.ib(default=None)
    prodigy_decouple = attr.ib(default=True)
    prodigy_use_bias_correction = attr.ib(default=True)
    prodigy_safeguard_warmup = attr.ib(default=True)
