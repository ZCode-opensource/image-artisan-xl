import json

from iartisanxl.generation.schedulers.schedulers import schedulers


def load_scheduler(scheduler_index):
    selected_scheduler = schedulers[scheduler_index]
    scheduler_class = selected_scheduler.scheduler_class
    scheduler_args = selected_scheduler.scheduler_args
    use_karras_sigmas = scheduler_args.get("use_karras_sigmas", None)
    algorithm_type = scheduler_args.get("algorithm_type", None)
    noise_sampler_seed = scheduler_args.get("noise_sampler_seed", None)

    scheduler_config = None
    with open("./configs/scheduler_config.json", "r", encoding="utf-8") as config_file:
        scheduler_config = json.load(config_file)
    scheduler = scheduler_class.from_config(scheduler_config)

    if use_karras_sigmas is not None:
        scheduler.config.use_karras_sigmas = use_karras_sigmas

    if algorithm_type is not None:
        scheduler.config.algorithm_type = algorithm_type

    if noise_sampler_seed is not None:
        scheduler.config.noise_sampler_seed = noise_sampler_seed

    return scheduler