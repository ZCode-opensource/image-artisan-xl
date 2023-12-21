from iartisanxl.graph.nodes.node import Node

from iartisanxl.generation.schedulers.schedulers import schedulers


class SchedulerNode(Node):
    OUTPUTS = ["scheduler"]

    def __init__(self, scheduler_index: int = None, **kwargs):
        super().__init__(**kwargs)

        self.scheduler_index = scheduler_index

    def update_value(self, scheduler_index: int):
        self.scheduler_index = scheduler_index
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["scheduler_index"] = self.scheduler_index
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(SchedulerNode, cls).from_dict(node_dict)
        node.scheduler_index = node_dict["scheduler_index"]
        return node

    def update_inputs(self, node_dict):
        self.scheduler_index = node_dict["scheduler_index"]

    def __call__(self):
        super().__call__()
        scheduler = self.load_scheduler(self.scheduler_index)
        self.values["scheduler"] = scheduler
        return self.values

    def load_scheduler(self, scheduler_index):
        scheduler_config_dict = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
        }

        selected_scheduler = schedulers[scheduler_index]

        is_turbo = self.check_turbo(selected_scheduler.name)

        scheduler_class = selected_scheduler.scheduler_class
        scheduler_args = selected_scheduler.scheduler_args
        use_karras_sigmas = scheduler_args.get("use_karras_sigmas", None)
        algorithm_type = scheduler_args.get("algorithm_type", None)
        noise_sampler_seed = scheduler_args.get("noise_sampler_seed", None)
        euler_at_final = scheduler_args.get("euler_at_final", None)
        use_lu_lambdas = scheduler_args.get("use_lu_lambdas", None)
        rescale_betas_zero_snr = scheduler_args.get("rescale_betas_zero_snr", None)

        scheduler = scheduler_class.from_config(scheduler_config_dict)

        if is_turbo:
            scheduler.config.timestep_spacing = "trailing"

        if selected_scheduler.name != "LCM":
            scheduler.config.interpolation_type = "linear"
            scheduler.config.skip_prk_steps = True

        if use_karras_sigmas is not None:
            scheduler.config.use_karras_sigmas = use_karras_sigmas

        if algorithm_type is not None:
            scheduler.config.algorithm_type = algorithm_type

        if noise_sampler_seed is not None:
            scheduler.config.noise_sampler_seed = noise_sampler_seed

        if euler_at_final is not None:
            scheduler.config.euler_at_final = euler_at_final

        if use_lu_lambdas is not None:
            scheduler.config.use_lu_lambdas = use_lu_lambdas

        if rescale_betas_zero_snr is not None:
            scheduler.config.rescale_betas_zero_snr = rescale_betas_zero_snr

        return scheduler

    def check_turbo(self, scheduler_name):
        return "Turbo" in scheduler_name
