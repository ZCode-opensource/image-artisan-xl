from iartisanxl.nodes.node import Node
import json

from iartisanxl.generation.schedulers.schedulers import schedulers


class SchedulerNode(Node):
    OUTPUTS = ["scheduler"]

    def __init__(self, scheduler_index: int = None, **kwargs):
        super().__init__(**kwargs)

        self.scheduler_index = scheduler_index
        self.scheduler_config = None

        with open(
            "./configs/scheduler_config.json", "r", encoding="utf-8"
        ) as config_file:
            self.scheduler_config = json.load(config_file)

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["scheduler_index"] = self.scheduler_index
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(SchedulerNode, cls).from_dict(node_dict)
        node.scheduler_index = node_dict["scheduler_index"]
        return node

    def __call__(self):
        super().__call__()
        scheduler = self.load_scheduler(self.scheduler_index)
        self.values["scheduler"] = scheduler
        return self.values

    def load_scheduler(self, scheduler_index):
        if self.scheduler_config is not None:
            selected_scheduler = schedulers[scheduler_index]
            scheduler_class = selected_scheduler.scheduler_class
            scheduler_args = selected_scheduler.scheduler_args
            use_karras_sigmas = scheduler_args.get("use_karras_sigmas", None)
            algorithm_type = scheduler_args.get("algorithm_type", None)
            noise_sampler_seed = scheduler_args.get("noise_sampler_seed", None)
            euler_at_final = scheduler_args.get("euler_at_final", None)
            use_lu_lambdas = scheduler_args.get("use_lu_lambdas", None)

            scheduler = scheduler_class.from_config(self.scheduler_config)

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

            return scheduler
