from typing import List, Optional

import torch
from torch import nn

from diffusers.models.adapter import T2IAdapter


class T2IAdaptersWrapper(torch.nn.Module):
    def __init__(self, adapters: List["T2IAdapter"]):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)

        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError("Not all adapters have the same downscaling behavior.")

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[List[torch.Tensor]]:
        assert len(self.adapters) == len(xs), "Number of inputs must match number of adapters"
        assert len(self.adapters) == len(adapter_weights), "Number of weights must match number of adapters"

        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / len(self.adapters)] * len(self.adapters))
        else:
            adapter_weights = torch.tensor(adapter_weights)

        accume_states = [[torch.zeros_like(feature) for feature in adapter(xs[0])] for adapter in self.adapters]
        for x, w, adapter in zip(xs, adapter_weights, self.adapters):
            features = adapter(x)
            for i, feature in enumerate(features):
                for _j, state in enumerate(accume_states):
                    state[i] += w * feature
        return accume_states
