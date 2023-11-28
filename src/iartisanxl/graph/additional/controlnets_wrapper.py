from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.controlnet import ControlNetModel, ControlNetOutput


class ControlnetsWrapper(torch.nn.Module):
    def __init__(
        self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]
    ):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        assert len(self.nets) == len(
            controlnet_cond
        ), "Number of controlnet conditions must match number of controlnets"
        assert len(self.nets) == len(
            conditioning_scale
        ), "Number of conditioning scales must match number of controlnets"

        for i, (image, scale, controlnet) in enumerate(
            zip(controlnet_cond, conditioning_scale, self.nets)
        ):
            down_samples, mid_sample = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                conditioning_scale=scale,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                guess_mode=guess_mode,
                return_dict=return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(
                        down_block_res_samples, down_samples
                    )
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample
