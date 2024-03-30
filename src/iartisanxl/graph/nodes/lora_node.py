import os
import re

from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules
from diffusers.utils.peft_utils import delete_adapter_layers, get_adapter_name, get_peft_kwargs, scale_lora_layers
from diffusers.utils.state_dict_utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
)
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

from iartisanxl.graph.iartisan_node_error import IArtisanNodeError
from iartisanxl.graph.nodes.node import Node


class LoraNode(Node):
    PRIORITY = 1
    REQUIRED_INPUTS = ["unet", "text_encoder_1", "text_encoder_2", "global_lora_scale"]
    OUTPUTS = ["lora"]

    def __init__(
        self,
        path: str = None,
        adapter_name: str = None,
        scale: dict = None,
        lora_name: str = None,
        version: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.path = path
        self.adapter_name = adapter_name
        self.scale = scale
        self.lora_name = lora_name
        self.version = version

    def update_lora(self, scale: dict, enabled: bool):
        self.scale = scale
        self.enabled = enabled
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["adapter_name"] = self.adapter_name
        node_dict["scale"] = self.scale
        node_dict["lora_name"] = self.lora_name
        node_dict["version"] = self.version
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(LoraNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.adapter_name = node_dict["adapter_name"]
        node.scale = node_dict["scale"]
        node.lora_name = node_dict["lora_name"]
        node.version = node_dict["version"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]
        self.adapter_name = node_dict["adapter_name"]
        self.scale = node_dict["scale"]
        self.lora_name = node_dict["lora_name"]
        self.version = node_dict["version"]

    def __call__(self):
        if self.adapter_name not in getattr(self.unet, "peft_config", {}):
            # Check if the file exists
            if not os.path.isfile(self.path):
                raise IArtisanNodeError(f"LoRA file not found: {self.path}", self.name)

            state_dict, network_alphas = self.lora_state_dict(self.path, unet_config=self.unet.config)

            is_correct_format = all("lora" in key for key in state_dict.keys())
            if not is_correct_format:
                raise ValueError("Invalid LoRA checkpoint.")

            self.load_lora_into_unet(
                state_dict, network_alphas=network_alphas, unet=self.unet, adapter_name=self.adapter_name
            )

            text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
            if len(text_encoder_state_dict) > 0:
                self.load_lora_into_text_encoder(
                    text_encoder_state_dict,
                    network_alphas=network_alphas,
                    text_encoder=self.text_encoder_1,
                    prefix="text_encoder",
                    lora_scale=self.global_lora_scale,
                    adapter_name=self.adapter_name,
                )

            text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
            if len(text_encoder_2_state_dict) > 0:
                self.load_lora_into_text_encoder(
                    text_encoder_2_state_dict,
                    network_alphas=network_alphas,
                    text_encoder=self.text_encoder_2,
                    prefix="text_encoder_2",
                    lora_scale=self.global_lora_scale,
                    adapter_name=self.adapter_name,
                )

        scale = self.scale
        if not self.enabled:
            scale = 0.0

        self.values["lora"] = (self.adapter_name, scale)

        return self.values

    def before_delete(self):
        if self.unet is not None:
            delete_adapter_layers(self.unet, self.adapter_name)

            if hasattr(self.unet, "peft_config") and self.unet.peft_config is not None:
                self.unet.peft_config.pop(self.adapter_name, None)

        if self.text_encoder_1 is not None:
            delete_adapter_layers(self.text_encoder_1, self.adapter_name)

        if self.text_encoder_2 is not None:
            delete_adapter_layers(self.text_encoder_2, self.adapter_name)

    def lora_state_dict(self, lora_path, **kwargs):
        unet_config = kwargs.pop("unet_config", None)

        state_dict = load_file(lora_path, device="cpu")

        network_alphas = None

        if all(
            (
                k.startswith("lora_te_")
                or k.startswith("lora_unet_")
                or k.startswith("lora_te1_")
                or k.startswith("lora_te2_")
            )
            for k in state_dict.keys()
        ):
            # Map SDXL blocks correctly.
            if unet_config is not None:
                # use unet config to remap block numbers
                state_dict = self._maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)

            (state_dict, network_alphas) = self._convert_kohya_lora_to_diffusers(state_dict)

        return state_dict, network_alphas

    def load_lora_into_unet(self, state_dict, network_alphas, unet, adapter_name=None):
        keys = list(state_dict.keys())

        if all(key.startswith("unet") or key.startswith("text_encoder") for key in keys):
            unet_keys = [k for k in keys if k.startswith("unet")]
            state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k in unet_keys}

            if network_alphas is not None:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith("unet")]
                network_alphas = {k.replace("unet.", ""): v for k, v in network_alphas.items() if k in alpha_keys}

        if len(state_dict.keys()) > 0:
            if adapter_name in getattr(unet, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the Unet - please select a new adapter name."
                )

            state_dict = convert_unet_state_dict_to_peft(state_dict)

            if network_alphas is not None:
                # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                # `convert_unet_state_dict_to_peft` method.
                network_alphas = convert_unet_state_dict_to_peft(network_alphas)

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(unet)

            inject_adapter_in_model(lora_config, unet, adapter_name=adapter_name)
            set_peft_model_state_dict(unet, state_dict, adapter_name)

        unet.load_attn_procs(state_dict, network_alphas=network_alphas)

    def load_lora_into_text_encoder(
        self, state_dict, network_alphas, text_encoder, prefix=None, lora_scale=1.0, adapter_name=None
    ):
        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        prefix = "text_encoder" if prefix is None else prefix

        # Safe prefix to check with.
        if any("text_encoder" in key for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
            text_encoder_lora_state_dict = {
                k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }

            if len(text_encoder_lora_state_dict) > 0:
                rank = {}
                text_encoder_lora_state_dict = convert_state_dict_to_diffusers(text_encoder_lora_state_dict)

                # convert state dict
                text_encoder_lora_state_dict = convert_state_dict_to_peft(text_encoder_lora_state_dict)

                for name, _ in text_encoder_attn_modules(text_encoder):
                    rank_key = f"{name}.out_proj.lora_B.weight"
                    rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

                patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                if patch_mlp:
                    for name, _ in text_encoder_mlp_modules(text_encoder):
                        rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                        rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                        rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                        rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]

                if network_alphas is not None:
                    alpha_keys = [
                        k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix
                    ]
                    network_alphas = {
                        k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                lora_config_kwargs = get_peft_kwargs(
                    rank,
                    network_alphas,
                    text_encoder_lora_state_dict,
                    is_unet=False,
                )

                lora_config = LoraConfig(**lora_config_kwargs)

                # adapter_name
                if adapter_name is None:
                    adapter_name = get_adapter_name(text_encoder)

                # inject LoRA layers and load the state dict
                # in transformers we automatically check whether the adapter name is already in use or not
                text_encoder.load_adapter(
                    adapter_name=adapter_name,
                    adapter_state_dict=text_encoder_lora_state_dict,
                    peft_config=lora_config,
                )

                # scale LoRA layers with `lora_scale`
                scale_lora_layers(text_encoder, weight=lora_scale)

                text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

    def _maybe_map_sgm_blocks_to_diffusers(self, state_dict, unet_config, delimiter="_", block_slice_pos=5):
        # 1. get all state_dict_keys
        all_keys = list(state_dict.keys())
        sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

        # 2. check if needs remapping, if not return original dict
        is_in_sgm_format = False
        for key in all_keys:
            if any(p in key for p in sgm_patterns):
                is_in_sgm_format = True
                break

        if not is_in_sgm_format:
            return state_dict

        # 3. Else remap from SGM patterns
        new_state_dict = {}
        inner_block_map = ["resnets", "attentions", "upsamplers"]

        # Retrieves # of down, mid and up blocks
        input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

        for layer in all_keys:
            if "text" in layer:
                new_state_dict[layer] = state_dict.pop(layer)
            else:
                layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
                if sgm_patterns[0] in layer:
                    input_block_ids.add(layer_id)
                elif sgm_patterns[1] in layer:
                    middle_block_ids.add(layer_id)
                elif sgm_patterns[2] in layer:
                    output_block_ids.add(layer_id)
                else:
                    raise ValueError(f"Checkpoint not supported because layer {layer} not supported.")

        input_blocks = {
            layer_id: [key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key]
            for layer_id in input_block_ids
        }
        middle_blocks = {
            layer_id: [key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key]
            for layer_id in middle_block_ids
        }
        output_blocks = {
            layer_id: [key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key]
            for layer_id in output_block_ids
        }

        # Rename keys accordingly
        for i in input_block_ids:
            block_id = (i - 1) // (unet_config.layers_per_block + 1)
            layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

            for key in input_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
                inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in middle_block_ids:
            key_part = None
            if i == 0:
                key_part = [inner_block_map[0], "0"]
            elif i == 1:
                key_part = [inner_block_map[1], "0"]
            elif i == 2:
                key_part = [inner_block_map[0], "1"]
            else:
                raise ValueError(f"Invalid middle block id {i}.")

            for key in middle_blocks[i]:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1] + key_part + key.split(delimiter)[block_slice_pos:]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in output_block_ids:
            block_id = i // (unet_config.layers_per_block + 1)
            layer_in_block_id = i % (unet_config.layers_per_block + 1)

            for key in output_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = inner_block_map[inner_block_id]
                inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        if len(state_dict) > 0:
            raise ValueError("At this point all state dict entries have to be converted.")

        return new_state_dict

    def _convert_kohya_lora_to_diffusers(self, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        te2_state_dict = {}
        network_alphas = {}

        # every down weight has a corresponding up weight and potentially an alpha weight
        lora_keys = [k for k in state_dict.keys() if k.endswith("lora_down.weight")]
        for key in lora_keys:
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"

            if lora_name.startswith("lora_unet_"):
                diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

                if "input.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
                else:
                    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")

                if "middle.block" in diffusers_name:
                    diffusers_name = diffusers_name.replace("middle.block", "mid_block")
                else:
                    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                if "output.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
                else:
                    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")

                diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                diffusers_name = diffusers_name.replace("proj.in", "proj_in")
                diffusers_name = diffusers_name.replace("proj.out", "proj_out")
                diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

                # SDXL specificity.
                if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
                    pattern = r"\.\d+(?=\D*$)"
                    diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
                if ".in." in diffusers_name:
                    diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
                if ".out." in diffusers_name:
                    diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
                if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
                    diffusers_name = diffusers_name.replace("op", "conv")
                if "skip" in diffusers_name:
                    diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

                # LyCORIS specificity.
                if "time.emb.proj" in diffusers_name:
                    diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
                if "conv.shortcut" in diffusers_name:
                    diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

                # General coverage.
                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                        diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                        unet_state_dict[diffusers_name] = state_dict.pop(key)
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                    elif "ff" in diffusers_name:
                        unet_state_dict[diffusers_name] = state_dict.pop(key)
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                else:
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            elif lora_name.startswith("lora_te_"):
                diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # (sayakpaul): Duplicate code. Needs to be cleaned.
            elif lora_name.startswith("lora_te1_"):
                diffusers_name = key.replace("lora_te1_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # (sayakpaul): Duplicate code. Needs to be cleaned.
            elif lora_name.startswith("lora_te2_"):
                diffusers_name = key.replace("lora_te2_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Rename the alphas so that they can be mapped appropriately.
            if lora_name_alpha in state_dict:
                alpha = state_dict.pop(lora_name_alpha).item()
                if lora_name_alpha.startswith("lora_unet_"):
                    prefix = "unet."
                elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
                    prefix = "text_encoder."
                else:
                    prefix = "text_encoder_2."
                new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
                network_alphas.update({new_name: alpha})

        if len(state_dict) > 0:
            raise ValueError(
                f"The following keys have not been correctly be renamed: \n\n {', '.join(state_dict.keys())}"
            )

        unet_state_dict = {f"unet.{module_name}": params for module_name, params in unet_state_dict.items()}
        te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}
        te2_state_dict = (
            {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
            if len(te2_state_dict) > 0
            else None
        )
        if te2_state_dict is not None:
            te_state_dict.update(te2_state_dict)

        new_state_dict = {**unet_state_dict, **te_state_dict}
        return new_state_dict, network_alphas
