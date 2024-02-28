from iartisanxl.diffusers_patch.ip_adapter_attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
from iartisanxl.graph.nodes.node import Node


class IPAdapterMergeNode(Node):
    REQUIRED_INPUTS = ["ip_adapter", "unet"]
    OUTPUTS = ["ip_adapter"]

    def __call__(self) -> dict:
        self.unet.set_attn_processor(AttnProcessor2_0())

        if self.ip_adapter is not None:
            ip_adapters = self.ip_adapter

            if isinstance(ip_adapters, dict):
                ip_adapters = [ip_adapters]

            weights = []
            scales = []

            for ip_adapter in ip_adapters:
                weights.append(ip_adapter["weights"])
                scales.append(ip_adapter["scale"])

            attn_procs = self.convert_ip_adapter_attn_to_diffusers(weights)
            self.unet.set_attn_processor(attn_procs)

            for attn_processor in self.unet.attn_processors.values():
                if isinstance(attn_processor, IPAdapterAttnProcessor2_0):
                    attn_processor.scale = scales

        self.values["ip_adapter"] = self.ip_adapter

        return self.values

    def before_delete(self):
        if self.unet is not None:
            self.unet.set_attn_processor(AttnProcessor2_0())

    def convert_ip_adapter_attn_to_diffusers(self, state_dicts):
        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = AttnProcessor2_0
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = IPAdapterAttnProcessor2_0
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                ).to(dtype=self.torch_dtype, device=self.device)

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                attn_procs[name].load_state_dict(value_dict)
                key_id += 2

        return attn_procs
