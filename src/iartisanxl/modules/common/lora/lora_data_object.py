import attr


@attr.s(slots=True)
class LoraDataObject:
    name = attr.ib(type=str)
    filename = attr.ib(type=str)
    version = attr.ib(type=str)
    path = attr.ib(type=str)
    enabled = attr.ib(type=bool, default=True)
    unet_weight = attr.ib(type=float, default=1.00)
    text_encoder_one_weight = attr.ib(type=float, default=1.00)
    text_encoder_two_weight = attr.ib(type=float, default=1.00)
    granular_unet_weights_enabled: bool = attr.ib(default=False)
    granular_unet_weights: dict = attr.ib(
        default=attr.Factory(
            lambda: {
                "down": {"block_1": [1.0, 1.0], "block_2": [1.0, 1.0]},
                "mid": 1.0,
                "up": {"block_0": [1.0, 1.0, 1.0], "block_1": [1.0, 1.0, 1.0]},
            }
        )
    )
    node_id: int = attr.ib(default=None)
    lora_id = attr.ib(default=None)
    locked = attr.ib(type=bool, default=True)
    is_slider = attr.ib(type=bool, default=False)
