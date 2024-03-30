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
    advanced_weights = attr.ib(type=dict, default=None)
    node_id: int = attr.ib(default=None)
    lora_id = attr.ib(default=None)

    def get_weights(self):
        if self.advanced_weights is None:
            return {
                "text_encoder_one": self.text_encoder_one_weight,
                "text_encoder_two": self.text_encoder_two_weight,
                "unet": self.unet_weight,
            }
        else:
            return self.advanced_weights

    def set_weights(self, weights):
        if isinstance(weights, float):
            self.text_encoder_one_weight = weights
            self.text_encoder_two_weight = weights
            self.unet_weight = weights
        elif isinstance(weights, dict):
            if "unet" in weights and isinstance(weights["unet"], dict):
                self.advanced_weights = weights["unet"]
            else:
                self.text_encoder_one_weight = weights.get("text_encoder_one", self.text_encoder_one_weight)
                self.text_encoder_two_weight = weights.get("text_encoder_two", self.text_encoder_two_weight)
                self.unet_weight = weights.get("unet", self.unet_weight)
