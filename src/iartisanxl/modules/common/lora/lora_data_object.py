import attr


@attr.s(slots=True)
class LoraDataObject:
    name = attr.ib(type=str)
    filename = attr.ib(type=str)
    version = attr.ib(type=str)
    path = attr.ib(type=str)
    enabled = attr.ib(type=bool, default=True)
    weight = attr.ib(type=float, default=1.00)
    node_id: int = attr.ib(default=None)
    lora_id = attr.ib(default=None)
