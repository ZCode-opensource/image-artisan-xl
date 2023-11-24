import attr


@attr.s(slots=True)
class ModelDataObject:
    name: str = attr.ib(default="")
    path: str = attr.ib(default="")
    version: str = attr.ib(default="")
    type: str = attr.ib(default="")
