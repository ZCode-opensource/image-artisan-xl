import attr


@attr.s(slots=True)
class VaeDataObject:
    name: str = attr.ib(default="")
    path: str = attr.ib(default="")
