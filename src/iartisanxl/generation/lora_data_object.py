import attr


@attr.s
class LoraDataObject:
    name = attr.ib(type=str)
    filename = attr.ib(type=str)
    version = attr.ib(type=str)
    path = attr.ib(type=str)
    weight = attr.ib(type=float, default=1.0)

    def copy(self):
        new_obj = LoraDataObject(
            name=self.name,
            filename=self.filename,
            version=self.version,
            path=self.path,
            weight=self.weight,
        )
        return new_obj