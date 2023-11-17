import attr


@attr.s
class ModelDataObject:
    name: str = attr.ib(default="")
    path: str = attr.ib(default="")
    version: str = attr.ib(default="")
    type: str = attr.ib(default="")

    def copy(self):
        new_obj = ModelDataObject(
            name=self.name,
            path=self.path,
            version=self.version,
            type=self.type,
        )
        return new_obj
