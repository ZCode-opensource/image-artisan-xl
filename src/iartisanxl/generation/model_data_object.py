import attr


@attr.s
class ModelDataObject:
    name = attr.ib(type=str)
    path = attr.ib(type=str)
    version = attr.ib(type=str)
    type = attr.ib(type=str)

    def copy(self):
        new_obj = ModelDataObject(
            name=self.name,
            path=self.path,
            version=self.version,
            type=self.type,
        )
        return new_obj
