import attr


@attr.s
class VaeDataObject:
    name: str = attr.ib(default="")
    path: str = attr.ib(default="")

    def copy(self):
        new_obj = VaeDataObject(
            name=self.name,
            path=self.path,
        )
        return new_obj
