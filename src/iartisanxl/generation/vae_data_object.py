import attr


@attr.s
class VaeDataObject:
    name = attr.ib(type=str)
    path = attr.ib(type=str)

    def copy(self):
        new_obj = VaeDataObject(
            name=self.name,
            path=self.path,
        )
        return new_obj
