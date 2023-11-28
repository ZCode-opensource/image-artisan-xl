import copy

import attr

from iartisanxl.generation.controlnet_data_object import ControlNetDataObject


@attr.s(auto_attribs=True, slots=True)
class ControlNetList:
    controlnets: list[ControlNetDataObject] = attr.Factory(list)
    _original_controlnets: list[ControlNetDataObject] = attr.Factory(list)
    dropped_image: bool = attr.ib(default=False)
    _next_id: int = attr.ib(default=0)

    def add(self, controlnet: ControlNetDataObject):
        controlnet.controlnet_id = self._next_id
        self._next_id += 1
        self.controlnets.append(controlnet)

        return controlnet.controlnet_id

    def update_controlnet(self, controlnet_id, new_values):
        for controlnet in self.controlnets:
            if controlnet.controlnet_id == controlnet_id:
                for attr_name, new_value in new_values.items():
                    if attr_name != "controlnet_id":
                        setattr(controlnet, attr_name, new_value)
                break

    def update_with_controlnet_data_object(self, new_controlnet: ControlNetDataObject):
        print(f"{new_controlnet=}")
        for i, controlnet in enumerate(self.controlnets):
            if controlnet.controlnet_id == new_controlnet.controlnet_id:
                self.controlnets[i] = new_controlnet
                break

    def get_controlnet(self, controlnet_id):
        for controlnet in self.controlnets:
            if controlnet.id == controlnet_id:
                return controlnet
        return None

    def remove(self, controlnet):
        self.controlnets.remove(controlnet)

    def save_state(self):
        self._original_controlnets = copy.deepcopy(self.controlnets)

    def get_added(self):
        original_ids = [controlnet.controlnet_id for controlnet in self._original_controlnets]
        return [controlnet for controlnet in self.controlnets if controlnet.controlnet_id not in original_ids]

    def get_removed(self):
        current_ids = [controlnet.controlnet_id for controlnet in self.controlnets]
        return [controlnet for controlnet in self._original_controlnets if controlnet.controlnet_id not in current_ids]

    def get_modified(self):
        original_controlnets_dict = {controlnet.controlnet_id: controlnet for controlnet in self._original_controlnets}
        return [
            controlnet
            for controlnet in self.controlnets
            if controlnet.controlnet_id in original_controlnets_dict and controlnet != original_controlnets_dict[controlnet.controlnet_id]
        ]

    def clear_controlnets(self):
        self.controlnets.clear()

    def get_used_types(self):
        return list(set([controlnet.controlnet_type for controlnet in self.controlnets]))
