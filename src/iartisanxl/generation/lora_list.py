import copy

import attr

from iartisanxl.generation.lora_data_object import LoraDataObject


@attr.s(auto_attribs=True, slots=True)
class LoraList:
    loras: list[LoraDataObject] = attr.Factory(list)
    _original_loras: list[LoraDataObject] = attr.Factory(list)

    def add(self, lora):
        self.loras.append(lora)

    def update_lora(self, lora_filename, new_values):
        for lora in self.loras:
            if lora.filename == lora_filename:
                for attr_name, new_value in new_values.items():
                    if attr_name != "filename":
                        setattr(lora, attr_name, new_value)
                break

    def get_lora_by_filename(self, filename):
        for lora in self.loras:
            if lora.filename == filename:
                return lora
        return None

    def update_lora_by_id(self, lora_id, new_values):
        for lora in self.loras:
            if lora.id == lora_id:
                for attr_name, new_value in new_values.items():
                    if attr_name != "filename":
                        setattr(lora, attr_name, new_value)
                break

    def get_lora_by_id(self, lora_id):
        for lora in self.loras:
            if lora.id == lora_id:
                return lora
        return None

    def remove(self, lora):
        self.loras.remove(lora)

    def save_state(self):
        self._original_loras = copy.deepcopy(self.loras)

    def get_added(self):
        return [lora for lora in self.loras if lora not in self._original_loras]

    def get_removed(self):
        return [lora for lora in self._original_loras if lora not in self.loras]

    def get_modified(self):
        return [
            lora
            for lora, original_lora in zip(self.loras, self._original_loras)
            if lora != original_lora
        ]
