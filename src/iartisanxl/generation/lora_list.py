import copy

import attr

from iartisanxl.generation.lora_data_object import LoraDataObject


@attr.s(auto_attribs=True, slots=True)
class LoraList:
    loras: list[LoraDataObject] = attr.Factory(list)
    _original_loras: list[LoraDataObject] = attr.Factory(list)

    def add(self, lora):
        if not any(
            existing_lora.filename == lora.filename for existing_lora in self.loras
        ):
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
        original_filenames = [lora.filename for lora in self._original_loras]
        return [lora for lora in self.loras if lora.filename not in original_filenames]

    def get_removed(self):
        current_filenames = [lora.filename for lora in self.loras]
        return [
            lora
            for lora in self._original_loras
            if lora.filename not in current_filenames
        ]

    def get_modified(self):
        original_loras_dict = {lora.filename: lora for lora in self._original_loras}
        return [
            lora
            for lora in self.loras
            if lora.filename in original_loras_dict
            and lora != original_loras_dict[lora.filename]
        ]
