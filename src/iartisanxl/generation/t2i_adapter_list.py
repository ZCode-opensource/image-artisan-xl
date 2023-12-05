import copy

import attr

from iartisanxl.generation.t2i_adapter_data_object import T2IAdapterDataObject


@attr.s(auto_attribs=True, slots=True)
class T2IAdapterList:
    adapters: list[T2IAdapterDataObject] = attr.Factory(list)
    _original_adapters: list[T2IAdapterDataObject] = attr.Factory(list)
    dropped_image: bool = attr.ib(default=False)
    _next_id: int = attr.ib(default=0)

    def add(self, adapter: T2IAdapterDataObject):
        adapter.adapter_id = self._next_id
        self._next_id += 1
        self.adapters.append(adapter)

        return adapter.adapter_id

    def update_adapter(self, adapter_id, new_values):
        for adapter in self.adapters:
            if adapter.adapter_id == adapter_id:
                for attr_name, new_value in new_values.items():
                    if attr_name != "adapter_id":
                        setattr(adapter, attr_name, new_value)
                break

    def update_with_adapter_data_object(self, new_adapter: T2IAdapterDataObject):
        for i, adapter in enumerate(self.adapters):
            if adapter.adapter_id == new_adapter.adapter_id:
                self.adapters[i] = new_adapter
                break

    def get_adapter(self, adapter_id):
        for adapter in self.adapters:
            if adapter.id == adapter_id:
                return adapter
        return None

    def remove(self, controlnet):
        self.adapters.remove(controlnet)

    def save_state(self):
        self._original_adapters = copy.deepcopy(self.adapters)

    def get_added(self):
        original_ids = [controlnet.adapter_id for controlnet in self._original_adapters]
        return [controlnet for controlnet in self.adapters if controlnet.adapter_id not in original_ids]

    def get_removed(self):
        current_ids = [adapter.adapter_id for adapter in self.adapters]
        return [adapter for adapter in self._original_adapters if adapter.adapter_id not in current_ids]

    def get_modified(self):
        original_adapters_dict = {adapter.adapter_id: adapter for adapter in self._original_adapters}
        return [
            adapter for adapter in self.adapters if adapter.adapter_id in original_adapters_dict and adapter != original_adapters_dict[adapter.adapter_id]
        ]

    def clear_adapters(self):
        self.adapters.clear()

    def get_used_types(self):
        return list(set([adapter.adapter_type for adapter in self.adapters]))
