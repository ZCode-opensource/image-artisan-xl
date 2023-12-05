from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.lora_list import LoraList
from iartisanxl.generation.controlnet_list import ControlNetList
from iartisanxl.generation.t2i_adapter_list import T2IAdapterList
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple


class BasePanel(QWidget):
    dialog_opened = pyqtSignal(object, type, str)

    def __init__(
        self,
        module_options: dict,
        preferences: PreferencesObject,
        directories: DirectoriesObject,
        image_viewer: ImageViewerSimple,
        prompt_window: PromptWindow,
        show_error: callable,
        image_generation_data: ImageGenerationData,
        lora_list: LoraList,
        controlnet_list: ControlNetList,
        t2i_adapter_list: T2IAdapterList,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module_options = module_options
        self.preferences = preferences
        self.directories = directories
        self.image_viewer = image_viewer
        self.prompt_window = prompt_window
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.controlnet_list = controlnet_list
        self.t2i_adapter_list = t2i_adapter_list
        self.show_error = show_error

    def clean_up(self):
        pass
