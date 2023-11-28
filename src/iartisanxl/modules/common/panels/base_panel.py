from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.lora_list import LoraList
from iartisanxl.generation.controlnet_list import ControlNetList
from iartisanxl.modules.common.prompt_window import PromptWindow


class BasePanel(QWidget):
    dialog_opened = pyqtSignal(object, type, str)

    def __init__(
        self,
        directories: DirectoriesObject,
        prompt_window: PromptWindow,
        show_error: callable,
        image_generation_data: ImageGenerationData,
        lora_list: LoraList,
        controlnet_list: ControlNetList,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.controlnet_list = controlnet_list
        self.prompt_window = prompt_window
        self.show_error = show_error
        self.current_dialog = None

    def __del__(self):
        self.image_generation_data = None
