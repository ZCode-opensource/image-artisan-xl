from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.lora_list import LoraList
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog


class BasePanel(QWidget):
    dialog_opened = pyqtSignal(type, str)

    def __init__(
        self,
        directories: DirectoriesObject,
        prompt_window: PromptWindow,
        show_error: callable,
        image_generation_data: ImageGenerationData,
        lora_list: LoraList,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.prompt_window = prompt_window
        self.show_error = show_error

    def update_ui(self, image_generation_data: ImageGenerationData):
        self.image_generation_data = image_generation_data

    def process_dialog(self, dialog: BaseDialog):
        pass

    def __del__(self):
        self.image_generation_data = None
