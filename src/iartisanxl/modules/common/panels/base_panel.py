from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.modules.common.prompt_window import PromptWindow


class BasePanel(QWidget):
    dialog_opened = pyqtSignal(type, str)

    def __init__(
        self,
        directories: DirectoriesObject,
        prompt_window: PromptWindow,
        image_generation_data: ImageGenData,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.prompt_window = prompt_window

    def update_ui(self, image_generation_data: ImageGenData):
        self.image_generation_data = image_generation_data

    def __del__(self):
        self.image_generation_data = None
