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
        image_generation_data: ImageGenData,
        prompt_window: PromptWindow,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.prompt_window = prompt_window

    def update_ui(self):
        pass
