import logging
import os

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.pipelines.convert_pipeline import ImageArtisanConvertPipeline


class ConvertSafetensorsThread(QThread):
    status_changed = pyqtSignal(str, int)

    def __init__(self, safetensors_filepath: str, root_filename: str, diffusers_directory: str):
        super().__init__()

        self.safetensors_filepath = safetensors_filepath
        self.root_filename = root_filename
        self.diffusers_directory = diffusers_directory
        self.logger = logging.getLogger()

    def run(self):
        self.status_changed.emit("Loading model", 0)

        pipeline = ImageArtisanConvertPipeline.from_single_file(
            self.safetensors_filepath,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
            original_config_file="./configs/sd_xl_base.yaml",
        )

        self.status_changed.emit("Model loaded", 1)

        pipeline.save_to_diffusers(
            os.path.join(self.diffusers_directory, self.root_filename),
            status_update=self.on_status_changed,
            variant="fp16",
        )

        pipeline = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        os.remove(self.safetensors_filepath)

    def on_status_changed(self, message: str, step: int):
        self.status_changed.emit(message, step)
