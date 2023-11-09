import os

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.taesd import taesd


class TaesdLoaderThread(QThread):
    taesd_loaded = pyqtSignal()
    taesd_error = pyqtSignal(str)

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.taesd_dec = None

    def run(self):
        self.taesd_dec = taesd.Decoder()
        self.taesd_dec = self.taesd_dec.to(self.device)

        try:
            self.taesd_dec.load_state_dict(
                torch.load(
                    "./models/taesd/taesdxl_decoder.pth", map_location=self.device
                )
            )
        except FileNotFoundError:
            self.taesd_error.emit(
                "Taesd model not found, can't enable intermediate images."
            )
            return

        self.taesd_loaded.emit()
