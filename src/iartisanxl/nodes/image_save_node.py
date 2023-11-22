import os
from datetime import datetime

from iartisanxl.nodes.node import Node


class ImageSaveNode(Node):
    REQUIRED_INPUTS = ["image"]
    OUTPUTS = []

    def __init__(self, directory: str = None, filename: str = None, **kwargs):
        super().__init__(**kwargs)

        self.directory = directory
        self.filename = filename

        if self.directory is None:
            self.directory = "./outputs/images"

            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

    def __call__(self):
        if self.filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}.png"
        else:
            filename = self.filename

        self.image.save(os.path.join(self.directory, filename))
