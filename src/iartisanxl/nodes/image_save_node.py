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

        if self.filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename = f"{timestamp}.png"

    def __call__(self):
        self.image.save(os.path.join(self.directory, self.filename))
