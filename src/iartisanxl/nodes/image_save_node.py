# pylint: disable=no-member
import os
from datetime import datetime
from PIL import Image

from iartisanxl.nodes.node import Node


class ImageSaveNode(Node):
    PRIORITY = 7
    REQUIRED_ARGS = []
    INPUTS = ["image"]
    OUTPUTS = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.directory = kwargs.get("directory", None)
        self.filename = kwargs.get("filename", None)

        if self.directory is None:
            self.directory = "./outputs/images"

            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

        if self.filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename = f"{timestamp}.png"

    def __call__(self, image: Image):
        image.save(os.path.join(self.directory, self.filename))
