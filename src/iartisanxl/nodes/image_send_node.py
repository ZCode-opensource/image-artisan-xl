from iartisanxl.nodes.node import Node


class ImageSendNode(Node):
    REQUIRED_INPUTS = ["image"]
    OUTPUTS = []

    def __init__(self, image_callback: callable, **kwargs):
        super().__init__(**kwargs)

        self.image_callback = image_callback

    def __call__(self):
        self.image_callback(self.image)
