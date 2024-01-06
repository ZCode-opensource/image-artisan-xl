import math

import torch
import cv2
import collections
import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from iartisanxl.models.upscalers.rrdbnet import RRDBNet


class ImageUpscaleThread(QThread):
    status_update = pyqtSignal(str)
    setup_progress = pyqtSignal(int)
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    upscale_done = pyqtSignal(QPixmap)

    def __init__(self, device, image_path: str, model_path: str):
        super().__init__()

        self.device = device
        self.image_path = image_path
        self.model_path = model_path
        self.scale_index = 2
        self.model = None
        self.scale_factor = self.scale_index**2
        self.tile_size = 1024
        self.tile_padding = 0.05

    def run(self):
        if self.model_path is not None:
            self.status_update.emit("Loading upscaler model...")

            state_dict = torch.load(self.model_path)
            keymap = self.build_legacy_keymap(2)
            state_dict = {keymap[k]: v for k, v in state_dict.items()}

            model_net = RRDBNet(3, 3, 64, 23)
            model_net.load_state_dict(state_dict, 2, strict=True)

            del keymap
            del state_dict

            for _, v in model_net.named_parameters():
                v.requires_grad = False

            self.model = model_net.to(self.device)

            self.status_update.emit("Model loaded.")

            self.status_update.emit("loading image...")
            input_image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member

            if input_image.shape[2] == 4:
                input_image = input_image[:, :, 0:3]

            self.status_update.emit("Starting upscale process...")
            width, height, depth = input_image.shape
            output_width = width * self.scale_factor
            output_height = height * self.scale_factor
            output_shape = (output_width, output_height, depth)

            # start with black image
            output_image = np.zeros(output_shape, np.uint8)
            tile_padding = math.ceil(self.tile_size * self.tile_padding)
            tile_size = math.ceil(self.tile_size / self.scale_factor)

            tiles_x = math.ceil(width / tile_size)
            tiles_y = math.ceil(height / tile_size)

            self.setup_progress.emit(tiles_x * tiles_y)
            self.status_update.emit("Upscaling...")

            # modified from https://github.com/ata4/esrgan-launcher
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # extract tile from input image
                    ofs_x = x * tile_size
                    ofs_y = y * tile_size

                    # input tile area on total image
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_size, width)

                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_size, height)

                    # input tile area on total image with padding
                    input_start_x_pad = max(input_start_x - tile_padding, 0)
                    input_end_x_pad = min(input_end_x + tile_padding, width)

                    input_start_y_pad = max(input_start_y - tile_padding, 0)
                    input_end_y_pad = min(input_end_y + tile_padding, height)

                    # input tile dimensions
                    input_tile_width = input_end_x - input_start_x
                    input_tile_height = input_end_y - input_start_y

                    tile_idx = y * tiles_x + x + 1

                    # update progress
                    self.progress_update.emit(tile_idx)

                    input_tile = input_image[input_start_x_pad:input_end_x_pad, input_start_y_pad:input_end_y_pad]
                    output_tile = self.upscale(input_tile)

                    # output tile area on total image
                    output_start_x = input_start_x * self.scale_factor
                    output_end_x = input_end_x * self.scale_factor

                    output_start_y = input_start_y * self.scale_factor
                    output_end_y = input_end_y * self.scale_factor

                    # output tile area without padding
                    output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                    output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor

                    output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                    output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                    # put tile into output image
                    output_image[output_start_x:output_end_x, output_start_y:output_end_y] = output_tile[
                        output_start_x_tile:output_end_x_tile, output_start_y_tile:output_end_y_tile
                    ]

            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            image = Image.fromarray(output_image)
            image = image.resize((image.size[0] // 2, image.size[1] // 2), Image.LANCZOS)

            qimage = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format.Format_RGB888)
            qpixmap = QPixmap.fromImage(qimage)

            self.upscale_done.emit(qpixmap)
            self.status_update.emit("Upscale done...")

            del model_net
            self.model = None

    def build_legacy_keymap(self, n_upscale):
        keymap = collections.OrderedDict()
        keymap["model.0"] = "conv_first"

        for i in range(23):
            for j in range(1, 4):
                for k in range(1, 6):
                    k1 = f"model.1.sub.{i}.RDB{j}.conv{k}.0"
                    k2 = f"RRDB_trunk.{i}.RDB{j}.conv{k}"
                    keymap[k1] = k2

        keymap["model.1.sub.23"] = "trunk_conv"

        n = 0
        for i in range(1, n_upscale + 1):
            n += 3
            k1 = f"model.{n}"
            k2 = f"upconv{i}"
            keymap[k1] = k2

        keymap[f"model.{(n + 2)}"] = "HRconv"
        keymap[f"model.{(n + 4)}"] = "conv_last"

        # add ".weigth" and ".bias" suffixes to all keys
        keymap_final = collections.OrderedDict()

        for k1, k2 in keymap.items():
            for k_type in ("weight", "bias"):
                k1_f = k1 + "." + k_type
                k2_f = k2 + "." + k_type
                keymap_final[k1_f] = k2_f

        return keymap_final

    def upscale(self, input_image):
        input_image = input_image * 1.0 / np.iinfo(input_image.dtype).max
        input_image = np.transpose(input_image[:, :, [2, 1, 0]], (2, 0, 1))
        input_image = torch.from_numpy(input_image).float()
        input_image = input_image.unsqueeze(0).to(self.device)

        image_output = self.model(input_image).data.squeeze().float().cpu().clamp(0, 1).numpy()  # pylint: disable=not-callable
        image_output = np.transpose(image_output[[2, 1, 0], :, :], (1, 2, 0))
        image_output = (image_output * 255.0).round().astype(np.uint8)

        return image_output
