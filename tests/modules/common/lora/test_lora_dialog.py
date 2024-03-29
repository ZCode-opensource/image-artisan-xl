import sys

import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.modules.common.lora.lora_dialog import LoraDialog


app = QApplication(sys.argv)


class TestLoraDialog(unittest.TestCase):
    def setUp(self):
        directories = DirectoriesObject(
            models_diffusers="",
            models_safetensors="",
            models_vaes="",
            models_loras="",
            models_controlnets="",
            models_t2i_adapters="",
            models_ip_adapters="",
            models_upscalers="",
            outputs_images="",
            outputs_loras="",
            datasets="",
        )

        preferences = PreferencesObject(
            intermediate_images=False,
            use_tomes=False,
            sequential_offload=False,
            model_offload=False,
            save_image_metadata=False,
            save_image_control_preprocessors=False,
            save_image_control_sources=False,
            hide_nsfw=False,
        )

        title = "Test"
        show_error = lambda x: None
        image_generation_data = None
        image_viewer = None
        prompt_window = None

        self.dialog = LoraDialog(directories, preferences, title, show_error, image_generation_data, image_viewer, prompt_window)

    def test_init(self):
        self.assertEqual(self.dialog.windowTitle(), "LoRAs")
        self.assertEqual(self.dialog.minimumSize().width(), 1160)
        self.assertEqual(self.dialog.minimumSize().height(), 800)

    def test_on_lora_item_clicked(self):
        data = {"name": "test", "root_filename": "test_filename", "version": "1.0", "filepath": "/path/to/file"}

        with patch(
            "iartisanxl.modules.common.lora.lora_info_widget.get_metadata_from_safetensors",
            return_value={},
        ):
            self.dialog.on_lora_item_clicked(data)

            self.assertIsNotNone(self.dialog.selected_lora)
            self.assertEqual(self.dialog.selected_lora.name, data["name"])
            self.assertEqual(self.dialog.selected_lora.filename, data["root_filename"])
            self.assertEqual(self.dialog.selected_lora.version, data["version"])
            self.assertEqual(self.dialog.selected_lora.path, data["filepath"])
