import sys
import unittest
from PyQt6.QtWidgets import QApplication

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog

app = QApplication(sys.argv)


class TestBaseDialog(unittest.TestCase):
    def setUp(self):
        directories = None
        preferences = None
        title = "Test"
        show_error = lambda x: None
        image_generation_data = None
        image_viewer = None
        prompt_window = None

        self.dialog = BaseDialog(
            directories,
            preferences,
            title,
            show_error,
            image_generation_data,
            image_viewer,
            prompt_window,
        )

    def test_init(self):
        self.assertEqual(self.dialog.windowTitle(), "Test")
