import sys
import unittest
from PyQt6.QtWidgets import QApplication

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog

app = QApplication(sys.argv)


class TestBaseDialog(unittest.TestCase):
    def setUp(self):
        directories = None
        title = "Test"
        show_error = lambda x: None
        image_viewer = None
        prompt_window = None
        auto_generate_function = lambda: None

        self.dialog = BaseDialog(
            directories,
            title,
            show_error,
            image_viewer,
            prompt_window,
            auto_generate_function,
        )

    def test_init(self):
        self.assertEqual(self.dialog.windowTitle(), "Test")
