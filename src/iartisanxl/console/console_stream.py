import io
import re
import sys

from PyQt6.QtCore import QObject


class ConsoleStream(QObject):
    def __init__(self):
        super().__init__()
        self.string_io = io.StringIO()
        self.last_percentage = 0

    def write(self, text):
        self.string_io.write(text)
        # sys.__stdout__.write("Text: " + repr(text) + "\n")

        match = re.search(r"(\d+)%", text)
        if match:
            percentage = int(match.group(1))
            if abs(percentage - self.last_percentage) >= 1:
                self.last_percentage = percentage
