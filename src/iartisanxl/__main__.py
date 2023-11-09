import sys
import os
import logging.config

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .app.artisan import ArtisanApplication
from .app.logging_conf import logging_config


def my_exception_hook(exctype, value, traceback):
    # Do something with the exception here, such as logging it
    print(f"Unhandled exception: {value}")
    sys.__excepthook__(exctype, value, traceback)


sys.excepthook = my_exception_hook


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    app = ArtisanApplication(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    os.makedirs(
        os.path.dirname(logging_config["handlers"]["fileHandler"]["filename"]),
        exist_ok=True,
    )
    logging.config.dictConfig(logging_config)
    main()
