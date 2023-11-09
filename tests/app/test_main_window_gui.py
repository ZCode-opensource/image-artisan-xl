# from unittest.mock import patch, Mock

# from pytestqt.qtbot import QtBot
# from PyQt6.QtWidgets import QApplication

# from iartisanxl.app.main_window import MainWindow
# from iartisanxl.app.directories import DirectoriesObject
# from iartisanxl.app.preferences import PreferencesObject

# directories = DirectoriesObject(
#     models_diffusers="",
#     models_safetensors="",
#     vaes="",
#     models_loras="",
#     outputs_images="",
# )

# preferences = PreferencesObject(
#     intermediate_images=False,
#     use_tomes=False,
#     sequential_offload=False,
#     model_offload=False,
# )


# def test_close_splash_called_after_timer(qtbot: QtBot):
#     with patch.object(QApplication, "instance") as mock_instance:
#         mock_instance.return_value.close_splash = Mock()

#         window = MainWindow(directories, preferences, splash_timer_duration=100)
#         qtbot.wait(200)

#         assert mock_instance.return_value.close_splash.called
