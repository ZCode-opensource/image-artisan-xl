from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QFileDialog, QCheckBox, QGridLayout, QSpacerItem, QFrame
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPainter, QPen, QColor

from iartisanxl.app.title_bar import TitleBar
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.configuration.directories_panel import DirectoriesPanel
from iartisanxl.configuration.optimizations_panel import OptimizationsPanel


class SelectDirectoryWidget(QFrame):
    def __init__(
        self,
        selected_dir: str,
        directory_text: str,
        directory_type: int,
        select_function: callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setObjectName("setup_directory_widget")

        self.selected_dir = selected_dir
        self.directory_text = directory_text
        self.directory_type = directory_type
        self.select_function = select_function

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        directory_widget = QWidget(self)
        directory_widget.setObjectName("setup_directory_widget")
        directory_layout = QVBoxLayout(directory_widget)

        select_directory_button = QPushButton(f"Select {self.directory_text.lower()} directory")
        select_directory_button.clicked.connect(lambda: self.select_function(self.directory_type))
        select_directory_button.parent_widget = self
        directory_layout.addWidget(select_directory_button)

        path_layout = QHBoxLayout()
        path_label = QLabel("Path: ")
        path_label.setFixedWidth(90)
        path_layout.addWidget(path_label)
        self.directory_label = QLabel(self.selected_dir)
        path_layout.addWidget(self.directory_label, alignment=Qt.AlignmentFlag.AlignLeft)
        directory_layout.addLayout(path_layout)

        main_layout.addWidget(directory_widget)
        self.setLayout(main_layout)


class PreferencesDialog(QDialog):
    border_color = QColor("#ff6b6b6b")

    steps_panels = [DirectoriesPanel, OptimizationsPanel]

    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(1050, 750)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("preferences_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.directories = directories
        self.preferences = preferences

        self.init_ui()

    def init_ui(self):
        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        title_bar = TitleBar(title="Preferences", is_dialog=True)
        self.dialog_layout.addWidget(title_bar)

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(1)
        diffusers_widget = SelectDirectoryWidget(self.directories.models_diffusers, "Diffusers", 1, self.on_select_directory)
        left_layout.addWidget(diffusers_widget, stretch=0)
        safetensors_widget = SelectDirectoryWidget(self.directories.models_safetensors, "Safetensors", 2, self.on_select_directory)
        left_layout.addWidget(safetensors_widget, stretch=0)
        vaes_widget = SelectDirectoryWidget(self.directories.models_vaes, "Vaes", 3, self.on_select_directory)
        left_layout.addWidget(vaes_widget, stretch=0)
        loras_widget = SelectDirectoryWidget(self.directories.models_loras, "LoRAs", 4, self.on_select_directory)
        left_layout.addWidget(loras_widget, stretch=0)
        controlnets_widget = SelectDirectoryWidget(self.directories.models_controlnets, "ControlNets", 5, self.on_select_directory)
        left_layout.addWidget(controlnets_widget, stretch=0)
        t2i_adapters_widget = SelectDirectoryWidget(self.directories.models_t2i_adapters, "T2I Adapters", 6, self.on_select_directory)
        left_layout.addWidget(t2i_adapters_widget, stretch=0)
        ip_adapters_widget = SelectDirectoryWidget(self.directories.models_ip_adapters, "IP Adapters", 7, self.on_select_directory)
        left_layout.addWidget(ip_adapters_widget, stretch=0)
        upscalers_widget = SelectDirectoryWidget(self.directories.models_upscalers, "Upscalers", 11, self.on_select_directory)
        left_layout.addWidget(upscalers_widget, stretch=0)
        output_loras_widget = SelectDirectoryWidget(self.directories.outputs_loras, "Output loras", 9, self.on_select_directory)
        left_layout.addWidget(output_loras_widget, stretch=0)
        datasets_widget = SelectDirectoryWidget(self.directories.datasets, "Datasets", 10, self.on_select_directory)
        left_layout.addWidget(datasets_widget, stretch=0)
        # left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 0)

        inner_right_layout = QGridLayout()
        inner_right_layout.setSpacing(10)
        self.intermediate_images_checkbox = QCheckBox("Display intermediate images")
        self.intermediate_images_checkbox.setChecked(self.preferences.intermediate_images)
        self.intermediate_images_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        inner_right_layout.addWidget(self.intermediate_images_checkbox, 0, 0)
        self.tomes_base_checkbox = QCheckBox("Enable token merging")
        self.tomes_base_checkbox.setChecked(self.preferences.use_tomes)
        self.tomes_base_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        inner_right_layout.addWidget(self.tomes_base_checkbox, 0, 1)
        self.offload_base_checkbox = QCheckBox("Enable model CPU offload")
        self.offload_base_checkbox.setChecked(self.preferences.model_offload)
        self.offload_base_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        inner_right_layout.addWidget(self.offload_base_checkbox, 1, 0)
        self.sequential_offload_checkbox = QCheckBox("Enable sequential CPU offload")
        self.sequential_offload_checkbox.setChecked(self.preferences.sequential_offload)
        self.sequential_offload_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        inner_right_layout.addWidget(self.sequential_offload_checkbox, 1, 1)
        right_layout.addLayout(inner_right_layout)

        right_layout.addSpacerItem(QSpacerItem(0, 10))

        image_options_layout = QGridLayout()
        image_options_layout.setSpacing(10)
        self.save_image_metadata_checkbox = QCheckBox("Save image metadata")
        self.save_image_metadata_checkbox.setChecked(self.preferences.save_image_metadata)
        self.save_image_metadata_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        image_options_layout.addWidget(self.save_image_metadata_checkbox, 0, 0)
        self.save_image_control_preprocessors_checkbox = QCheckBox("Save image control preprocessors")
        self.save_image_control_preprocessors_checkbox.setChecked(self.preferences.save_image_control_preprocessors)
        self.save_image_control_preprocessors_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        image_options_layout.addWidget(self.save_image_control_preprocessors_checkbox, 0, 1)
        self.save_image_control_sources_checkbox = QCheckBox("Save image control sources")
        self.save_image_control_sources_checkbox.setChecked(self.preferences.save_image_control_sources)
        self.save_image_control_sources_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        image_options_layout.addWidget(self.save_image_control_sources_checkbox, 1, 0)
        self.hide_nsfw_checkbox = QCheckBox("Hide NSFW tag")
        self.hide_nsfw_checkbox.setChecked(self.preferences.hide_nsfw)
        self.hide_nsfw_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        image_options_layout.addWidget(self.hide_nsfw_checkbox, 1, 1)
        right_layout.addLayout(image_options_layout, stretch=1)

        right_layout.addSpacerItem(QSpacerItem(0, 10))

        images_widget = SelectDirectoryWidget(self.directories.outputs_images, "Output images", 8, self.on_select_directory)
        right_layout.addWidget(images_widget, stretch=0)
        right_layout.addStretch()

        self.main_layout.addLayout(left_layout)
        self.main_layout.addLayout(right_layout)
        self.main_layout.setStretch(0, 1)
        self.main_layout.setStretch(1, 1)

        self.dialog_layout.addLayout(self.main_layout)
        self.setLayout(self.dialog_layout)

    def on_checkbox_state_changed(self):
        sender = self.sender()

        settings = QSettings("ZCode", "ImageArtisanXL")

        if sender == self.intermediate_images_checkbox:
            settings.setValue("intermediate_images", sender.isChecked())
            self.preferences.intermediate_images = sender.isChecked()
        elif sender == self.tomes_base_checkbox:
            settings.setValue("use_tomes", sender.isChecked())
            self.preferences.use_tomes = sender.isChecked()
        elif sender == self.offload_base_checkbox:
            settings.setValue("model_offload", sender.isChecked())
            self.preferences.model_offload = sender.isChecked()
        elif sender == self.sequential_offload_checkbox:
            settings.setValue("sequential_offload", sender.isChecked())
            self.preferences.sequential_offload = sender.isChecked()
        elif sender == self.save_image_metadata_checkbox:
            settings.setValue("save_image_metadata", sender.isChecked())
            self.preferences.save_image_metadata = sender.isChecked()
        elif sender == self.save_image_control_preprocessors_checkbox:
            settings.setValue("save_image_control_preprocessors", sender.isChecked())
            self.preferences.save_image_control_preprocessors = sender.isChecked()
        elif sender == self.save_image_control_sources_checkbox:
            settings.setValue("save_image_control_sources", sender.isChecked())
            self.preferences.save_image_control_sources = sender.isChecked()
        elif sender == self.hide_nsfw_checkbox:
            settings.setValue("hide_nsfw", sender.isChecked())
            self.preferences.hide_nsfw = sender.isChecked()

    def on_select_directory(self, dir_type):
        dialog = QFileDialog()
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        settings = QSettings("ZCode", "ImageArtisanXL")

        if dir_type == 1:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_diffusers)
            if len(selected_path) > 0:
                self.directories.models_diffusers = selected_path
                settings.setValue("models_diffusers", selected_path)
        elif dir_type == 2:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_safetensors)
            if len(selected_path) > 0:
                self.directories.models_safetensors = selected_path
                settings.setValue("models_safetensors", selected_path)
        elif dir_type == 3:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_vaes)
            if len(selected_path) > 0:
                self.directories.models_vaes = selected_path
                settings.setValue("vaes", selected_path)
        elif dir_type == 4:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_loras)
            if len(selected_path) > 0:
                self.directories.models_loras = selected_path
                settings.setValue("models_loras", selected_path)
        elif dir_type == 5:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_controlnets)
            if len(selected_path) > 0:
                self.directories.models_controlnets = selected_path
                settings.setValue("models_controlnets", selected_path)
        elif dir_type == 6:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_t2i_adapters)
            if len(selected_path) > 0:
                self.directories.models_t2i_adapters = selected_path
                settings.setValue("models_t2i_adapters", selected_path)
        elif dir_type == 7:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_ip_adapters)
            if len(selected_path) > 0:
                self.directories.models_ip_adapters = selected_path
                settings.setValue("models_ip_adapters", selected_path)
        elif dir_type == 8:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.outputs_images)
            if len(selected_path) > 0:
                self.directories.outputs_images = selected_path
                settings.setValue("outputs_images", selected_path)
        elif dir_type == 9:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.outputs_loras)
            if len(selected_path) > 0:
                self.directories.outputs_loras = selected_path
                settings.setValue("outputs_loras", selected_path)
        elif dir_type == 10:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.datasets)
            if len(selected_path) > 0:
                self.directories.models_upscalers = selected_path
                settings.setValue("datasets", selected_path)
        elif dir_type == 11:
            selected_path = dialog.getExistingDirectory(None, "Select a directory", self.directories.models_upscalers)
            if len(selected_path) > 0:
                self.directories.datasets = selected_path
                settings.setValue("models_upscalers", selected_path)

        if len(selected_path) > 0:
            sender_button = self.sender()
            sender_button.parent_widget.directory_label.setText(selected_path)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def closeEvent(self, event):
        self.settings.beginGroup("preferences_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)
