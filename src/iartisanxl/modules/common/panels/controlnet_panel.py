import os

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.controlnet_dialog import ControlNetDialog
from iartisanxl.modules.common.controlnet_added_item import ControlNetAddedItem
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.generation.controlnet_data_object import ControlNetDataObject
from iartisanxl.formats.image import ImageProcessor


class ControlNetPanel(BasePanel):
    def __init__(
        self,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.preferences = preferences

        self.controlnets = []
        self.controlnet_id_counter = 0

        self.init_ui()
        self.update_ui(self.image_generation_data)

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_controlnet_button = QPushButton("Add ControlNet")
        add_controlnet_button.clicked.connect(self.open_controlnet_dialog)
        main_layout.addWidget(add_controlnet_button)

        added_controlnets_widget = QWidget()
        self.controlnets_layout = QVBoxLayout(added_controlnets_widget)
        main_layout.addWidget(added_controlnets_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_controlnet_dialog(self):
        self.dialog_opened.emit(ControlNetDialog, "ControlNet")

    def update_ui(self, image_generation_data: ImageGenData):
        super().update_ui(image_generation_data)

        controlnets = self.image_generation_data.controlnets
        self.clear_controlnets()

        if len(controlnets) > 0:
            for controlnet in controlnets:
                controlnet_widget = ControlNetAddedItem(controlnet)
                controlnet_widget.remove_clicked.connect(
                    lambda lw=controlnet_widget: self.on_remove_controlnet(lw)
                )
                controlnet_widget.enabled.connect(
                    lambda lw=controlnet_widget: self.on_controlnet_enabled(lw)
                )
                self.controlnets_layout.addWidget(controlnet_widget)
                self.controlnets.append(controlnet_widget)

    def process_dialog(self, dialog: ControlNetDialog):
        if not dialog.annotator_widget.image_editor.has_photo():
            self.show_error("You'll need to generate or set an annotator image.")
            return

        # Check if a ControlNetDataObject with the same controlnet_id already exists
        existing_controlnet_widget = next(
            (
                cn
                for cn in self.controlnets
                if cn.controlnet.controlnet_id == dialog.controlnet_id
            ),
            None,
        )

        type_index = dialog.controlnet_combo.currentIndex()

        controlnet_model = None
        if type_index == 0:
            controlnet_model = "controlnet-canny-sdxl-1.0-small"
        elif type_index == 1:
            controlnet_model = "controlnet-depth-sdxl-1.0-small"

        if existing_controlnet_widget is not None:
            # If it exists, update it
            existing_controlnet_widget.controlnet.name = (
                dialog.controlnet_combo.currentText()
            )
            existing_controlnet_widget.controlnet.model_path = os.path.join(
                self.directories.models_controlnets, controlnet_model
            )
            existing_controlnet_widget.controlnet.enabled = True
            existing_controlnet_widget.controlnet.guess_mode = False
            existing_controlnet_widget.controlnet.conditioning_scale = round(
                dialog.conditioning_scale, 2
            )
            existing_controlnet_widget.controlnet.guidance_start = (
                dialog.control_guidance_start
            )
            existing_controlnet_widget.controlnet.guidance_end = (
                dialog.control_guidance_end
            )

            annotator_image = ImageProcessor()
            annotator_image.set_qimage(
                dialog.annotator_widget.image_editor.get_painted_image()
            )
            existing_controlnet_widget.controlnet.annotator_image = (
                annotator_image.get_pillow_image()
            )
        else:
            if controlnet_model is not None:
                controlnet = ControlNetDataObject(
                    controlnet_id=self.controlnet_id_counter,
                    name=dialog.controlnet_combo.currentText(),
                    model_path=os.path.join(
                        self.directories.models_controlnets,
                        controlnet_model,
                    ),
                    enabled=True,
                    guess_mode=False,
                    conditioning_scale=round(dialog.conditioning_scale, 2),
                    guidance_start=dialog.control_guidance_start,
                    guidance_end=dialog.control_guidance_end,
                )

                annotator_image = ImageProcessor()
                annotator_image.set_qimage(
                    dialog.annotator_widget.image_editor.get_painted_image()
                )
                controlnet.annotator_image = annotator_image.get_pillow_image()

                controlnet_widget = ControlNetAddedItem(controlnet)
                controlnet_widget.remove_clicked.connect(
                    lambda lw=controlnet_widget: self.on_remove_controlnet(lw)
                )
                controlnet_widget.enabled.connect(
                    lambda lw=controlnet_widget: self.on_controlnet_enabled(lw)
                )
                self.controlnets_layout.addWidget(controlnet_widget)
                self.controlnets.append(controlnet_widget)
                self.image_generation_data.add_controlnet(controlnet)
                dialog.controlnet_id = self.controlnet_id_counter

                self.controlnet_id_counter += 1

    def clear_controlnets(self):
        self.controlnets = []

        while self.controlnets_layout.count():
            item = self.controlnets_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def on_remove_controlnet(self, controlnet_widget: ControlNetAddedItem):
        index = self.controlnets_layout.indexOf(controlnet_widget)
        if index != -1:
            self.controlnets_layout.takeAt(index)
            controlnet_widget.deleteLater()
            self.image_generation_data.remove_controlnet(
                self.controlnets[index].controlnet
            )
            del self.controlnets[index]

    def on_controlnet_enabled(self, controlnet_widget: ControlNetAddedItem):
        self.image_generation_data.change_controlnet_enabled(
            controlnet_widget.controlnet, controlnet_widget.enabled_checkbox.isChecked()
        )
