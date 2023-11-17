import os

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QComboBox,
    QSpacerItem,
)
from PyQt6.QtCore import Qt

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.image_dimensions import ImageDimensionsWidget
from iartisanxl.modules.common.dialogs.model_dialog import ModelDialog
from iartisanxl.generation.vae_data_object import VaeDataObject
from iartisanxl.generation.generation_data_object import ImageGenData


class GenerationPanel(BasePanel):
    def __init__(
        self,
        schedulers,
        module_options: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedulers = schedulers
        self.module_options = module_options

        self.vaes = []
        if self.directories.vaes and os.path.isdir(self.directories.vaes):
            self.vaes = next(os.walk(self.directories.vaes))[1]

        self.model_dialog = None
        self.init_ui()
        self.update_ui(self.image_generation_data)

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.image_dimensions = ImageDimensionsWidget(self.image_generation_data)
        main_layout.addWidget(self.image_dimensions)

        step_guidance_layout = QGridLayout()
        steps_label = QLabel("Steps:")
        step_guidance_layout.addWidget(steps_label, 0, 0)

        self.steps_slider = QSlider()
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setSingleStep(1)
        self.steps_slider.setOrientation(Qt.Orientation.Horizontal)
        self.steps_slider.valueChanged.connect(self.on_steps_value_changed)
        step_guidance_layout.addWidget(self.steps_slider, 0, 1)

        self.label_steps_value = QLabel()
        step_guidance_layout.addWidget(self.label_steps_value, 0, 2)

        guidance_label = QLabel("Guidance")
        guidance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_guidance_layout.addWidget(guidance_label, 1, 0)

        self.guidance_dial = QSlider()
        self.guidance_dial.setRange(10, 200)
        self.guidance_dial.setSingleStep(1)
        self.guidance_dial.setOrientation(Qt.Orientation.Horizontal)
        self.guidance_dial.setValue(int(self.image_generation_data.guidance * 10))
        self.guidance_dial.valueChanged.connect(self.on_dial_value_changed)
        step_guidance_layout.addWidget(self.guidance_dial, 1, 1)

        self.guidance_value_label = QLabel(f"{self.image_generation_data.guidance}")
        self.guidance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_guidance_layout.addWidget(self.guidance_value_label, 1, 2)
        main_layout.addLayout(step_guidance_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 8))

        clip_skip_layout = QGridLayout()
        clip_skip_label = QLabel("Clip skip")
        clip_skip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clip_skip_layout.addWidget(clip_skip_label, 0, 0)
        self.clip_slip_slider = QSlider()
        self.clip_slip_slider.setRange(0, 11)
        self.clip_slip_slider.setSingleStep(1)
        self.clip_slip_slider.setOrientation(Qt.Orientation.Horizontal)
        self.clip_slip_slider.valueChanged.connect(self.on_clip_skip_value_changed)
        clip_skip_layout.addWidget(self.clip_slip_slider, 0, 1)

        self.clip_skip_value_label = QLabel(f"{self.image_generation_data.clip_skip}")
        self.clip_skip_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clip_skip_layout.addWidget(self.clip_skip_value_label, 0, 2)
        main_layout.addLayout(clip_skip_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 20))

        select_base_model_button = QPushButton("Load model")
        select_base_model_button.clicked.connect(self.open_model_dialog)
        main_layout.addWidget(select_base_model_button)

        base_model_layout = QHBoxLayout()
        base_model_label = QLabel("Model: ")
        base_model_layout.addWidget(base_model_label)

        model_name = "no model selected"
        if (
            self.image_generation_data.model is not None
            and self.image_generation_data.model.name is not None
        ):
            model_name = self.image_generation_data.model.name
        self.selected_base_model_label = QLabel(model_name)
        base_model_layout.addWidget(self.selected_base_model_label)
        main_layout.addLayout(base_model_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 8))

        vae_label = QLabel("Vae")
        vae_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(vae_label)

        self.vae_combobox = QComboBox()
        self.vae_combobox.addItem("Model default", "")
        self.vae_combobox.addItem("Vae FP16 Fixed", "./models/vae-fp16")
        if self.vaes:
            for vae in self.vaes:
                self.vae_combobox.addItem(vae, self.directories.vaes + "/" + vae)
        self.vae_combobox.currentIndexChanged.connect(self.on_vae_selected)
        main_layout.addWidget(self.vae_combobox)

        main_layout.addSpacerItem(QSpacerItem(0, 8))

        base_scheduler_label = QLabel("Scheduler")
        base_scheduler_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(base_scheduler_label)

        self.scheduler_combobox = QComboBox()
        self.scheduler_combobox.addItems(
            [scheduler.name for scheduler in self.schedulers]
        )
        self.scheduler_combobox.setCurrentIndex(
            self.image_generation_data.base_scheduler
        )
        self.scheduler_combobox.currentIndexChanged.connect(
            self.base_scheduler_selected
        )
        main_layout.addWidget(self.scheduler_combobox)

        main_layout.addSpacerItem(QSpacerItem(0, 8))

        split_prompts_layout = QHBoxLayout()
        self.split_positive_prompt = QCheckBox("Split positive prompt")
        self.split_positive_prompt.clicked.connect(self.on_split_positive_prompt)
        split_prompts_layout.addWidget(self.split_positive_prompt)

        self.split_negative_prompt = QCheckBox("Split negative prompt")
        self.split_negative_prompt.clicked.connect(self.on_split_negative_prompt)
        split_prompts_layout.addWidget(
            self.split_negative_prompt, alignment=Qt.AlignmentFlag.AlignRight
        )
        main_layout.addLayout(split_prompts_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_steps_value_changed(self, value):
        self.image_generation_data.steps = value
        self.label_steps_value.setText(f"{value}")

    def on_dial_value_changed(self, value):
        self.image_generation_data.guidance = value / 10.0
        self.guidance_value_label.setText(f"{self.image_generation_data.guidance}")

    def on_clip_skip_value_changed(self, value):
        self.image_generation_data.clip_skip = value
        self.clip_skip_value_label.setText(f"{value}")

    def update_ui(self, image_generation_data: ImageGenData):
        super().update_ui(image_generation_data)

        if len(self.image_generation_data.positive_prompt_clipl) > 0:
            self.split_positive_prompt.setChecked(True)
            self.module_options["positive_prompt_split"] = True
            self.prompt_window.split_positive_prompt(True)
        else:
            self.split_positive_prompt.setChecked(
                self.module_options.get("positive_prompt_split")
            )

        if len(self.image_generation_data.negative_prompt_clipl) > 0:
            self.split_negative_prompt.setChecked(True)
            self.module_options["negative_prompt_split"] = True
            self.prompt_window.split_negative_prompt(True)
        else:
            self.split_negative_prompt.setChecked(
                self.module_options.get("negative_prompt_split")
            )

        try:
            self.steps_slider.valueChanged.disconnect(self.on_steps_value_changed)
            self.guidance_dial.valueChanged.disconnect(self.on_dial_value_changed)
        except TypeError:
            pass

        self.steps_slider.setValue(self.image_generation_data.steps)
        self.label_steps_value.setText(f"{self.image_generation_data.steps}")

        self.guidance_dial.setValue(int(self.image_generation_data.guidance * 10))
        self.guidance_value_label.setText(f"{self.image_generation_data.guidance}")

        self.steps_slider.valueChanged.connect(self.on_steps_value_changed)
        self.guidance_dial.valueChanged.connect(self.on_dial_value_changed)

        self.scheduler_combobox.setCurrentIndex(
            self.image_generation_data.base_scheduler
        )

        if self.image_generation_data.model is not None:
            version_string = ""
            if (
                self.image_generation_data.model.version is not None
                and len(self.image_generation_data.model.version) > 0
            ):
                version_string = f"v{self.image_generation_data.model.version}"
            self.selected_base_model_label.setText(
                f"{self.image_generation_data.model.name} {version_string}"
            )

        if self.image_generation_data.vae is not None:
            self.vae_combobox.setCurrentText(self.image_generation_data.vae.name)

        self.clip_slip_slider.setValue(self.image_generation_data.clip_skip)

        self.image_dimensions.update()

    def base_scheduler_selected(self, index):
        self.image_generation_data.base_scheduler = index

    def open_model_dialog(self):
        self.dialog_opened.emit(ModelDialog, "Models")

    def on_vae_selected(self):
        vae = VaeDataObject(
            name=self.vae_combobox.currentText(), path=self.vae_combobox.currentData()
        )
        self.image_generation_data.vae = vae

    def on_split_positive_prompt(self, value):
        self.module_options["positive_prompt_split"] = value
        self.prompt_window.split_positive_prompt(value)

    def on_split_negative_prompt(self, value):
        self.module_options["negative_prompt_split"] = value
        self.prompt_window.split_negative_prompt(value)
