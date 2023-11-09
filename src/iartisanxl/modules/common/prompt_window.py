from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLineEdit,
    QCheckBox,
    QLabel,
    QWidget,
)
from PyQt6.QtCore import pyqtSignal, Qt
from transformers import CLIPTokenizer

from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.modules.common.prompt_input import PromptInput
from iartisanxl.modules.common.generate_button import GenerateButton


class PromptWindow(QFrame):
    generate_signal = pyqtSignal(bool, bool)

    def __init__(
        self, image_generation_data: ImageGenData, module_options: dict, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_generation_data = image_generation_data
        self.module_options = module_options

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "./configs/clip-vit-large-patch14"
        )
        self.max_tokens = self.tokenizer.model_max_length - 2

        self.init_ui()
        self.update_ui()
        self.set_button_generate()

    def init_ui(self):
        main_layout = QHBoxLayout()
        text_layout = QVBoxLayout()

        positive_prompt_layout = QGridLayout()
        self.positive_prompt = PromptInput(True, 0, self.max_tokens)
        self.positive_prompt.text_changed.connect(self.on_prompt_changed)
        positive_prompt_layout.addWidget(self.positive_prompt, 0, 0)

        self.positive_style_prompt = PromptInput(True, 0, self.max_tokens)
        self.positive_style_prompt.text_changed.connect(self.on_prompt_changed)
        positive_prompt_layout.addWidget(self.positive_style_prompt, 0, 1)
        text_layout.addLayout(positive_prompt_layout)

        negative_prompt_layout = QGridLayout()
        self.negative_prompt = PromptInput(False, 0, self.max_tokens)
        self.negative_prompt.text_changed.connect(self.on_prompt_changed)
        negative_prompt_layout.addWidget(self.negative_prompt, 0, 0)
        text_layout.addLayout(negative_prompt_layout)

        self.negative_style_prompt = PromptInput(False, 0, self.max_tokens)
        self.negative_style_prompt.text_changed.connect(self.on_prompt_changed)
        negative_prompt_layout.addWidget(self.negative_style_prompt, 0, 1)

        main_layout.addLayout(text_layout)

        actions_layout = QVBoxLayout()

        seed_widget = QWidget()
        seed_widget.setMaximumWidth(180)
        seed_layout = QVBoxLayout(seed_widget)
        seed_horizontal_layout = QHBoxLayout()
        seed_label = QLabel("Seed:")
        seed_horizontal_layout.addWidget(seed_label)
        self.seed_text = QLineEdit()
        self.seed_text.setDisabled(True)
        seed_horizontal_layout.addWidget(self.seed_text)
        seed_layout.addLayout(seed_horizontal_layout)
        random_checkbox_layout = QHBoxLayout()
        self.random_checkbox = QCheckBox("Randomize seed")
        self.random_checkbox.setChecked(True)
        self.random_checkbox.clicked.connect(self.randomize_clicked)
        random_checkbox_layout.addWidget(self.random_checkbox)
        random_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        seed_layout.addLayout(random_checkbox_layout)
        actions_layout.addWidget(seed_widget)

        self.generate_button = GenerateButton()
        self.generate_button.setMaximumWidth(180)
        self.generate_button.clicked.connect(self.generate)
        actions_layout.addWidget(self.generate_button)
        actions_layout.setStretch(0, 1)
        actions_layout.setStretch(1, 1)
        main_layout.addLayout(actions_layout)

        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 2)

        self.setLayout(main_layout)

    def generate(self):
        try:
            seed = int(self.seed_text.text())
        except ValueError:
            seed = 0

        values = {"seed": seed}

        if (
            self.positive_style_prompt.isVisible()
            and len(self.positive_style_prompt.toPlainText()) > 0
        ):
            values["positive_prompt_clipg"] = self.positive_prompt.toPlainText()
            values["positive_prompt_clipl"] = self.positive_style_prompt.toPlainText()
        else:
            values["positive_prompt_clipg"] = self.positive_prompt.toPlainText()
            values["positive_prompt_clipl"] = ""

        if (
            self.negative_style_prompt.isVisible()
            and len(self.negative_style_prompt.toPlainText()) > 0
        ):
            values["negative_prompt_clipg"] = self.negative_prompt.toPlainText()
            values["negative_prompt_clipl"] = self.negative_style_prompt.toPlainText()
        else:
            values["negative_prompt_clipg"] = self.negative_prompt.toPlainText()
            values["negative_prompt_clipl"] = ""

        self.image_generation_data.update_attributes(values)
        self.generate_signal.emit(
            self.generate_button.auto_save, self.generate_button.continuous_generation
        )

    def update_ui(self):
        self.split_positive_prompt(self.module_options.get("positive_prompt_split"))
        self.positive_prompt.setPlainText(
            self.image_generation_data.positive_prompt_clipg
        )
        self.positive_style_prompt.setPlainText(
            self.image_generation_data.positive_prompt_clipl
        )

        self.split_negative_prompt(self.module_options.get("negative_prompt_split"))
        self.negative_prompt.setPlainText(
            self.image_generation_data.negative_prompt_clipg
        )
        self.negative_style_prompt.setPlainText(
            self.image_generation_data.negative_prompt_clipl
        )

        if self.image_generation_data.seed == 0:
            seed_text = ""
        else:
            seed_text = str(self.image_generation_data.seed)
            self.unblock_seed()

        self.seed_text.setText(seed_text)

    def unblock_seed(self):
        self.random_checkbox.setChecked(False)
        self.seed_text.setDisabled(False)

    def randomize_clicked(self):
        if self.random_checkbox.isChecked():
            self.seed_text.setDisabled(True)
        else:
            self.seed_text.setDisabled(False)

    def split_positive_prompt(self, value):
        if value:
            self.positive_prompt.set_title_text("Positive subject")
            self.positive_style_prompt.set_title_text("Positive style")
            self.positive_style_prompt.setVisible(True)
        else:
            self.positive_prompt.set_title_text("Positive")
            self.positive_style_prompt.set_title_text("Positive")
            self.positive_style_prompt.setVisible(False)

    def split_negative_prompt(self, value):
        if value:
            self.negative_prompt.set_title_text("Negative subject")
            self.negative_style_prompt.set_title_text("Negative style")
            self.negative_style_prompt.setVisible(True)
        else:
            self.negative_prompt.set_title_text("Negative")
            self.negative_style_prompt.set_title_text("Negative")
            self.negative_style_prompt.setVisible(False)

    def set_button_generate(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #23923b, stop: 1 #124e1f);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2cb349, stop: 1 #176427);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #35d457, stop: 1 #1f8834);
            }            
            """
        )
        self.generate_button.setText("Generate")
        self.generate_button.setShortcut("Ctrl+Return")

    def set_button_abort(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #c42b2b, stop: 1 #861e1e);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #db2f2f, stop: 1 #9b2222);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #fa3333, stop: 1 #b62828);
            }            
            """
        )
        self.generate_button.setText("Abort")
        self.generate_button.setShortcut("Ctrl+Return")

    def on_prompt_changed(self):
        prompt = self.sender()
        text = prompt.toPlainText()

        tokens = self.tokenizer(text).input_ids[1:-1]
        num_tokens = len(tokens)

        prompt.update_token_count(num_tokens)
