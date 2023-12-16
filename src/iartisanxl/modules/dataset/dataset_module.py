import os
import io

import torch
from PIL import Image
from transformers import CLIPTokenizer, BlipProcessor, BlipForConditionalGeneration
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel
from PyQt6.QtCore import Qt, QBuffer

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.dataset_items_view import DatasetItemsView
from iartisanxl.modules.common.image_cropper_widget import ImageCropperWidget
from iartisanxl.modules.common.prompt_input import PromptInput


class DatasetModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_dir = None

        self.tokenizer = CLIPTokenizer.from_pretrained("./configs/clip-vit-large-patch14")
        self.max_tokens = self.tokenizer.model_max_length - 2

        self.current_image_path = None
        self.current_caption_path = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

        self.init_ui()

    def init_ui(self):
        super().init_ui()
        main_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        select_dataset_button = QPushButton("Select dataset")
        select_dataset_button.clicked.connect(self.on_click_select_dataset)
        button_layout.addWidget(select_dataset_button)
        mass_caption_button = QPushButton("Mass caption")
        mass_caption_button.clicked.connect(self.on_mass_caption)
        button_layout.addWidget(mass_caption_button)
        ai_mass_caption_button = QPushButton("AI mass caption")
        button_layout.addWidget(ai_mass_caption_button)

        middle_layout = QHBoxLayout()

        dataset_view_layout = QVBoxLayout()
        dataset_view_layout.setContentsMargins(0, 0, 0, 0)
        dataset_view_layout.setSpacing(0)
        self.dataset_items_count_label = QLabel("0/0")
        dataset_view_layout.addWidget(self.dataset_items_count_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.dataset_items_view = DatasetItemsView()
        self.dataset_items_view.item_selected.connect(self.on_item_selected)
        self.dataset_items_view.finished_loading.connect(self.on_finished_loading_dataset)
        self.dataset_items_view.items_changed.connect(self.set_item)
        dataset_view_layout.addWidget(self.dataset_items_view)
        middle_layout.addLayout(dataset_view_layout)

        image_layout = QVBoxLayout()
        self.image_cropper_widget = ImageCropperWidget()
        image_layout.addWidget(self.image_cropper_widget)
        self.image_caption_edit = PromptInput(True, 0, self.max_tokens, title="Caption")
        self.image_caption_edit.text_changed.connect(self.on_caption_changed)
        image_layout.addWidget(self.image_caption_edit)
        image_buttons_layout = QHBoxLayout()
        ai_caption_button = QPushButton("AI caption")
        ai_caption_button.clicked.connect(self.on_ai_caption)
        image_buttons_layout.addWidget(ai_caption_button)
        save_caption_button = QPushButton("Save")
        save_caption_button.clicked.connect(self.on_image_save)
        image_buttons_layout.addWidget(save_caption_button)
        image_layout.addLayout(image_buttons_layout)

        image_layout.setStretch(0, 1)
        image_layout.setStretch(1, 0)
        image_layout.setStretch(2, 0)

        middle_layout.addLayout(image_layout)

        middle_layout.setStretch(0, 2)
        middle_layout.setStretch(1, 3)

        main_layout.addLayout(button_layout)
        self.dataset_title = QLabel("")
        self.dataset_title.setStyleSheet("font: bold 18px Arial")
        main_layout.addWidget(self.dataset_title, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(middle_layout)

        self.setLayout(main_layout)

    def on_click_select_dataset(self):
        dialog = QFileDialog()
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        self.dataset_dir = dialog.getExistingDirectory(None, "Select directory", self.directories.datasets)

        dataset_name = os.path.basename(self.dataset_dir)
        self.dataset_title.setText(f"Dataset: {dataset_name}")

        self.dataset_items_view.load_items(self.dataset_dir)

    def on_finished_loading_dataset(self):
        if self.dataset_items_view.selected_path is not None:
            self.set_item()

    def on_item_selected(self):
        self.set_item()

    def set_item(self):
        if self.dataset_items_view.current_item is not None:
            self.current_image_path = self.dataset_items_view.selected_path
            self.image_cropper_widget.set_image(self.current_image_path)

            self.current_caption_path = os.path.splitext(self.current_image_path)[0] + ".txt"

            if os.path.isfile(self.current_caption_path):
                with open(self.current_caption_path, "r", encoding="utf-8") as caption_file:
                    caption_text = caption_file.read()
                    self.image_caption_edit.setPlainText(caption_text)
            else:
                self.image_caption_edit.clear()

            self.dataset_items_count_label.setText(f"{self.dataset_items_view.current_item_index + 1}/{self.dataset_items_view.item_count}")
        else:
            self.dataset_items_count_label.setText(f"0/{self.dataset_items_view.item_count}")
            self.image_caption_edit.clear()
            self.image_cropper_widget.clear()

    def on_caption_changed(self):
        text = self.image_caption_edit.toPlainText()

        tokens = self.tokenizer(text).input_ids[1:-1]
        num_tokens = len(tokens)

        self.image_caption_edit.update_token_count(num_tokens)

    def on_image_save(self):
        pixmap = self.image_cropper_widget.get_image()

        if pixmap is not None:
            pixmap.save(self.current_image_path)
            self.dataset_items_view.update_current_item_image(pixmap)
            self.image_cropper_widget.set_pixmap(pixmap)
            self.image_cropper_widget.reset_values()

        captions_text = self.image_caption_edit.toPlainText()

        if len(captions_text) > 0:
            with open(self.current_caption_path, "w", encoding="utf-8") as caption_file:
                caption_file.write(captions_text)

    def on_ai_caption(self):
        if self.model is None:
            self.processor = BlipProcessor.from_pretrained("models/captions/fusecap")
            self.model = BlipForConditionalGeneration.from_pretrained("models/captions/fusecap").to(self.device)

        text = self.image_caption_edit.toPlainText()
        pixmap = self.dataset_items_view.current_item.pixmap
        qimage = pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        qimage.save(buffer, "PNG")

        raw_image = Image.open(io.BytesIO(buffer.data()))
        inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, num_beams=3)
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        self.image_caption_edit.setPlainText(generated_text)

    def on_mass_caption(self):
        captions = self.image_caption_edit.toPlainText()

        print(f"{captions=}")

        if self.dataset_dir is not None and len(captions) > 0:
            if os.path.isdir(self.dataset_dir):
                for file in os.listdir(self.dataset_dir):
                    print(f"{file=}")
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        captions_file = os.path.splitext(file)[0] + ".txt"
                        captions_path = os.path.join(self.dataset_dir, captions_file)

                        with open(captions_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(captions)
