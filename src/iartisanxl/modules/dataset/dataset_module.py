import os


import torch

from transformers import CLIPTokenizer
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QProgressBar
from PyQt6.QtCore import Qt

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.dataset_items_view import DatasetItemsView
from iartisanxl.modules.common.image_cropper_widget import ImageCropperWidget
from iartisanxl.modules.common.prompt_input import PromptInput
from iartisanxl.threads.generate_captions_thread import GenerateCaptionsThread


class DatasetModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_dir = None

        self.tokenizer = CLIPTokenizer.from_pretrained("./configs/clip-vit-large-patch14")
        self.max_tokens = self.tokenizer.model_max_length - 2

        self.current_image_path = None
        self.current_caption_path = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generate_captions_thread = None

        self.init_ui()

    def init_ui(self):
        super().init_ui()
        main_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.select_dataset_button = QPushButton("Select dataset")
        self.select_dataset_button.clicked.connect(self.on_click_select_dataset)
        button_layout.addWidget(self.select_dataset_button)
        self.mass_caption_button = QPushButton("Mass caption")
        self.mass_caption_button.clicked.connect(self.on_mass_caption)
        button_layout.addWidget(self.mass_caption_button)
        self.ai_mass_caption_button = QPushButton("AI mass caption")
        self.ai_mass_caption_button.clicked.connect(self.on_ai_mass_caption)
        button_layout.addWidget(self.ai_mass_caption_button)

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
        self.ai_caption_button = QPushButton("AI caption")
        self.ai_caption_button.clicked.connect(self.on_ai_caption)
        image_buttons_layout.addWidget(self.ai_caption_button)
        self.save_caption_button = QPushButton("Save")
        self.save_caption_button.clicked.connect(self.on_image_save)
        image_buttons_layout.addWidget(self.save_caption_button)
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

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

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

        if len(self.dataset_dir) > 0:
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

    def disable_ui(self):
        self.select_dataset_button.setDisabled(True)
        self.mass_caption_button.setDisabled(True)
        self.ai_mass_caption_button.setDisabled(True)
        self.ai_caption_button.setDisabled(True)
        self.save_caption_button.setDisabled(True)
        self.image_caption_edit.setDisabled(True)

    def enable_ui(self):
        self.select_dataset_button.setDisabled(False)
        self.mass_caption_button.setDisabled(False)
        self.ai_mass_caption_button.setDisabled(False)
        self.ai_caption_button.setDisabled(False)
        self.save_caption_button.setDisabled(False)
        self.image_caption_edit.setDisabled(False)

    def on_ai_caption(self):
        self.disable_ui()

        if self.generate_captions_thread is None:
            self.generate_captions_thread = GenerateCaptionsThread(self.device)
            self.generate_captions_thread.status_update.connect(self.update_status_bar)
            self.generate_captions_thread.caption_done.connect(self.on_ai_caption_done)
        else:
            self.generate_captions_thread.caption_done.disconnect(self.generate_item_ai_caption_done)
            self.generate_captions_thread.caption_done.connect(self.on_ai_caption_done)

        text = self.image_caption_edit.toPlainText()
        pixmap = self.dataset_items_view.current_item.pixmap

        self.generate_captions_thread.text = text
        self.generate_captions_thread.pixmap = pixmap

        self.generate_captions_thread.start()

    def on_ai_caption_done(self, text):
        self.image_caption_edit.setPlainText(text)
        self.enable_ui()
        self.update_status_bar("Ready")

    def on_mass_caption(self):
        captions = self.image_caption_edit.toPlainText()

        if self.dataset_dir is None or len(self.dataset_dir) == 0:
            self.show_snackbar("You must select a dataset first.")
            return

        if len(captions) == 0:
            self.show_snackbar("You must enter a caption first.")
            return

        self.progress_bar.setMaximum(self.dataset_items_view.item_count)
        current_item = self.dataset_items_view.get_first_item()
        self.image_caption_edit.setPlainText(captions)
        process_count = 0

        while True:
            path = current_item.path

            captions_file = os.path.splitext(path)[0] + ".txt"
            captions_path = os.path.join(self.dataset_dir, captions_file)
            with open(captions_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(captions)

            process_count += 1
            self.progress_bar.setValue(process_count)

            current_item = self.dataset_items_view.get_next_item()

            if current_item is None:
                break

    def on_ai_mass_caption(self):
        self.disable_ui()

        if self.generate_captions_thread is None:
            self.generate_captions_thread = GenerateCaptionsThread(self.device)
            self.generate_captions_thread.status_update.connect(self.update_status_bar)
            self.generate_captions_thread.caption_done.connect(self.generate_item_ai_caption_done)
        else:
            self.generate_captions_thread.caption_done.disconnect(self.on_ai_caption_done)
            self.generate_captions_thread.caption_done.connect(self.generate_item_ai_caption_done)

        text = self.image_caption_edit.toPlainText()

        self.progress_bar.setMaximum(self.dataset_items_view.item_count)
        self.dataset_items_view.get_first_item()
        self.update_status_bar("Generating captions...")
        self.generate_item_ai_caption(text)

    def generate_item_ai_caption(self, text):
        item = self.dataset_items_view.current_item
        self.image_caption_edit.setPlainText(text)

        self.generate_captions_thread.text = text
        self.generate_captions_thread.pixmap = item.pixmap

        self.generate_captions_thread.start()

    def generate_item_ai_caption_done(self, captions):
        text = self.generate_captions_thread.text
        process_count = self.progress_bar.value()

        item = self.dataset_items_view.current_item
        self.image_caption_edit.setPlainText(captions)

        path = item.path
        captions_file = os.path.splitext(path)[0] + ".txt"
        captions_path = os.path.join(self.dataset_dir, captions_file)
        with open(captions_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(captions)

        process_count += 1
        self.progress_bar.setValue(process_count)

        next_item = self.dataset_items_view.get_next_item()

        if next_item is not None:
            self.generate_item_ai_caption(text)
        else:
            self.update_status_bar("Ready")

    def closeEvent(self, event):
        self.generate_captions_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        super().closeEvent(event)
