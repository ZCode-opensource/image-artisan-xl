import os
import shutil
import json

import torch

from transformers import CLIPTokenizer
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QProgressBar, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PIL import Image

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.dataset_items_view import DatasetItemsView
from iartisanxl.modules.dataset.image_cropper_widget import ImageCropperWidget
from iartisanxl.modules.common.prompt_input import PromptInput
from iartisanxl.threads.generate_captions_thread import GenerateCaptionsThread
from iartisanxl.threads.dataset_item_saver_thread import DatasetItemSaverThread
from iartisanxl.threads.image_upscale_thread import ImageUpscaleThread


class DatasetModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = CLIPTokenizer.from_pretrained("./configs/clip-vit-large-patch14")
        self.max_tokens = self.tokenizer.model_max_length - 2

        self.dataset_dir = None
        self.originals_dir = None
        self.current_image_path = None
        self.current_caption_path = None
        self.thumb_width = 73
        self.thumb_height = 73
        self.image_loaded = False
        self.image_upscaled = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generate_captions_thread = None
        self.dataset_image_saver_thread = None
        self.image_upscale_thread = None

        self.init_ui()

        self.fit_button.clicked.connect(self.image_cropper_widget.image_cropper.fit_image)
        self.reset_button.clicked.connect(self.image_cropper_widget.reset_values)

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
        self.dataset_items_view = DatasetItemsView(self.thumb_width, self.thumb_height)
        self.dataset_items_view.item_selected.connect(self.on_item_selected)
        self.dataset_items_view.finished_loading.connect(self.on_finished_loading_dataset)
        self.dataset_items_view.items_changed.connect(self.set_item)
        self.dataset_items_view.error.connect(self.show_snackbar)
        dataset_view_layout.addWidget(self.dataset_items_view)
        middle_layout.addLayout(dataset_view_layout)

        image_layout = QVBoxLayout()
        image_top_layout = QHBoxLayout()
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItem("Square 1024x1024", "square")
        self.aspect_combo.addItem("Portrait 896x1152", "portrait")
        self.aspect_combo.addItem("Landscape 1152x896", "landscape")
        self.aspect_combo.addItem("HD Video 1344x768", "hd_video")
        self.aspect_combo.addItem("Cinema 1536x704", "cinema")
        self.aspect_combo.currentIndexChanged.connect(self.on_aspect_change)
        image_top_layout.addWidget(self.aspect_combo)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.on_load_image)
        image_top_layout.addWidget(self.load_button)
        self.fit_button = QPushButton("Fit")
        image_top_layout.addWidget(self.fit_button)
        self.reset_button = QPushButton("Reset")
        image_top_layout.addWidget(self.reset_button)
        self.upscale_button = QPushButton("Upscale")
        self.upscale_button.clicked.connect(self.on_clicked_upscale)
        image_top_layout.addWidget(self.upscale_button)
        image_layout.addLayout(image_top_layout)

        self.image_cropper_widget = ImageCropperWidget()
        self.image_cropper_widget.image_loaded.connect(self.on_image_loaded)
        image_layout.addWidget(self.image_cropper_widget)
        self.image_caption_edit = PromptInput(True, 0, self.max_tokens, title="Caption")
        self.image_caption_edit.text_changed.connect(self.on_caption_changed)
        image_layout.addWidget(self.image_caption_edit)
        image_buttons_layout = QHBoxLayout()
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("red_button")
        image_buttons_layout.addWidget(self.delete_button)
        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.on_new_clicked)
        self.new_button.setObjectName("yellow_button")
        image_buttons_layout.addWidget(self.new_button)
        self.ai_caption_button = QPushButton("AI caption")
        self.ai_caption_button.setObjectName("blue_button")
        self.ai_caption_button.clicked.connect(self.on_ai_caption)
        image_buttons_layout.addWidget(self.ai_caption_button)
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("green_button")
        self.save_button.clicked.connect(self.on_click_save)
        image_buttons_layout.addWidget(self.save_button)
        image_layout.addLayout(image_buttons_layout)

        image_layout.setStretch(0, 0)
        image_layout.setStretch(1, 1)
        image_layout.setStretch(2, 0)
        image_layout.setStretch(3, 0)

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

    def on_aspect_change(self):
        aspect_index = self.aspect_combo.currentIndex()
        self.image_cropper_widget.set_aspect(aspect_index)

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
            self.image_cropper_widget.clear_image()
            dataset_name = os.path.basename(self.dataset_dir)
            self.dataset_title.setText(f"Dataset: {dataset_name}")

            originals_dir = os.path.join(self.dataset_dir, ".originals/")

            if not os.path.isdir(originals_dir):
                os.makedirs(originals_dir)

            self.originals_dir = originals_dir

            self.dataset_items_view.load_items(self.dataset_dir)
            self.dataset_items_view.originals_dir = self.originals_dir

    def on_finished_loading_dataset(self):
        if self.dataset_items_view.selected_path is not None:
            self.set_item()

    def on_item_selected(self):
        self.save_button.setText("Update")
        self.set_item()

    def set_item(self):
        if self.dataset_items_view.current_item is not None:
            self.current_image_path = self.dataset_items_view.selected_path
            filename = os.path.basename(self.current_image_path)
            name = os.path.splitext(filename)[0]

            # check if it has a original image with params
            original_path = os.path.join(self.originals_dir, filename)
            if os.path.isfile(original_path):
                image_params = None

                json_path = os.path.join(self.originals_dir, f"{name}.json")
                if os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as json_file:
                        image_params = json.load(json_file)

                if image_params is not None:
                    self.image_cropper_widget.set_image(original_path)
                    self.image_cropper_widget.update_image_params(
                        image_params["pos_x"], image_params["pos_y"], image_params["scale"], image_params["angle"]
                    )
                    self.aspect_combo.setCurrentIndex(image_params["aspect_index"])
                    self.on_aspect_change()
            else:
                self.image_cropper_widget.set_image(self.current_image_path)

            self.current_caption_path = os.path.join(self.dataset_dir, f"{name}.txt")

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
            self.image_cropper_widget.clear_image()
            self.on_new_clicked()

    def on_caption_changed(self):
        text = self.image_caption_edit.toPlainText()

        tokens = self.tokenizer(text).input_ids[1:-1]
        num_tokens = len(tokens)

        self.image_caption_edit.update_token_count(num_tokens)

    def on_click_save(self):
        if self.dataset_dir is None:
            self.show_snackbar("You must select a dataset directory first.")
            return

        pixmap = self.image_cropper_widget.get_image()
        has_original = False

        if pixmap is not None:
            if self.image_loaded:
                image_path = self.image_cropper_widget.image_path
                filename = (os.path.basename(image_path)).lower()
                name = os.path.splitext(filename)[0]
                new_filename = f"{name}.jpg"

                original_path = os.path.join(self.originals_dir, new_filename)
                new_image_path = os.path.join(self.dataset_dir, new_filename)

                if os.path.isfile(original_path) or os.path.isfile(new_image_path):
                    self.show_snackbar("File already exists in dataset.")
                    return

                with Image.open(image_path) as img:
                    if img.format != "JPEG":
                        img.save(original_path, "JPEG")
                        filename = new_filename
                    else:
                        shutil.copy2(image_path, original_path)

                has_original = True
            else:
                filename = os.path.basename(self.current_image_path)
                original_path = os.path.join(self.originals_dir, filename)

                # if the image was upscaled, append a suffix and save
                if self.image_upscaled:
                    os.remove(original_path)

                    name = os.path.splitext(filename)[0]
                    upscaled_filename = f"{name}_upscaled.jpg"
                    original_path = os.path.join(self.originals_dir, upscaled_filename)

                    original_pixmap = self.image_cropper_widget.image_cropper.original_pixmap
                    original_pixmap.save(original_path, "JPEG", 100)

                if os.path.isfile(original_path):
                    has_original = True

            self.dataset_image_saver_thread = DatasetItemSaverThread(
                self.dataset_dir,
                self.originals_dir,
                filename,
                self.aspect_combo.currentIndex(),
                self.thumb_width,
                self.thumb_height,
                self.image_cropper_widget.image_cropper,
                has_original=has_original,
                upscaled=self.image_upscaled,
            )
            self.dataset_image_saver_thread.image_saved.connect(self.image_saved)
            self.dataset_image_saver_thread.start()

    def image_saved(self, image_path: str, thumbnail: QPixmap):
        if self.current_image_path is None:
            self.dataset_items_view.add_item(image_path, thumbnail)
            self.current_image_path = image_path
            self.dataset_items_count_label.setText(f"{self.dataset_items_view.item_count}/{self.dataset_items_view.item_count}")
        else:
            if self.image_loaded:
                # if an image was loaded on top of the existing one we need to replace the old one
                remove_file_name = os.path.basename(self.current_image_path)
                remove_original_path = os.path.join(self.originals_dir, remove_file_name)

                os.remove(self.current_image_path)
                os.remove(remove_original_path)

                self.current_image_path = image_path

            self.dataset_items_view.update_current_item(image_path, thumbnail)

        captions_text = self.image_caption_edit.toPlainText()

        if len(captions_text) > 0:
            captions_file = os.path.splitext(image_path)[0] + ".txt"
            self.current_caption_path = captions_file

            with open(self.current_caption_path, "w", encoding="utf-8") as captions_file:
                captions_file.write(captions_text)

        self.image_loaded = False
        self.image_upscaled = False
        self.save_button.setText("Update")

    def disable_ui(self):
        self.select_dataset_button.setDisabled(True)
        self.mass_caption_button.setDisabled(True)
        self.ai_mass_caption_button.setDisabled(True)
        self.delete_button.setDisabled(True)
        self.new_button.setDisabled(True)
        self.ai_caption_button.setDisabled(True)
        self.save_button.setDisabled(True)
        self.image_caption_edit.setDisabled(True)

    def enable_ui(self):
        self.select_dataset_button.setDisabled(False)
        self.mass_caption_button.setDisabled(False)
        self.ai_mass_caption_button.setDisabled(False)
        self.delete_button.setDisabled(False)
        self.new_button.setDisabled(False)
        self.ai_caption_button.setDisabled(False)
        self.save_button.setDisabled(False)
        self.image_caption_edit.setDisabled(False)

    def on_ai_caption(self):
        self.disable_ui()

        if self.generate_captions_thread is None:
            self.generate_captions_thread = GenerateCaptionsThread(self.device)
            self.generate_captions_thread.status_update.connect(self.update_status_bar)
            self.generate_captions_thread.caption_done.connect(self.on_ai_caption_done)
            self.generate_captions_thread.error.connect(self.ai_caption_error)
        else:
            try:
                self.generate_captions_thread.caption_done.disconnect(self.generate_item_ai_caption_done)
            except TypeError:
                pass
            self.generate_captions_thread.caption_done.connect(self.on_ai_caption_done)

        text = self.image_caption_edit.toPlainText()
        pixmap = self.dataset_items_view.current_item.pixmap

        self.generate_captions_thread.text = text
        self.generate_captions_thread.pixmap = pixmap

        self.generate_captions_thread.start()

    def ai_caption_error(self, text):
        self.enable_ui()
        self.show_snackbar(text)
        self.update_status_bar(text)

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
        if self.dataset_dir is not None and len(self.dataset_dir) > 0:
            self.disable_ui()

            if self.generate_captions_thread is None:
                self.generate_captions_thread = GenerateCaptionsThread(self.device)
                self.generate_captions_thread.status_update.connect(self.update_status_bar)
                self.generate_captions_thread.caption_done.connect(self.generate_item_ai_caption_done)
                self.generate_captions_thread.error.connect(self.ai_caption_error)
            else:
                try:
                    self.generate_captions_thread.caption_done.disconnect(self.on_ai_caption_done)
                except TypeError:
                    pass
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

    def on_new_clicked(self):
        self.image_cropper_widget.clear_image()
        self.dataset_items_view.clear_selection()
        self.image_caption_edit.setPlainText("")
        self.current_image_path = None
        self.current_caption_path = None
        self.save_button.setText("Save")

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg ¨.jpeg *.webp)", options=options)
        if image_path:
            self.image_cropper_widget.clear_image()
            self.image_cropper_widget.image_path = image_path
            pixmap = QPixmap(self.image_cropper_widget.image_path)
            self.image_cropper_widget.image_cropper.set_pixmap(pixmap)
            self.image_loaded = True

    def on_image_loaded(self):
        self.image_loaded = True

    def on_clicked_upscale(self):
        if self.dataset_dir is None:
            self.show_snackbar("You must select a dataset directory first.")
            return

        if self.current_image_path is None:
            self.show_snackbar("You first need to save the image.")
            return

        file_name = os.path.basename(self.current_image_path)
        original_path = os.path.join(self.originals_dir, file_name)

        if not os.path.isfile(original_path):
            self.show_snackbar("You can only upscale the original image and this file doesn't have it.")
            return

        model_path = os.path.join(self.directories.models_upscalers, "4x-UltraSharp.pth")
        self.image_upscale_thread = ImageUpscaleThread(self.device, original_path, model_path)
        self.image_upscale_thread.status_update.connect(self.update_status_bar)
        self.image_upscale_thread.setup_progress.connect(self.image_upscale_setup)
        self.image_upscale_thread.progress_update.connect(self.image_upscale_update_progress)
        self.image_upscale_thread.upscale_done.connect(self.image_upscale_set_image)
        self.image_upscale_thread.finished.connect(self.image_upscale_finished)
        self.image_upscale_thread.start()

    def image_upscale_setup(self, total_tiles):
        self.progress_bar.setMaximum(total_tiles)
        self.progress_bar.setValue(0)

    def image_upscale_update_progress(self, tile_count):
        self.progress_bar.setValue(tile_count)

    def image_upscale_set_image(self, pixmap: QPixmap):
        self.image_cropper_widget.set_pixmap(pixmap)
        self.image_upscaled = True

    def image_upscale_finished(self):
        self.image_upscale_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
