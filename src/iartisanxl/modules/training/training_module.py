import os
import gc
import math
import json

import attr
import torch
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QFileDialog,
    QLineEdit,
    QProgressBar,
    QTextEdit,
    QComboBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import pyqtgraph as pg

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.image_label import ImageLabel
from iartisanxl.console.console_stream import ConsoleStream
from iartisanxl.windows.log_window import LogWindow
from iartisanxl.threads.dreambooth_lora_train_thread import DreamboothLoraTrainThread
from iartisanxl.modules.training.lora_train_args import LoraTrainArgs


class TrainingModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.training = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_dir = ""
        self.model_path = ""
        self.dataset_path = ""
        self.train_thread = None
        self.epoch_data = []
        self.loss_data = []
        self.lr_data = []
        self.max_steps = 0

        self.total_dataset_images = 0

        self.vaes = []
        if self.directories.models_vaes and os.path.isdir(self.directories.models_vaes):
            self.vaes = next(os.walk(self.directories.models_vaes))[1]

        self.init_ui()
        self.console_stream = ConsoleStream()

        # This module doesn't share the graph and needs the VRAM for training, so we cleanup the graph
        self.node_graph.clean_up()

        self.set_button_train()

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(10)

        top_layout = QHBoxLayout()

        training_type_layout = QHBoxLayout()
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItem("Diffusers - Dreambooth LoRA", "diffusers_dreambooth_lora")
        training_type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(training_type_layout)

        model_layout = QHBoxLayout()
        model_layout.setSpacing(10)
        model_select_button = QPushButton("Select model")
        model_select_button.clicked.connect(lambda: self.select_directory(2))
        model_layout.addWidget(model_select_button)
        self.model_path_label = QLabel()
        model_layout.addWidget(self.model_path_label)
        model_layout.setStretch(0, 2)
        model_layout.setStretch(1, 8)
        top_layout.addLayout(model_layout)

        vae_layout = QHBoxLayout()
        self.vae_combobox = QComboBox()
        self.vae_combobox.addItem("Model Vae", "")
        if self.vaes:
            for vae in self.vaes:
                self.vae_combobox.addItem(vae, os.path.join(self.directories.models_vaes, vae))
        vae_layout.addWidget(self.vae_combobox)
        top_layout.addLayout(vae_layout)

        top_layout.setStretch(0, 0)
        top_layout.setStretch(1, 1)
        top_layout.setStretch(2, 0)

        parameters_layout = QGridLayout()
        parameters_layout.setSpacing(10)

        rank_label = QLabel("Rank:")
        parameters_layout.addWidget(rank_label, 0, 0)
        self.rank_text_edit = QLineEdit()
        self.rank_text_edit.setText("32")
        parameters_layout.addWidget(self.rank_text_edit, 0, 1)

        save_epochs_label = QLabel("Save # epochs:")
        save_epochs_label.setToolTip("Save a model and optimizer state every every number of epochs")
        parameters_layout.addWidget(save_epochs_label, 0, 2)
        self.save_epochs_text_edit = QLineEdit()
        self.save_epochs_text_edit.setText("10")
        parameters_layout.addWidget(self.save_epochs_text_edit, 0, 3)

        accumulation_steps_label = QLabel("G. Acc. steps:")
        accumulation_steps_label.setToolTip("Gradient accumulation steps")
        parameters_layout.addWidget(accumulation_steps_label, 0, 4)
        self.accumulation_steps_text_edit = QLineEdit()
        self.accumulation_steps_text_edit.setText("1")
        parameters_layout.addWidget(self.accumulation_steps_text_edit, 0, 5)

        batch_size_label = QLabel("Batch size:")
        parameters_layout.addWidget(batch_size_label, 0, 6)
        self.batch_size_text_edit = QLineEdit()
        self.batch_size_text_edit.setText("1")
        parameters_layout.addWidget(self.batch_size_text_edit, 0, 7)

        epochs_label = QLabel("Epochs:")
        parameters_layout.addWidget(epochs_label, 0, 8)
        self.epochs_text_edit = QLineEdit()
        self.epochs_text_edit.setText("120")
        parameters_layout.addWidget(self.epochs_text_edit, 0, 9)

        self.scheduler_label = QLabel("LR Scheduler:")
        parameters_layout.addWidget(self.scheduler_label, 0, 10)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItem("Constant", "constant")
        self.lr_scheduler_combo.addItem("Cosine", "cosine")
        self.lr_scheduler_combo.addItem("Linear", "linear")
        self.lr_scheduler_combo.addItem("Constant with warmup", "constant_with_warmup")
        self.lr_scheduler_combo.addItem("Cosine with restarts", "cosine_with_restarts")
        self.lr_scheduler_combo.addItem("Polynomial", "polynomial")
        parameters_layout.addWidget(self.lr_scheduler_combo, 0, 11)

        workers_label = QLabel("Workers:")
        parameters_layout.addWidget(workers_label, 1, 0)
        self.workers_text_edit = QLineEdit()
        self.workers_text_edit.setText("8")
        parameters_layout.addWidget(self.workers_text_edit, 1, 1)

        learning_rate_label = QLabel("Learning rate:")
        parameters_layout.addWidget(learning_rate_label, 1, 2)
        self.learning_rate_text_edit = QLineEdit()
        self.learning_rate_text_edit.setText("1e-4")
        parameters_layout.addWidget(self.learning_rate_text_edit, 1, 3)

        text_encoder_learning_rate_label = QLabel("Text encoder LR:")
        text_encoder_learning_rate_label.setToolTip("Text encoder learning rate")
        parameters_layout.addWidget(text_encoder_learning_rate_label, 1, 4)
        self.text_encoder_learning_rate_text_edit = QLineEdit()
        self.text_encoder_learning_rate_text_edit.setText("1e-5")
        parameters_layout.addWidget(self.text_encoder_learning_rate_text_edit, 1, 5)

        self.optimizer_label = QLabel("Optimizer:")
        parameters_layout.addWidget(self.optimizer_label, 1, 6)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("AdamW8bit ", "adamw8bit")
        self.optimizer_combo.addItem("AdamW", "adamw")
        self.optimizer_combo.addItem("Prodigy ", "prodigy")
        parameters_layout.addWidget(self.optimizer_combo, 1, 7)

        snr_label = QLabel("SNR Gamma:")
        parameters_layout.addWidget(snr_label, 1, 8)
        self.snr_text_edit = QLineEdit()
        self.snr_text_edit.setText("5")
        parameters_layout.addWidget(self.snr_text_edit, 1, 9)

        warmup_steps_label = QLabel("Warmup steps:")
        parameters_layout.addWidget(warmup_steps_label, 1, 10)
        self.warmup_steps_text_edit = QLineEdit()
        self.warmup_steps_text_edit.setText("0")
        parameters_layout.addWidget(self.warmup_steps_text_edit, 1, 11)

        middle_layout = QHBoxLayout()

        info_layout = QVBoxLayout()
        loss_layout = QVBoxLayout()
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_layout.setSpacing(0)
        loss_graph_label = QLabel("Loss:")
        loss_layout.addWidget(loss_graph_label)
        self.loss_graph_widget = pg.PlotWidget()
        loss_layout.addWidget(self.loss_graph_widget)
        info_layout.addLayout(loss_layout)
        learning_rate_layout = QVBoxLayout()
        learning_rate_layout.setContentsMargins(0, 0, 0, 0)
        learning_rate_layout.setSpacing(0)
        learning_rate_graph_label = QLabel("Learning rate:")
        learning_rate_layout.addWidget(learning_rate_graph_label)
        self.learning_rate_graph_widget = pg.PlotWidget()
        learning_rate_layout.addWidget(self.learning_rate_graph_widget)
        info_layout.addLayout(learning_rate_layout)
        self.log_window = LogWindow()
        info_layout.addWidget(self.log_window)

        info_layout.setStretch(0, 4)
        info_layout.setStretch(1, 4)
        info_layout.setStretch(2, 3)
        middle_layout.addLayout(info_layout)

        middle_right_layout = QVBoxLayout()

        dataset_layout = QGridLayout()
        dataset_layout.setContentsMargins(3, 3, 3, 3)
        dataset_layout.setSpacing(5)
        dataset_select_button = QPushButton("Select dataset")
        dataset_select_button.clicked.connect(lambda: self.select_directory(3))
        dataset_layout.addWidget(dataset_select_button, 0, 0)
        self.dataset_path_label = QLabel()
        dataset_layout.addWidget(self.dataset_path_label, 0, 1)
        dataset_image_count_label = QLabel("Total images:")
        dataset_layout.addWidget(dataset_image_count_label, 0, 2)
        self.dataset_count_label_value = QLabel("0")
        dataset_layout.addWidget(self.dataset_count_label_value, 0, 3)
        output_dir_button = QPushButton("Select output directory")
        output_dir_button.clicked.connect(lambda: self.select_directory(1))
        dataset_layout.addWidget(output_dir_button, 1, 0)
        self.output_dir_label = QLabel()
        dataset_layout.addWidget(self.output_dir_label, 1, 1)
        resume_checkpoint_label = QLabel("Resume: ")
        dataset_layout.addWidget(resume_checkpoint_label, 1, 2)
        self.resume_checkpoint_combobox = QComboBox()
        dataset_layout.addWidget(self.resume_checkpoint_combobox, 1, 3)
        middle_right_layout.addLayout(dataset_layout)

        image_progress_layout = QVBoxLayout()
        self.image_label = ImageLabel()
        image_progress_layout.addWidget(self.image_label)

        validation_prompt_label = QLabel("Validation prompt:")
        image_progress_layout.addWidget(validation_prompt_label)

        validation_layout = QHBoxLayout()
        self.validation_prompt_edit = QTextEdit()
        self.validation_prompt_edit.setMaximumHeight(60)
        validation_layout.addWidget(self.validation_prompt_edit)

        right_validation_layout = QGridLayout()
        seed_label = QLabel("Seed:")
        right_validation_layout.addWidget(seed_label, 0, 0)
        self.seed_text_edit = QLineEdit()
        self.seed_text_edit.setText("")
        right_validation_layout.addWidget(self.seed_text_edit, 0, 1)
        self.save_webui_format_checkbox = QCheckBox("Also save webui format")
        right_validation_layout.addWidget(self.save_webui_format_checkbox, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        validation_layout.addLayout(right_validation_layout)
        validation_layout.setStretch(0, 7)
        validation_layout.setStretch(0, 3)
        image_progress_layout.addLayout(validation_layout)

        image_progress_layout.setStretch(0, 1)
        image_progress_layout.setStretch(1, 0)
        image_progress_layout.setStretch(2, 0)
        image_progress_layout.setStretch(3, 0)

        middle_right_layout.addLayout(image_progress_layout)

        epoch_progress_layout = QHBoxLayout()
        self.epoch_progress_label = QLabel("Epoch 0/0")
        epoch_progress_layout.addWidget(self.epoch_progress_label, alignment=Qt.AlignmentFlag.AlignRight)
        self.steps_progress_label = QLabel("Steps 0/0")
        epoch_progress_layout.addWidget(self.steps_progress_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loss_progress_label = QLabel("AVG loss: 0.0")
        epoch_progress_layout.addWidget(self.loss_progress_label, alignment=Qt.AlignmentFlag.AlignLeft)
        image_progress_layout.addLayout(epoch_progress_layout)
        middle_layout.addLayout(middle_right_layout)

        middle_layout.setStretch(0, 2)
        middle_layout.setStretch(1, 3)

        bottom_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        bottom_layout.addWidget(self.progress_bar)

        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton()
        self.train_button.clicked.connect(self.train_clicked)
        buttons_layout.addWidget(self.train_button)
        bottom_layout.addLayout(buttons_layout)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(parameters_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(bottom_layout)

        self.accumulation_steps_text_edit.textChanged.connect(self.calculate_total_steps)
        self.batch_size_text_edit.textChanged.connect(self.calculate_total_steps)
        self.epochs_text_edit.textChanged.connect(self.calculate_total_steps)

        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.train_thread = None
        self.epoch_data = []
        self.loss_data = []
        self.lr_data = []

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        super().closeEvent(event)

    def set_button_train(self):
        self.train_button.setStyleSheet(
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
        self.train_button.setText("Train")
        self.train_button.setShortcut("Ctrl+Return")

    def set_button_abort(self):
        self.train_button.setStyleSheet(
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
        self.train_button.setText("Abort")
        self.train_button.setShortcut("Ctrl+Return")

    def select_directory(self, directory_type):
        dialog = QFileDialog()
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        if directory_type == 1:
            output_dir = dialog.getExistingDirectory(None, "Select directory", self.directories.outputs_loras)
            if len(output_dir) > 0:
                self.output_dir = output_dir
                self.output_dir_label.setText(os.path.basename(self.output_dir))
                self.log_window.success("Output directory selected.")
                config_path = os.path.join(self.output_dir, "train_args.json")
                if os.path.isfile(config_path):
                    self.log_window.add_message("Configuration file found, restoring configuration...")
                    self.load_config_file(config_path)
                    self.load_resume_checkpoints()

        elif directory_type == 2:
            model_path = dialog.getExistingDirectory(None, "Select directory", self.directories.models_diffusers)
            if len(model_path) > 0:
                self.model_path = model_path
                self.model_path_label.setText(os.path.basename(self.model_path))
                self.log_window.success("Model selected.")
        elif directory_type == 3:
            dataset_path = dialog.getExistingDirectory(None, "Select directory", self.directories.datasets)
            if len(dataset_path) > 0:
                self.set_dataset(dataset_path)

    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_path_label.setText(os.path.basename(self.dataset_path))
        image_count = self.check_and_count_dataset(dataset_path)

        if image_count > 0:
            self.dataset_path = dataset_path
            self.total_dataset_images = image_count
            self.dataset_count_label_value.setText(str(image_count))
            self.log_window.success("Dataset selected.")
            self.calculate_total_steps()

    def load_config_file(self, json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        model_path = data.get("model_path", None)
        if model_path is not None:
            self.model_path = model_path
            self.model_path_label.setText(os.path.basename(self.model_path))
            self.log_window.success("Model selected.")

        dataset_path = data.get("dataset_path", None)
        if dataset_path is not None:
            self.set_dataset(dataset_path)

        vae_path = data.get("vae_path", None)
        vae_index = self.vae_combobox.findData(vae_path)
        if vae_index != -1:
            self.vae_combobox.setCurrentIndex(vae_index)

        self.rank_text_edit.setText(str(data.get("rank", 32)))
        self.workers_text_edit.setText(str(data.get("workers", 8)))
        self.save_epochs_text_edit.setText(str(data.get("save_epochs", 1)))
        self.batch_size_text_edit.setText(str(data.get("batch_size", 1)))
        self.accumulation_steps_text_edit.setText(str(data.get("accumulation_steps", 1)))
        self.epochs_text_edit.setText(str(data.get("epochs", 15)))
        self.warmup_steps_text_edit.setText(str(data.get("lr_warmup_steps", 0)))
        self.validation_prompt_edit.setPlainText(data.get("validation_prompt", ""))
        self.save_webui_format_checkbox.setChecked(data.get("save_webui", False))

        seed = data.get("seed", None)
        if seed is not None:
            self.seed_text_edit.setText(str(seed))

        learning_rate = data.get("learning_rate", None)
        learning_rate_str = f"{learning_rate:.0e}" if learning_rate is not None else "1e-04"
        self.learning_rate_text_edit.setText(learning_rate_str)

        text_learning_rate = data.get("text_encoder_learning_rate", None)
        text_learning_rate_str = f"{text_learning_rate:.0e}" if text_learning_rate is not None else "1e-05"
        self.text_encoder_learning_rate_text_edit.setText(text_learning_rate_str)

        optimizer = data.get("optimizer", "adamw8bit")
        optimizer_index = self.optimizer_combo.findData(optimizer)
        if optimizer_index != -1:
            self.optimizer_combo.setCurrentIndex(optimizer_index)

        lr_scheduler = data.get("lr_scheduler", "constant")
        lr_scheduler_index = self.lr_scheduler_combo.findData(lr_scheduler)
        if lr_scheduler_index != -1:
            self.lr_scheduler_combo.setCurrentIndex(lr_scheduler_index)

        self.log_window.success("Finished loading configuration from file.")

    def load_resume_checkpoints(self):
        self.resume_checkpoint_combobox.clear()

        checkpoints = next(os.walk(self.output_dir))[1]
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)

        for checkpoint in sorted_checkpoints:
            self.resume_checkpoint_combobox.addItem(checkpoint)

        self.resume_checkpoint_combobox.addItem("start over")

    def train_clicked(self):
        if self.training:
            if self.train_thread is not None:
                self.train_thread.abort = True
            return

        self.training = True
        self.set_button_abort()

        if self.output_dir is None or len(self.output_dir) == 0:
            self.show_error("You must select an output directory.")
            return

        if self.output_dir is None or len(self.model_path) == 0:
            self.show_error("You must select a model.")
            return

        if self.output_dir is None or len(self.dataset_path) == 0:
            self.show_error("You must select a dataset.")
            return

        train_args = LoraTrainArgs(
            output_dir=self.output_dir,
            model_path=self.model_path,
            vae_path=self.vae_combobox.currentData(),
            rank=int(self.rank_text_edit.text()),
            learning_rate=float(self.learning_rate_text_edit.text()),
            text_encoder_learning_rate=float(self.text_encoder_learning_rate_text_edit.text()),
            dataset_path=self.dataset_path,
            batch_size=int(self.batch_size_text_edit.text()),
            workers=int(self.workers_text_edit.text()),
            accumulation_steps=int(self.accumulation_steps_text_edit.text()),
            epochs=int(self.epochs_text_edit.text()),
            save_epochs=int(self.save_epochs_text_edit.text()),
            seed=int(self.seed_text_edit.text()) if len(self.seed_text_edit.text()) > 0 else None,
            validation_prompt=self.validation_prompt_edit.toPlainText(),
            optimizer=self.optimizer_combo.currentData(),
            lr_scheduler=self.lr_scheduler_combo.currentData(),
            lr_warmup_steps=int(self.warmup_steps_text_edit.text()),
            save_webui=self.save_webui_format_checkbox.isChecked(),
            snr_gamma=self.snr_text_edit.text(),
        )

        if self.resume_checkpoint_combobox.currentText() != "start over":
            train_args.resume_checkpoint = self.resume_checkpoint_combobox.currentText()

        train_args_dict = attr.asdict(train_args)
        json_file_path = os.path.join(self.output_dir, "train_args.json")
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(train_args_dict, json_file, indent=4)

        self.train_thread = DreamboothLoraTrainThread(train_args, self.device)
        self.train_thread.output.connect(self.update_output)
        self.train_thread.output_done.connect(self.update_output_done)
        self.train_thread.warning.connect(self.show_warning)
        self.train_thread.error.connect(self.show_error)
        self.train_thread.ready_to_start.connect(self.on_ready_to_start)
        self.train_thread.update_step.connect(self.update_step)
        self.train_thread.update_epoch.connect(self.update_epoch)
        self.train_thread.training_finished.connect(self.on_training_finished)
        self.train_thread.finished.connect(self.on_thread_finish)
        self.train_thread.aborted.connect(self.on_training_aborted)

        self.epoch_data = []
        self.lr_data = []
        self.loss_data = []
        self.loss_graph_widget.clear()
        self.learning_rate_graph_widget.clear()

        self.train_thread.start()

    def update_output(self, text):
        self.log_window.add_message(text)

    def update_output_done(self):
        self.log_window.append_success("done.")

    def show_error(self, text, show_snackbar=True):
        self.log_window.error(text)
        if show_snackbar:
            self.show_snackbar("Could not start training since there was an error.")
        self.set_button_train()
        self.training = False

    def show_warning(self, text):
        self.log_window.warning(text)

    def on_ready_to_start(self, max_steps):
        self.max_steps = max_steps
        self.progress_bar.setMaximum(self.max_steps)
        self.progress_bar.setValue(0)
        self.epoch_progress_label.setText(f"Epoch 0/{self.epochs_text_edit.text()}")
        self.steps_progress_label.setText(f"Steps 0/{self.max_steps}")

    def update_step(self, step):
        self.progress_bar.setValue(step)
        self.steps_progress_label.setText(f"Steps {step}/{self.max_steps}")

    def update_epoch(self, epoch, learning_rate, loss, image_path):
        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.lr_data.append(learning_rate)
        self.epoch_progress_label.setText(f"Epoch {epoch}/{self.epochs_text_edit.text()}")
        self.loss_progress_label.setText(f"loss: {loss:.4f}")
        self.loss_graph_widget.plot(self.epoch_data, self.loss_data, clear=True)
        self.learning_rate_graph_widget.plot(self.epoch_data, self.lr_data, clear=True)

        if len(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)

    def on_training_finished(self, epoch, loss, image_path):
        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.epoch_progress_label.setText(f"Epoch {epoch}/{self.epochs_text_edit.text()}")
        self.epoch_progress_label.setText(f"Epoch {epoch}/{self.epochs_text_edit.text()}")
        self.loss_progress_label.setText(f"loss: {loss:.4f}")
        self.loss_graph_widget.plot(self.epoch_data, self.loss_data, clear=True)

        if len(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)

        self.progress_bar.setValue(self.max_steps)
        self.training = False
        self.set_button_train()
        self.log_window.success("Training finished.")

    def on_thread_finish(self):
        self.train_thread.accelerator = None
        self.train_thread.unet = None
        self.train_thread.tokenizer_one = None
        self.train_thread.tokenizer_two = None
        self.train_thread.text_encoder_one = None
        self.train_thread.text_encoder_two = None
        self.train_thread.vae = None
        self.train_thread.scheduler = None
        self.train_thread.lora_train_args = None

        self.training = False
        self.train_thread = None
        gc.collect()
        torch.cuda.empty_cache()

    def on_training_aborted(self):
        self.training = False
        self.set_button_train()
        self.update_status_bar("Training aborted.")

    def calculate_total_steps(self):
        batch_size = int(self.batch_size_text_edit.text())
        dataset_length = math.ceil(self.total_dataset_images / batch_size)
        num_update_steps_per_epoch = math.ceil(dataset_length / int(self.accumulation_steps_text_edit.text()))
        max_train_steps = int(self.epochs_text_edit.text()) * num_update_steps_per_epoch
        self.steps_progress_label.setText(f"Steps 0/{max_train_steps}")

    def check_and_count_dataset(self, path):
        image_count = 0
        for entry in os.scandir(path):
            if entry.is_file():
                file = entry.name
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_count += 1
                    txt_file = os.path.join(path, os.path.splitext(file)[0] + ".txt")
                    if not os.path.isfile(txt_file) or os.path.getsize(txt_file) == 0:
                        self.show_snackbar("Not all images have captions, invalid dataset")
                        self.show_error(f"Stopped at {file} because it doesn't have a caption file or is empty.", show_snackbar=False)
                        return 0
        return image_count
