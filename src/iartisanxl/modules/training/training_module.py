import os
import gc

import torch
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QFileDialog, QLineEdit, QProgressBar, QTextEdit, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import pyqtgraph as pg

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.image_label import ImageLabel
from iartisanxl.console.console_stream import ConsoleStream
from iartisanxl.windows.log_window import LogWindow
from iartisanxl.threads.dreambooth_lora_train_thread import DreamboothLoraTrainThread
from iartisanxl.train.lora_train_args import LoraTrainArgs


class TrainingModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.training = False

        self.output_dir = ""
        self.model_path = ""
        self.dataset_path = ""
        self.train_thread = None
        self.epoch_data = []
        self.loss_data = []
        self.lr_data = []
        self.max_steps = 0

        self.vaes = []
        if self.directories.vaes and os.path.isdir(self.directories.vaes):
            self.vaes = next(os.walk(self.directories.vaes))[1]

        self.init_ui()
        self.console_stream = ConsoleStream()
        self.set_button_train()

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(10)

        top_layout = QGridLayout()

        training_type_layout = QHBoxLayout()
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItem("Diffusers - Dreambooth LoRA", "diffusers_dreambooth_lora")
        training_type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(training_type_layout, 0, 0)

        output_layout = QHBoxLayout()
        output_layout.setSpacing(10)
        output_dir_button = QPushButton("Select output directory")
        output_dir_button.clicked.connect(lambda: self.select_directory(1))
        output_layout.addWidget(output_dir_button)
        self.output_dir_label = QLabel()
        output_layout.addWidget(self.output_dir_label)
        top_layout.addLayout(output_layout, 0, 1)

        model_layout = QHBoxLayout()
        model_layout.setSpacing(10)
        model_select_button = QPushButton("Select model")
        model_select_button.clicked.connect(lambda: self.select_directory(2))
        model_layout.addWidget(model_select_button)
        self.model_path_label = QLabel()
        model_layout.addWidget(self.model_path_label)
        top_layout.addLayout(model_layout, 0, 2)

        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(10)
        dataset_select_button = QPushButton("Select dataset")
        dataset_select_button.clicked.connect(lambda: self.select_directory(3))
        dataset_layout.addWidget(dataset_select_button)
        self.dataset_path_label = QLabel()
        dataset_layout.addWidget(self.dataset_path_label)
        top_layout.addLayout(dataset_layout, 0, 3)

        parameters_layout = QGridLayout()
        parameters_layout.setSpacing(10)

        rank_label = QLabel("Rank:")
        parameters_layout.addWidget(rank_label, 0, 0)
        self.rank_text_edit = QLineEdit()
        self.rank_text_edit.setText("8")
        parameters_layout.addWidget(self.rank_text_edit, 0, 1)

        save_epochs_label = QLabel("Save N° epochs:")
        parameters_layout.addWidget(save_epochs_label, 0, 2)
        self.save_epochs_text_edit = QLineEdit()
        self.save_epochs_text_edit.setText("10")
        parameters_layout.addWidget(self.save_epochs_text_edit, 0, 3)

        accumulation_steps_label = QLabel("Gradient accumulation steps:")
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
        self.epochs_text_edit.setText("60")
        parameters_layout.addWidget(self.epochs_text_edit, 0, 9)

        seed_label = QLabel("Seed:")
        parameters_layout.addWidget(seed_label, 0, 10)
        self.seed_text_edit = QLineEdit()
        self.seed_text_edit.setText("")
        parameters_layout.addWidget(self.seed_text_edit, 0, 11)

        vae_label = QLabel("Vae:")
        parameters_layout.addWidget(vae_label, 0, 12)
        self.vae_combobox = QComboBox()
        self.vae_combobox.addItem("Model default", "")
        if self.vaes:
            for vae in self.vaes:
                self.vae_combobox.addItem(vae, self.directories.vaes + "/" + vae)
        parameters_layout.addWidget(self.vae_combobox, 0, 13)

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

        text_encoder_learning_rate_label = QLabel("Text encoder learning rate:")
        parameters_layout.addWidget(text_encoder_learning_rate_label, 1, 4)
        self.text_encoder_learning_rate_text_edit = QLineEdit()
        self.text_encoder_learning_rate_text_edit.setText("1e-4")
        parameters_layout.addWidget(self.text_encoder_learning_rate_text_edit, 1, 5)

        self.optimizer_label = QLabel("Optimizer:")
        parameters_layout.addWidget(self.optimizer_label, 1, 6)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("AdamW8bit ", "adamw8bit")
        self.optimizer_combo.addItem("AdamW", "adamw")
        self.optimizer_combo.addItem("Prodigy ", "prodigy")
        parameters_layout.addWidget(self.optimizer_combo, 1, 7)

        self.scheduler_label = QLabel("LR Scheduler:")
        parameters_layout.addWidget(self.scheduler_label, 1, 8)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItem("Constant", "constant")
        self.lr_scheduler_combo.addItem("Cosine", "cosine")
        self.lr_scheduler_combo.addItem("Linear", "linear")
        self.lr_scheduler_combo.addItem("Constant with warmup", "constant_with_warmup")
        self.lr_scheduler_combo.addItem("Cosine with restarts", "cosine_with_restarts")
        self.lr_scheduler_combo.addItem("Polynomial", "polynomial")
        parameters_layout.addWidget(self.lr_scheduler_combo, 1, 9)

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

        image_progress_layout = QVBoxLayout()
        self.image_label = ImageLabel()
        image_progress_layout.addWidget(self.image_label)
        validation_prompt_label = QLabel("Validation prompt:")
        image_progress_layout.addWidget(validation_prompt_label)
        self.validation_prompt_edit = QTextEdit()
        self.validation_prompt_edit.setMaximumHeight(60)
        image_progress_layout.addWidget(self.validation_prompt_edit)

        image_progress_layout.setStretch(0, 1)
        image_progress_layout.setStretch(1, 0)
        image_progress_layout.setStretch(2, 0)

        epoch_progress_layout = QHBoxLayout()
        self.epoch_progress_label = QLabel("Epoch 0/0")
        epoch_progress_layout.addWidget(self.epoch_progress_label, alignment=Qt.AlignmentFlag.AlignRight)
        self.steps_progress_label = QLabel("Steps 0/0")
        epoch_progress_layout.addWidget(self.steps_progress_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loss_progress_label = QLabel("AVG loss: 0.0")
        epoch_progress_layout.addWidget(self.loss_progress_label, alignment=Qt.AlignmentFlag.AlignLeft)
        image_progress_layout.addLayout(epoch_progress_layout)
        middle_layout.addLayout(image_progress_layout)

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

        self.setLayout(main_layout)

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
            self.output_dir = dialog.getExistingDirectory(None, "Select directory", self.directories.outputs_loras)
            self.output_dir_label.setText(os.path.basename(self.output_dir))
        elif directory_type == 2:
            self.model_path = dialog.getExistingDirectory(None, "Select directory", self.directories.models_diffusers)
            self.model_path_label.setText(os.path.basename(self.model_path))
        elif directory_type == 3:
            self.dataset_path = dialog.getExistingDirectory(None, "Select directory", self.directories.datasets)
            self.dataset_path_label.setText(os.path.basename(self.dataset_path))

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
        )
        self.train_thread = DreamboothLoraTrainThread(train_args)
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
        self.train_thread.start()

    def update_output(self, text):
        self.log_window.add_message(text)

    def update_output_done(self):
        self.log_window.append_success("done.")

    def show_error(self, text):
        self.log_window.error(text)
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