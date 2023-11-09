from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QFileDialog,
    QLineEdit,
)
import pyqtgraph as pg

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.console.console_stream import ConsoleStream
from iartisanxl.windows.log_window import LogWindow
from iartisanxl.threads.lora_train_thread import LoraTrainThread
from iartisanxl.train.lora_train_args import LoraTrainArgs


class TrainLoraModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_dir = ""
        self.model_path = ""
        self.dataset_path = ""
        self.train_thread = None
        self.epoch_data = []
        self.loss_data = []

        self.init_ui()
        self.console_stream = ConsoleStream()
        self.set_button_train()

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()

        self.log_window = LogWindow()
        top_layout.addWidget(self.log_window)

        right_layout = QVBoxLayout()
        self.graphWidget = pg.PlotWidget()
        right_layout.addWidget(self.graphWidget)

        top_layout.addLayout(right_layout)

        main_layout.addLayout(top_layout)

        bottom_layout = QHBoxLayout()

        configuration_layout = QGridLayout()
        configuration_layout.setSpacing(5)

        output_dir_button = QPushButton("Select output directory")
        output_dir_button.clicked.connect(lambda: self.select_directory(1))
        configuration_layout.addWidget(output_dir_button, 0, 0)
        output_dir_title_label = QLabel("Output directory:")
        configuration_layout.addWidget(output_dir_title_label, 0, 1)
        self.output_dir_label = QLabel()
        configuration_layout.addWidget(self.output_dir_label, 0, 2)

        rank_label = QLabel("Rank:")
        configuration_layout.addWidget(rank_label, 0, 3)
        self.rank_text_edit = QLineEdit()
        self.rank_text_edit.setText("4")
        configuration_layout.addWidget(self.rank_text_edit, 0, 4)

        workers_label = QLabel("Workers:")
        configuration_layout.addWidget(workers_label, 0, 5)
        self.workers_text_edit = QLineEdit()
        self.workers_text_edit.setText("8")
        configuration_layout.addWidget(self.workers_text_edit, 0, 6)

        save_steps_label = QLabel("Save steps:")
        configuration_layout.addWidget(save_steps_label, 0, 7)
        self.save_steps_text_edit = QLineEdit()
        self.save_steps_text_edit.setText("64")
        configuration_layout.addWidget(self.save_steps_text_edit, 0, 8)

        model_select_button = QPushButton("Select model")
        model_select_button.clicked.connect(lambda: self.select_directory(2))
        configuration_layout.addWidget(model_select_button, 1, 0)
        model_title_label = QLabel("Model:")
        configuration_layout.addWidget(model_title_label, 1, 1)
        self.model_path_label = QLabel()
        configuration_layout.addWidget(self.model_path_label, 1, 2)

        learning_rate_label = QLabel("Learning rate:")
        configuration_layout.addWidget(learning_rate_label, 1, 3)
        self.learning_rate_text_edit = QLineEdit()
        self.learning_rate_text_edit.setText("1e-4")
        configuration_layout.addWidget(self.learning_rate_text_edit, 1, 4)

        accumulation_steps_label = QLabel("Accumulation steps:")
        configuration_layout.addWidget(accumulation_steps_label, 1, 5)
        self.accumulation_steps_text_edit = QLineEdit()
        self.accumulation_steps_text_edit.setText("4")
        configuration_layout.addWidget(self.accumulation_steps_text_edit, 1, 6)

        dataset_select_button = QPushButton("Select dataset")
        dataset_select_button.clicked.connect(lambda: self.select_directory(3))
        configuration_layout.addWidget(dataset_select_button, 2, 0)
        dataset_title_label = QLabel("Dataset:")
        configuration_layout.addWidget(dataset_title_label, 2, 1)
        self.dataset_path_label = QLabel()
        configuration_layout.addWidget(self.dataset_path_label, 2, 2)

        batch_size_label = QLabel("Batch size:")
        configuration_layout.addWidget(batch_size_label, 2, 3)
        self.batch_size_text_edit = QLineEdit()
        self.batch_size_text_edit.setText("1")
        configuration_layout.addWidget(self.batch_size_text_edit, 2, 4)

        epochs_label = QLabel("Epochs:")
        configuration_layout.addWidget(epochs_label, 2, 5)
        self.epochs_text_edit = QLineEdit()
        self.epochs_text_edit.setText("200")
        configuration_layout.addWidget(self.epochs_text_edit, 2, 6)

        bottom_layout.addLayout(configuration_layout)

        button_layout = QVBoxLayout()
        self.train_button = QPushButton()
        self.train_button.clicked.connect(self.train_clicked)
        button_layout.addStretch()
        button_layout.addWidget(self.train_button)
        bottom_layout.addLayout(button_layout)

        bottom_layout.setStretch(0, 10)
        bottom_layout.setStretch(1, 2)

        main_layout.addLayout(bottom_layout)
        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 2)

        self.setLayout(main_layout)

    def set_button_train(self):
        self.train_button.setStyleSheet("background-color: #0b5e26;")
        self.train_button.setText("Train")
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
            self.output_dir = dialog.getExistingDirectory(
                None, "Select directory", "./outputs/lora/diffusers"
            )
            self.output_dir_label.setText(self.output_dir)
        elif directory_type == 2:
            self.model_path = dialog.getExistingDirectory(
                None, "Select directory", "./models/diffusers/base"
            )
            self.model_path_label.setText(self.model_path)
        elif directory_type == 3:
            self.dataset_path = dialog.getExistingDirectory(
                None, "Select directory", "./datasets"
            )
            self.dataset_path_label.setText(self.dataset_path)

    def train_clicked(self):
        train_args = LoraTrainArgs(
            output_dir=self.output_dir,
            model_path=self.model_path,
            rank=int(self.rank_text_edit.text()),
            learning_rate=float(self.learning_rate_text_edit.text()),
            dataset_path=self.dataset_path,
            batch_size=int(self.batch_size_text_edit.text()),
            workers=int(self.workers_text_edit.text()),
            accumulation_steps=int(self.accumulation_steps_text_edit.text()),
            epochs=int(self.epochs_text_edit.text()),
            save_steps=int(self.save_steps_text_edit.text()),
        )
        self.train_thread = LoraTrainThread(train_args)
        self.train_thread.output.connect(self.update_output)
        self.train_thread.output_done.connect(self.update_output_done)
        self.train_thread.warning.connect(self.show_warning)
        self.train_thread.error.connect(self.show_error)
        self.train_thread.update_epoch.connect(self.update_epoch)
        self.train_thread.start()

    def update_output(self, text):
        self.log_window.add_message(text)

    def update_output_done(self):
        self.log_window.append_success("done.")

    def show_error(self, text):
        self.log_window.error(text)
        self.show_snackbar("Could not start training since there was an error.")

    def show_warning(self, text):
        self.log_window.warning(text)

    def update_epoch(self, epoch, loss):
        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.graphWidget.plot(self.epoch_data, self.loss_data, clear=True)
