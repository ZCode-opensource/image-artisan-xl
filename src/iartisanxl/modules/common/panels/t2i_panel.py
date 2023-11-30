from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.t2i_dialog import T2IDialog
from iartisanxl.app.preferences import PreferencesObject


class T2IPanel(BasePanel):
    def __init__(self, preferences: PreferencesObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferences = preferences

        self.event_bus = EventBus()
        self.event_bus.subscribe("t2i-adapters", self.on_t2i_adapters)
        self.controlnets = []

        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_controlnet_button = QPushButton("Add T2I Adapter")
        add_controlnet_button.clicked.connect(self.open_t2i_dialog)
        main_layout.addWidget(add_controlnet_button)

        added_controlnets_widget = QWidget()
        self.controlnets_layout = QVBoxLayout(added_controlnets_widget)
        main_layout.addWidget(added_controlnets_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def update_ui(self):
        pass

    def open_t2i_dialog(self):
        self.dialog_opened.emit(self, T2IDialog, "T2IDialog")

    def on_t2i_adapters(self, data):
        pass

    def clean_up(self):
        self.event_bus.unsubscribe("t2i-adapters", self.on_t2i_adapters)
