from PyQt6.QtWidgets import QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QPixmap

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.expand_right_button import ExpandRightButton
from iartisanxl.buttons.vertical_button import VerticalButton
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.lora_list import LoraList
from iartisanxl.generation.adapter_list import AdapterList
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.modules.common.panels.panel_container import PanelContainer
from iartisanxl.modules.common.controlnet.controlnet_panel import ControlNetPanel
from iartisanxl.modules.common.controlnet.controlnet_added_item import ControlNetAddedItem


class RightMenu(QFrame):
    EXPANDED_WIDTH = 400
    NORMAL_WIDTH = 40

    def __init__(
        self,
        module_options: dict,
        preferences: PreferencesObject,
        directories: DirectoriesObject,
        image_generation_data: ImageGenerationData,
        lora_list: LoraList,
        controlnet_list: AdapterList,
        t2i_adapter_list: AdapterList,
        ip_adapter_list: AdapterList,
        image_viewer: ImageViewerSimple,
        prompt_window: PromptWindow,
        show_error: callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.module_options = module_options
        self.preferences = preferences
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.controlnet_list = controlnet_list
        self.t2i_adapter_list = t2i_adapter_list
        self.ip_adapter_list = ip_adapter_list
        self.image_viewer = image_viewer
        self.prompt_window = prompt_window
        self.show_error = show_error

        self.event_bus = EventBus()
        self.event_bus.subscribe("controlnet", self.on_controlnet)

        self.expanded = self.module_options.get("right_menu_expanded")
        self.animating = False
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.finished.connect(self.animation_finished)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.setDuration(300)

        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(self.EXPANDED_WIDTH, 50)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        self.panels = {}
        self.current_panel = None
        self.current_panel_text = None
        self.init_ui()

    def init_ui(self):
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.button_layout = QVBoxLayout()
        self.expand_btn = ExpandRightButton()
        self.expand_btn.clicked.connect(self.on_expand_clicked)
        self.button_layout.addWidget(self.expand_btn)

        self.button_layout.addStretch()
        self.main_layout.addLayout(self.button_layout)

        self.panel_container = PanelContainer()
        self.main_layout.addWidget(self.panel_container)

        self.setLayout(self.main_layout)

    def add_panel(self, text, panel_class, *args, **kwargs):
        button = VerticalButton(text)
        index = self.button_layout.count() - 1
        self.button_layout.insertWidget(index, button)

        self.panels[text] = {
            "class": panel_class,
            "args": (
                *args,
                self.module_options,
                self.preferences,
                self.directories,
                self.image_viewer,
                self.prompt_window,
                self.show_error,
                self.image_generation_data,
                self.lora_list,
                self.controlnet_list,
                self.t2i_adapter_list,
                self.ip_adapter_list,
            ),
            "kwargs": kwargs,
        }

        button.clicked.connect(lambda: self.on_button_clicked(text))

        if self.current_panel_text is None:
            if self.expanded:
                self.show_panel(text)
            else:
                self.setFixedWidth(self.NORMAL_WIDTH)
                self.expand_btn.extended = False
                self.current_panel_text = text

    def on_button_clicked(self, text):
        self.current_panel_text = text
        self.expand()

    def animation_finished(self):
        if self.expanded:
            self.expanded = False
        else:
            self.expanded = True

        self.module_options["right_menu_expanded"] = self.expanded

        self.animating = False

    def on_expand_clicked(self):
        if self.expanded:
            self.contract()
        else:
            self.expand()

    def expand(self):
        if self.animating:
            return

        if self.expanded:
            self.show_panel(self.current_panel_text)
        else:
            self.animation.setStartValue(self.NORMAL_WIDTH)
            self.animation.setEndValue(self.EXPANDED_WIDTH)
            self.animation.start()
            self.animating = True

            QTimer.singleShot(
                self.animation.duration(),
                lambda: self.show_panel(self.current_panel_text),
            )

    def contract(self):
        if self.animating:
            return

        self.current_panel.setParent(None)
        self.animation.setStartValue(self.EXPANDED_WIDTH)
        self.animation.setEndValue(self.NORMAL_WIDTH)
        self.animation.start()
        self.animating = True

    def show_panel(self, text):
        panel_info = self.panels[text]
        panel_class = panel_info["class"]
        args = panel_info["args"]
        kwargs = panel_info["kwargs"]

        if hasattr(self, "current_panel") and self.current_panel is not None and self.current_panel != panel_class:
            self.current_panel.setParent(None)
            self.current_panel.clean_up()
            del self.current_panel

        panel = panel_class(*args, **kwargs)
        self.panel_container.panel_layout.addWidget(panel)

        self.current_panel = panel
        self.current_panel_text = text

    def close_all_dialogs(self):
        self.panel_container.close_all_dialogs()

    def on_controlnet(self, data):
        if data["action"] == "add":
            adapter_id = self.controlnet_list.add(data["controlnet"])

            if isinstance(self.current_panel, ControlNetPanel):
                data["controlnet"].adapter_id = adapter_id
                controlnet_widget = ControlNetAddedItem(data["controlnet"])
                controlnet_widget.remove_clicked.connect(self.current_panel.on_remove_clicked)
                controlnet_widget.edit_clicked.connect(self.current_panel.on_edit_clicked)
                controlnet_widget.enabled.connect(self.current_panel.on_controlnet_enabled)
                self.current_panel.controlnets_layout.addWidget(controlnet_widget)
        elif data["action"] == "update":
            controlnet = data["controlnet"]

            if isinstance(self.current_panel, ControlNetPanel):
                self.current_panel.controlnet_list.update_with_adapter_data_object(controlnet)
                for i in range(self.current_panel.controlnets_layout.count()):
                    widget = self.current_panel.controlnets_layout.itemAt(i).widget()
                    if widget.controlnet.adapter_id == controlnet.adapter_id:
                        widget.enabled_checkbox.setText(controlnet.adapter_name)
                        source_thumb_pixmap = QPixmap(controlnet.source_image_thumb)
                        widget.source_thumb.setPixmap(source_thumb_pixmap)
                        annotator_thumb_pixmap = QPixmap(controlnet.annotator_image_thumb)
                        widget.annotator_thumb.setPixmap(annotator_thumb_pixmap)
                        widget.controlnet = data["controlnet"]
                        break
