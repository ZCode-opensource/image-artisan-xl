from PyQt6.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QTimer

from iartisanxl.buttons.expand_right_button import ExpandRightButton
from iartisanxl.buttons.vertical_button import VerticalButton
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.prompt_window import PromptWindow


class RightMenu(QFrame):
    EXPANDED_WIDTH = 400
    NORMAL_WIDTH = 40

    def __init__(
        self,
        module_options: dict,
        directories: DirectoriesObject,
        image_generation_data: ImageGenData,
        image_viewer: ImageViewerSimple,
        prompt_window: PromptWindow,
        auto_generate_function: callable,
        open_dialog: callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.module_options = module_options
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.image_viewer = image_viewer
        self.prompt_window = prompt_window
        self.auto_generate_function = auto_generate_function
        self.module_open_dialog = open_dialog

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

        self.panel_container = QWidget()
        self.panel_container.setObjectName("panel_container")
        self.panel_layout = QVBoxLayout()
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(0)
        self.panel_container.setLayout(self.panel_layout)
        self.panel_container.setMinimumWidth(0)
        self.panel_container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
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
                self.directories,
                self.image_generation_data,
                self.prompt_window,
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

        if (
            hasattr(self, "current_panel")
            and self.current_panel is not None
            and self.current_panel != panel_class
        ):
            self.current_panel.setParent(None)

        panel = panel_class(*args, **kwargs)
        panel.dialog_opened.connect(self.on_open_dialog)
        self.panel_layout.addWidget(panel)

        self.current_panel = panel
        self.current_panel_text = text

    def update_ui(self):
        if self.current_panel is not None:
            self.current_panel.update_ui()

    def on_open_dialog(self, dialog_class, title):
        dialog = dialog_class(
            self.directories,
            title,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
            self.auto_generate_function,
        )
        dialog.generation_updated.connect(self.update_ui)
        self.module_open_dialog(dialog)
