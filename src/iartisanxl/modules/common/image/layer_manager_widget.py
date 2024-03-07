from importlib.resources import files

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QListWidgetItem, QSizePolicy, QVBoxLayout, QWidget

from iartisanxl.buttons.expand_contract_button import ExpandContractButton
from iartisanxl.buttons.transparent_button import TransparentButton
from iartisanxl.modules.common.image.layer_list_widget import LayerListWidget
from iartisanxl.modules.common.image.layer_widget import LayerWidget


class LayerManagerWidget(QWidget):
    ADD_LAYER_IMG = files("iartisanxl.theme.icons").joinpath("add_layer.png")
    DELETE_LAYER_IMG = files("iartisanxl.theme.icons").joinpath("delete_layer.png")

    layer_selected = pyqtSignal(int)
    layer_lock_changed = pyqtSignal(int, bool)
    layers_reordered = pyqtSignal(list)
    add_layer_clicked = pyqtSignal()
    delete_layer_clicked = pyqtSignal()

    EXPANDED_WIDTH = 150
    NORMAL_WIDTH = 25

    def __init__(self, right_side_version: bool = False):
        super().__init__()
        # self.setMinimumSize(self.EXPANDED_WIDTH, 50)
        # self.setMaximumWidth(self.EXPANDED_WIDTH)
        self.setMinimumSize(self.NORMAL_WIDTH, 50)
        self.setMaximumWidth(self.NORMAL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.right_side_version = right_side_version

        self.expanded = False
        self.animating = False
        self.animation_min = QPropertyAnimation(self, b"minimumWidth")
        self.animation_max = QPropertyAnimation(self, b"maximumWidth")
        self.animation_min.finished.connect(self.animation_finished)
        self.animation_min.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation_max.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation_min.setDuration(300)
        self.animation_max.setDuration(300)

        self.init_ui()

        self.list_widget.setVisible(False)
        self.layers_controls_widget.setVisible(False)

    def init_ui(self):
        button_alignment = Qt.AlignmentFlag.AlignLeft if self.right_side_version else Qt.AlignmentFlag.AlignRight

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.expand_btn = ExpandContractButton(25, 25, False, self.right_side_version)
        self.expand_btn.clicked.connect(self.on_expand_clicked)

        main_layout.addWidget(self.expand_btn, alignment=Qt.AlignmentFlag.AlignTop | button_alignment)

        self.list_widget = LayerListWidget()
        self.list_widget.currentItemChanged.connect(self.handle_item_selected)
        self.list_widget.layers_reordered.connect(self.on_layers_reordered)
        self.list_widget.setSpacing(0)
        main_layout.addWidget(self.list_widget)

        self.layers_controls_widget = QWidget()
        layers_controls_layout = QHBoxLayout()
        self.add_layer_button = TransparentButton(self.ADD_LAYER_IMG, 28, 28)
        self.add_layer_button.setObjectName("bottom_layer_control")
        self.add_layer_button.clicked.connect(self.on_add_layer_clicked)
        layers_controls_layout.addWidget(self.add_layer_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.delete_layer_button = TransparentButton(self.DELETE_LAYER_IMG, 28, 28)
        self.delete_layer_button.clicked.connect(self.on_delete_layer_clicked)
        self.delete_layer_button.setObjectName("bottom_layer_control")
        layers_controls_layout.addWidget(self.delete_layer_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.layers_controls_widget.setLayout(layers_controls_layout)
        main_layout.addWidget(self.layers_controls_widget)

        self.setLayout(main_layout)

    def on_add_layer_clicked(self):
        self.add_layer_clicked.emit()

    def add_layer(self, layer_id: int, name: str):
        item = QListWidgetItem()
        widget = LayerWidget(layer_id, name)
        item.setSizeHint(widget.sizeHint())
        widget.lock_changed.connect(self.on_lock_changed)

        self.list_widget.insertItem(0, item)
        self.list_widget.setItemWidget(item, widget)
        self.list_widget.setCurrentItem(item)

    def on_delete_layer_clicked(self):
        self.delete_layer_clicked.emit()

    def delete_layer(self, layer_id: int):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.layer_id == layer_id:
                self.list_widget.takeItem(i)
                break

    def get_layer_name(self, layer_id):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.layer_id == layer_id:
                return widget.name
        return None

    def handle_item_selected(self, current_item, previous_item):
        if current_item is not None:
            widget = self.list_widget.itemWidget(current_item)
            self.layer_selected.emit(widget.layer_id)

    def on_lock_changed(self, layer_id: int, locked: bool):
        self.layer_lock_changed.emit(layer_id, locked)

    def on_layers_reordered(self, layers: list):
        self.layers_reordered.emit(layers)

    def on_expand_clicked(self):
        if self.expanded:
            self.contract()
        else:
            self.expand()

    def expand(self):
        if self.animating:
            return

        self.animation_min.setStartValue(self.NORMAL_WIDTH)
        self.animation_min.setEndValue(self.EXPANDED_WIDTH)
        self.animation_max.setStartValue(self.NORMAL_WIDTH)
        self.animation_max.setEndValue(self.EXPANDED_WIDTH)
        self.animation_min.start()
        self.animation_max.start()
        self.animating = True

    def contract(self):
        if self.animating:
            return

        self.list_widget.setVisible(False)
        self.layers_controls_widget.setVisible(False)
        self.animation_min.setStartValue(self.EXPANDED_WIDTH)
        self.animation_min.setEndValue(self.NORMAL_WIDTH)
        self.animation_max.setStartValue(self.EXPANDED_WIDTH)
        self.animation_max.setEndValue(self.NORMAL_WIDTH)
        self.animation_min.start()
        self.animation_max.start()
        self.animating = True

    def animation_finished(self):
        if self.expanded:
            self.expanded = False
        else:
            self.list_widget.setVisible(True)
            self.layers_controls_widget.setVisible(True)
            self.expanded = True

        self.animating = False
