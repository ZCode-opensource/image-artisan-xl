from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QApplication,
    QFrame,
)
from PyQt6.QtCore import Qt, QEvent, pyqtSignal


class ItemSelector(QWidget):
    item_changed = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        QApplication.instance().installEventFilter(self)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.line_edit = QLineEdit(self)
        self.line_edit.setReadOnly(True)
        self.line_edit.mousePressEvent = self.toggle_list_widget
        layout.addWidget(self.line_edit)

        self.frame = QFrame(self)
        self.frame.setWindowFlags(Qt.WindowType.Popup)
        self.frame.setLayout(QVBoxLayout())

        self.listWidget = QListWidget(self.frame)
        self.listWidget.itemChanged.connect(self.update_line_edit)
        self.listWidget.itemClicked.connect(self.toggle_check_state)
        self.frame.layout().addWidget(self.listWidget)

    def add_item(self, text):
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        self.listWidget.addItem(item)

    def add_items(self, texts):
        self.listWidget.clear()
        for text in texts:
            self.add_item(text)

    def toggle_list_widget(self, _):
        if self.frame.isVisible():
            self.frame.hide()
        else:
            global_pos = self.line_edit.mapToGlobal(self.line_edit.rect().bottomLeft())
            self.frame.setFixedWidth(self.line_edit.width())
            self.frame.move(global_pos)
            self.frame.show()
            self.frame.adjustSize()

    def update_line_edit(self, _):
        checked_items = [
            self.listWidget.item(i).text()
            for i in range(self.listWidget.count())
            if self.listWidget.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.line_edit.setText(", ".join(checked_items))
        self.item_changed.emit()

    def find_text(self, text):
        for i in range(self.listWidget.count()):
            if self.listWidget.item(i).text() == text:
                return i
        return -1

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if (
                isinstance(watched, QWidget)
                and not self.isAncestorOf(watched)
                and not self.frame.isAncestorOf(watched)
            ):
                self.frame.hide()
        return super().eventFilter(watched, event)

    def toggle_check_state(self, item: QListWidgetItem):
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)

    def clear_selected_items(self):
        self.line_edit.clear()

        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
