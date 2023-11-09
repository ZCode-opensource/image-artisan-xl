from importlib.resources import files

from PyQt6.QtWidgets import QHBoxLayout, QFrame, QLabel, QApplication, QWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPixmap, QCursor


class CloseButton(QWidget):
    CLOSE_IMG = files("iartisanxl.theme.icons").joinpath("close.png")
    red_color = QColor("#e81123")
    white_color = QColor("#ffffff")
    background_color = QColor("#161616")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(28, 28)
        self.hovered = False

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.hovered:
            painter.setBrush(QBrush(self.red_color))
        else:
            painter.setBrush(QBrush(self.background_color))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(1, 1, self.width() - 2, self.height())

        pixmap = QPixmap(str(self.CLOSE_IMG))
        pixmap = pixmap.scaled(
            20,
            20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(
            (self.width() - pixmap.width()) // 2,
            (self.height() - pixmap.height()) // 2,
            pixmap,
        )

    def enterEvent(self, _event):
        self.hovered = True
        self.update()

    def leaveEvent(self, _event):
        self.hovered = False
        self.update()


class MaxMinButton(QWidget):
    hover_color = QColor("#3c3c3c")
    background_color = QColor("#161616")

    def __init__(self, icon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(28, 28)
        self.hovered = False
        self.icon = icon

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.hovered:
            painter.setBrush(QBrush(self.hover_color))
        else:
            painter.setBrush(QBrush(self.background_color))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(1, 1, self.width(), self.height())

        pixmap = QPixmap(str(self.icon))
        pixmap = pixmap.scaled(
            20,
            20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(
            (self.width() - pixmap.width()) // 2,
            (self.height() - pixmap.height()) // 2,
            pixmap,
        )

    def enterEvent(self, _event):
        self.hovered = True
        self.update()

    def leaveEvent(self, _event):
        self.hovered = False
        self.update()

    def change_icon(self, new_icon):
        self.icon = new_icon
        self.update()


class TitleBar(QFrame):
    MINIMIZE_ICON = files("iartisanxl.theme.icons").joinpath("minimize.png")
    RESTORE_ICON = files("iartisanxl.theme.icons").joinpath("restore.png")
    MAXIMIZE_ICON = files("iartisanxl.theme.icons").joinpath("maximize.png")

    background_color = QColor("#161616")
    border_color = QColor("#ff6b6b6b")

    def __init__(self, *args, title: str = None, is_dialog: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.is_dialog = is_dialog
        self.setFixedHeight(28)
        self.start_pos = QPoint()
        self.close_button = None
        self.minimize_button = None
        self.maximize_button = None
        self.pressed_on_close = False
        self.pressed_on_minimize = False
        self.pressed_on_maximize = False
        self.was_maximized = False

        self.init_ui()

        if self.is_dialog:
            self.minimize_button.setVisible(False)
            self.maximize_button.setVisible(False)

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_label = QLabel(self.title)
        title_label.setContentsMargins(10, 0, 0, 0)
        main_layout.addWidget(title_label)

        self.minimize_button = MaxMinButton(self.MINIMIZE_ICON)
        main_layout.addWidget(self.minimize_button)

        self.maximize_button = MaxMinButton(self.MAXIMIZE_ICON)
        main_layout.addWidget(self.maximize_button)

        self.close_button = CloseButton()
        main_layout.addWidget(self.close_button)

        self.setLayout(main_layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(QBrush(self.background_color))
        pen = QPen(Qt.PenStyle.NoPen)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width(), self.height())

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, self.width(), 0)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.close_button.mapFromParent(event.pos())
            pos_minimize = self.minimize_button.mapFromParent(event.pos())
            pos_maximize = self.maximize_button.mapFromParent(event.pos())

            if self.close_button.rect().contains(pos):
                self.pressed_on_close = True
            elif self.minimize_button.rect().contains(pos_minimize):
                self.pressed_on_minimize = True
            elif self.maximize_button.rect().contains(pos_maximize):
                self.pressed_on_maximize = True
            else:
                self.start_pos = event.pos()

            self.was_maximized = self.window().isMaximized()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.close_button.mapFromParent(event.pos())
            pos_minimize = self.minimize_button.mapFromParent(event.pos())
            pos_maximize = self.maximize_button.mapFromParent(event.pos())

            if self.close_button.rect().contains(pos):
                self.close_app()
            elif self.minimize_button.rect().contains(pos_minimize):
                self.minimize_app()
            elif self.maximize_button.rect().contains(pos_maximize):
                self.maximize_app()

            self.pressed_on_close = False
            self.pressed_on_minimize = False
            self.pressed_on_maximize = False

        if self.was_maximized:
            self.was_maximized = False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if (
                isinstance(self.start_pos, QPoint)
                and not self.pressed_on_close
                and not self.pressed_on_minimize
                and not self.pressed_on_maximize
            ):
                if not self.close_button.rect().contains(event.pos()):
                    if self.window().isMaximized():
                        self.window().showNormal()
                        self.maximize_button.change_icon(self.MAXIMIZE_ICON)

                    if self.was_maximized:
                        new_pos = QCursor.pos() - QPoint(
                            self.width() // 2, self.height() // 2
                        )
                        self.window().move(new_pos)
                    else:
                        delta = event.pos() - self.start_pos
                        self.window().move(self.window().pos() + delta)

    def close_app(self):
        if self.is_dialog:
            self.parent().close()
        else:
            QApplication.quit()

    def minimize_app(self):
        if not self.is_dialog:
            self.window().showMinimized()

    def maximize_app(self):
        if not self.is_dialog:
            if self.window().isMaximized():
                self.maximize_button.change_icon(self.MAXIMIZE_ICON)
                self.window().showNormal()
            else:
                self.maximize_button.change_icon(self.RESTORE_ICON)
                self.window().showMaximized()
