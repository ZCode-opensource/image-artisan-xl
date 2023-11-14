from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QGraphicsView,
    QMenu,
    QGraphicsScene,
    QFileDialog,
    QApplication,
)
from PyQt6.QtGui import (
    QShortcut,
    QKeySequence,
    QWheelEvent,
    QTransform,
    QMouseEvent,
    QContextMenuEvent,
    QAction,
    QImageWriter,
    QScreen,
)
from PyQt6.QtCore import Qt

from iartisanxl.modules.common.dialogs.full_screen_preview import FullScreenPreview


class ImageViewerSimple(QGraphicsView):
    def __init__(self, output_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setScene(QGraphicsScene())

        self.selected_screen = None
        self.full_screen_preview = None

        # Create the context menu
        self.context_menu = QMenu(self)
        self.output_path = output_path
        self.save_action = self.context_menu.addAction("Save image")
        self.save_action.triggered.connect(self.save_image)

        # Create a QShortcut to trigger the save_image function when the user presses Ctrl+S
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_image)

        self.initial_scale_factor = None
        self.serialized_data = None
        self.pixmap_item = None
        self._drag_pos = None

    def set_pixmap(self, pixmap):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.initial_scale_factor = self.transform().m11()
        if self.selected_screen is not None and self.full_screen_preview is not None:
            self.full_screen_preview.image_preview_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        # Get the current position of the mouse
        mouse_pos = event.position().toPoint()

        # Map the mouse position from the viewport to the scene
        scene_pos = self.mapToScene(mouse_pos)

        # Get the current scale factor of the view
        scale_factor = self.transform().m11()

        # Set the zoom factor (how much to zoom in or out)
        zoom_factor = 1.25

        # Check if the Control key is being pressed
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Check the direction of the scroll
            delta = event.angleDelta().y()
            if delta > 0:
                # Zoom in
                scale_factor *= zoom_factor
            elif delta < 0:
                # Zoom out
                scale_factor /= zoom_factor

            # Prevent zooming out beyond the original image size
            if (
                self.initial_scale_factor is not None
                and scale_factor < self.initial_scale_factor
            ):
                scale_factor = self.initial_scale_factor

            # Set the new scale factor of the view
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.setTransform(QTransform.fromScale(scale_factor, scale_factor))

            # Center the view on the current mouse position
            new_scene_pos = self.mapToScene(mouse_pos)
            delta = new_scene_pos - scene_pos
            self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event: QMouseEvent):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # If the left mouse button is pressed, store the initial position of the mouse
            self._drag_pos = event.position().toPoint()

            # Change the cursor shape to a hand
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

            # Call the base class implementation of mousePressEvent
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self._drag_pos is not None
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # If the left mouse button is being held down and the initial drag position is set,
            # move the view
            delta = event.position().toPoint() - self._drag_pos

            # Set the speed factor (how fast to move the view)
            speed_factor = 0.5

            self.translate(delta.x(), delta.y() * speed_factor)

            # Update the initial drag position
            self._drag_pos = event.position().toPoint()

            # Call the base class implementation of mouseMoveEvent
            super().mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # If the left mouse button is released, reset the initial drag position
            self._drag_pos = None

            # Reset the cursor shape
            self.unsetCursor()

            # Call the base class implementation of mouseReleaseEvent
            super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        menu.addAction(self.save_action)
        full_screen_preview_action: QAction | None = menu.addAction(
            "Full Screen Preview"
        )

        screens = QApplication.screens()
        submenu = QMenu("Submenu", self)

        none_action = submenu.addAction("None")
        none_action.setCheckable(True)
        none_action.setChecked(self.selected_screen is None)
        none_action.triggered.connect(lambda checked: self.on_monitor_selected(None))

        current_screen = self.window().screen()

        for i, screen in enumerate(screens, start=1):
            action = submenu.addAction(f"Monitor {i} - {screen.name()}")
            action.setCheckable(True)
            action.setChecked(screen == self.selected_screen)

            if screen == current_screen:
                action.setEnabled(False)

            action.triggered.connect(
                lambda checked, s=screen: self.on_monitor_selected(s)
            )

        full_screen_preview_action.setMenu(submenu)
        menu.exec(event.globalPos())

    def on_monitor_selected(self, screen: Optional[QScreen]):
        self.selected_screen = screen

        if screen is not None:
            self.full_screen_preview = FullScreenPreview()
            self.full_screen_preview.move(screen.geometry().x(), screen.geometry().y())
            self.full_screen_preview.showFullScreen()
        else:
            self.full_screen_preview = None

    def save_image(self):
        if self.pixmap_item is not None:
            # Generate a timestamp string
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Ask the user for a file name to save the image
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save image",
                f"{self.output_path}/{timestamp}.png",
                "Images (*.png *.jpg)",
            )

            if file_name:
                qimage = self.pixmap_item.pixmap().toImage()
                writer = QImageWriter(file_name, b"png")
                writer.setText("data", self.serialized_data)
                writer.write(qimage)

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass
