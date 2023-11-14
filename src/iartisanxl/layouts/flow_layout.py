from PyQt6.QtCore import Qt, QMargins, QPoint, QRect, QSize
from PyQt6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(QMargins(0, 0, 0, 0))
        self._item_list = []
        self._visible_items = []
        self.sort_key = "name"
        self.sort_direction = "asc"
        self.tags_filter = None
        self.name_filter = None

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        # Add the item to the full list
        self._item_list.append(item)

        # Check if the item matches the current filters
        if self.matches_filters(item):
            # If it does, find the correct position in the visible items list
            index = self.find_insert_index(item)
            # And insert it there
            self._visible_items.insert(index, item)

    def matches_filters(self, item):
        if item.widget() is not None:
            data = item.widget().data
            item_tags = data.get("tags")
            item_name = data.get("name")

            tags_match = (
                set(self.tags_filter).issubset(item_tags.split(", "))
                if self.tags_filter
                else True
            )
            name_match = (
                self.name_filter.lower() in item_name.lower()
                if self.name_filter
                else True
            )

            return tags_match and name_match

        return False

    def find_insert_index(self, item):
        key = item.widget().data[self.sort_key].lower()
        for i, visible_item in enumerate(self._visible_items):
            visible_key = visible_item.widget().data[self.sort_key].lower()
            if (self.sort_direction == "asc" and key < visible_key) or (
                self.sort_direction == "desc" and key > visible_key
            ):
                return i

        # If no smaller item was found, the new item goes at the end
        return len(self._visible_items)

    def count(self):
        return len(self._item_list)

    def items(self):
        return self._item_list

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]

        return None

    def itemAtPosition(self, pos: QPoint):
        for i in range(self.count()):
            item = self.itemAt(i)
            if item.widget().geometry().contains(pos):
                return item
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)

        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        size += QSize(
            2 * self.contentsMargins().top(), 2 * self.contentsMargins().top()
        )
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._visible_items:
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QSizePolicy.ControlType.DefaultType,
                QSizePolicy.ControlType.DefaultType,
                Qt.Orientation.Horizontal,
            )
            layout_spacing_y = style.layoutSpacing(
                QSizePolicy.ControlType.DefaultType,
                QSizePolicy.ControlType.DefaultType,
                Qt.Orientation.Vertical,
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

    def clear(self):
        for i in reversed(range(self.count())):
            self.takeAt(i).widget().setParent(None)
        self._item_list = []
        self._visible_items = []

    def index_of(self, item: QWidget):
        for i in range(self.count()):
            if self.itemAt(i).widget() == item:
                return i
        return -1

    def remove_item(self, item: QWidget):
        index = self.index_of(item)
        if index >= 0:
            removed_item = self.takeAt(index)
            self._visible_items.remove(removed_item)
            removed_item.widget().deleteLater()
            self.update()

    def set_filters(self, tags, name):
        self.tags_filter = tags
        self.name_filter = name
        self.filter_items()

    def filter_items(self):
        self._visible_items.clear()
        for i in reversed(range(self.count())):
            item = self.itemAt(i)
            if item.widget() is not None:
                data = item.widget().data
                item_tags = data.get("tags")
                item_name = data.get("name")

                tags_match = (
                    set(self.tags_filter).issubset(item_tags.split(", "))
                    if self.tags_filter and item_tags
                    else not self.tags_filter
                )

                name_match = (
                    self.name_filter.lower() in item_name.lower()
                    if self.name_filter
                    else True
                )

                if tags_match and name_match:
                    item.widget().show()
                    self._visible_items.append(item)
                else:
                    item.widget().hide()

        self.order_by()
        self.update()

    def order_by(self):
        self._visible_items.sort(
            key=lambda item: item.widget().data[self.sort_key].lower(),
            reverse=(self.sort_direction == "desc"),
        )
        self.update()
