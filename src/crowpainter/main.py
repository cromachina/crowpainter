import sys
from pathlib import Path
import asyncio
from collections import deque
import logging
import traceback

import cv2
import numpy as np
from pyrsistent import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import PySide6.QtAsyncio as QtAsyncio

from . import layer_data, composite, util
from .constants import *
from .file_io import psd, image, native

class ExtendedInfoMessage(QDialog):
    def __init__(self, parent=None, title='', text=''):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.text = text
        copy_button = QPushButton(text='Copy text')
        copy_button.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.text))
        button = QDialogButtonBox.Close
        self.buttonBox = QDialogButtonBox(button)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout = QVBoxLayout()
        message = QPlainTextEdit(self)
        message.setPlainText(text)
        message.setReadOnly(True)
        layout.addWidget(message)
        layout.addWidget(copy_button)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

def show_error_message(text):
    ExtendedInfoMessage(title='Error', text=text).exec()

class CanvasState():
    def __init__(self, initial_state:layer_data.Canvas, file_path:Path, on_filesystem:bool):
        self.file_path = file_path
        self.on_filesystem = on_filesystem
        self.current_index = 0
        self.saved_state = initial_state
        self.selected_objects = []
        self.states = deque([initial_state])

    def get_current(self):
        return self.states[self.current_index]

    def append(self, state, state_limit:int|None=None):
        self.states = self.states.append(state)
        if state_limit is not None and len(self.states) > state_limit:
            self.states = self.states.popleft()
        self.current_index = len(self.states) - 1

    def undo(self):
        self.current_index = max(0, self.current_index - 1)
        return self.get_current()

    def redo(self):
        self.current_index = min(self.current_index + 1, len(self.states) - 1)
        return self.get_current()

    def can_undo(self):
        return self.current_index != 0

    def can_redo(self):
        return self.current_index != (len(self.states) - 1)

    def is_saved(self):
        return id(self.saved_state) == id(self.get_current())

    def set_saved(self):
        self.saved_state = self.get_current()

def np_to_qimage(img):
    h, w, _ = img.shape
    return QImage(img, w, h, 3 * w, QImage.Format.Format_RGBA64 .Format_RGB888)

def make_pyramid(image):
    pyramid = [image]
    while((np.array(pyramid[-1].shape[:2]) > 1).all()):
        pyramid.append(cv2.resize(pyramid[-1], dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return pyramid

class Viewport(QGraphicsView):
    '''Display a canvas and handle input events for it.'''
    def __init__(self, canvas_state:CanvasState, parent=None):
        super().__init__(parent)
        self.position = QPointF(0.0, 0.0)
        self.zoom = 1.0
        self.last_zoom = self.zoom
        self.rotation = 0.0
        self.canvas_state = canvas_state
        self.composite = None
        self.pyramid = None

        self.image:QImage = None
        self.last_mouse_pos = QPointF(0, 0)
        self.moving_view = False
        self.setScene(QGraphicsScene())
        self.setBackgroundBrush(QColor(176, 176, 176))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalScrollBar().disconnect(self)
        self.verticalScrollBar().disconnect(self)
        self.setMouseTracking(True)
        self.setTabletTracking(True)
        self.setSceneRect(QRect(-1000000, -1000000, 2000000, 2000000))

    # NOTE: opencv seems to resolve resizing adqeuately, but rotations still look like crap.
    # might create a custom 'framebuffer' to render nicely myself
    def apply_transform(self):
        t = QTransform().translate(self.position.x(), self.position.y()).rotate(self.rotation)
        if self.zoom > 1.0:
            t = t.scale(self.zoom, self.zoom)
            self.pixmap.setTransformationMode(Qt.TransformationMode.FastTransformation)
        elif self.zoom != self.last_zoom:
            self.last_zoom = self.zoom
            h, w, _ = self.image.shape
            img = cv2.resize(self.image, (int(w * self.zoom), int(h * self.zoom)), interpolation=cv2.INTER_AREA)
            qim = np_to_qimage(img)
            self.pixmap = QGraphicsPixmapItem(QPixmap(qim))
            self.pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.scene().clear()
            self.scene().addItem(self.pixmap)
        self.setTransform(t)

    async def reset_viewport(self):
        self.scene().clear()
        canvas = self.canvas_state.get_current()
        def comp_runner():
            size = canvas.size
            offset = (0, 0)
            color = np.zeros(size + (3,), dtype=DTYPE)
            alpha = np.zeros(size + (1,), dtype=DTYPE)
            color, _ = composite.composite(canvas.top_level, offset, (color, alpha))
            color *= 255
            color = color.astype(np.ubyte)
            return color
        self.image = await util.peval(comp_runner)
        self.pixmap = QGraphicsPixmapItem(QPixmap(np_to_qimage(self.image)))
        self.scene().addItem(self.pixmap)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.zoom = max(self.zoom, 0.0078)
        self.zoom = min(self.zoom, 32.0)
        self.apply_transform()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_BracketLeft:
            self.rotation += 15
        elif event.key() == Qt.Key.Key_BracketRight:
            self.rotation -= 15
        self.apply_transform()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == event.button().LeftButton:
            self.moving_view = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == event.button().LeftButton:
            self.moving_view = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.moving_view:
            delta = event.position() - self.last_mouse_pos
            self.position += delta
            self.apply_transform()
        self.last_mouse_pos = event.position()

class LayerList(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CrowPainter')
        self.setGeometry(0, 0, 1000, 1000)
        self.create_menus()

        self.canvases = []
        self.viewports = []
        self.active_viewport = None

        #self.setCentralWidget(viewport)
        #self.layout().addChildWidget(viewport)
        self.viewport_tab = QTabWidget(self)
        self.setCentralWidget(self.viewport_tab)
        self.dock = QDockWidget()
        self.dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

    def on_new(self):
        pass

    async def on_open(self):
        files, _ = QFileDialog.getOpenFileNames(self, caption='Open', filter='All files (*.*)', dir='.')
        for file_path in files:
            # TODO check if file is already open and ask to reopen without saving.
            await self.open(file_path)

    def on_save(self):
        pass

    def on_save_as(self):
        pass

    def on_close(self):
        pass

    async def open(self, file_path):
        file_path = Path(file_path)
        try:
            canvas = await util.peval(lambda: open_file(file_path))
        except Exception as ex:
            logging.exception(ex)
            show_error_message(ex)
            return

        canvas_state = CanvasState(
            initial_state=canvas,
            file_path=file_path,
            on_filesystem=True
        )
        viewport = Viewport(canvas_state)
        await viewport.reset_viewport()
        self.viewports.append(viewport)
        self.viewport_tab.addTab(viewport, viewport.canvas_state.file_path.name)

    def create_menu_action(self, menu:QMenu, text:str, callback, enabled=True):
        action = QAction(text=text, parent=self)
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def create_menus(self):
        menu = self.menuBar()
        menu_file = menu.addMenu('&File')

        self.action_new = self.create_menu_action(menu_file, '&New ...', self.on_new)
        self.action_open = self.create_menu_action(menu_file, '&Open ...', lambda: asyncio.ensure_future(self.on_open()))
        self.action_save = self.create_menu_action(menu_file, '&Save ...', self.on_save)
        self.action_save_as = self.create_menu_action(menu_file, 'Save &As ...', self.on_save_as)
        self.action_close = self.create_menu_action(menu_file, '&Close ...', self.on_close)

        menu_edit = menu.addMenu('&Edit')
        menu_canvas = menu.addMenu('&Canvas')
        menu_layer = menu.addMenu('&Layer')
        menu_selection = menu.addMenu('&Selection')
        self.menu_view = menu.addMenu('&View')
        menu_window = menu.addMenu('&Window')

def open_file(file_path:Path):
    file_type = file_path.suffix
    if file_type in ['.psd', '.psb']:
        return psd.read(file_path)
    elif file_type == '.crow':
        return native.read(file_path)
    else:
        return image.read(file_path)

def init_logging():
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)s] %(message)s',
        level=logging.INFO
    )

def main():
    init_logging()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    def excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.exception(tb)
        ExtendedInfoMessage(title='Error', text=tb).exec()
    sys.excepthook = excepthook
    main_window.show()
    QtAsyncio.run()

if __name__ == "__main__":
    main()