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
    return QImage(img, w, h, 3 * w, QImage.Format.Format_RGB888)

def make_pyramid(image):
    pyramid = [image]
    while((np.array(pyramid[-1].shape[:2]) > 1).all()):
        pyramid.append(cv2.resize(pyramid[-1], dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return pyramid

# Only allow specific zoom levels, otherwise the view result might look like crap.
scroll_zoom_levels = [2 ** (x / 4) for x in range(-28, 22)]
default_zoom_level = 28

def scale(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ], np.float64)

def rotate(angle):
    r = np.deg2rad(angle)
    cr = np.cos(r)
    sr = np.sin(r)
    return np.array([
        [cr, -sr, 0],
        [sr, cr, 0],
        [0, 0, 1],
    ], np.float64)

def translate(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ], np.float64)

def multiply(*matrices):
    result = np.identity(3, np.float64)
    for m in reversed(matrices):
        np.matmul(result, m, result)
    return result

def matrix_to_QTransform(matrix:np.ndarray) -> QTransform:
    m = matrix.T
    t = QTransform().setMatrix(m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2], m[2,0], m[2,1], m[2,2])
    return t

class Viewport(QGraphicsView):
    '''Display a canvas and handle input events for it.'''
    def __init__(self, canvas_state:CanvasState, parent=None):
        super().__init__(parent)
        self.position = QPointF(0.0, 0.0)
        self.zoom = default_zoom_level
        self.rotation = 0.0
        self.canvas_state = canvas_state
        self.composite_image:np.ndaraay = None
        self.last_mouse_pos = QPointF(0, 0)
        self.moving_view = False
        self.setScene(QGraphicsScene())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalScrollBar().disconnect(self)
        self.verticalScrollBar().disconnect(self)
        self.setMouseTracking(True)
        self.setTabletTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_transform()

    # TODO: Experiment with tiled pixmaps to help Qt optimize rendering.
    def apply_transform(self):
        size = self.size()
        view_w = size.width()
        view_h = size.height()
        view_x = self.position.x()
        view_y = self.position.y()
        image_x = self.composite_image.shape[1]
        image_y = self.composite_image.shape[0]
        zoom = scroll_zoom_levels[self.zoom]
        # TODO panning is correct, but zoom and rotation happens about the center of the pic
        # instead of the center of the screen.

        # TODO The reason I have to do this weird zoom hack is because Qt cannot display huge pixmaps
        # efficiently, however, opencv's warpAffine does not actually implement INTER_AREA, so downscaling
        # and transforming an image will look like crap, in which case it's better to use resize and then
        # let Qt take care of the transform again. It does not seem like this will be solved any time soon,
        # or perhaps ever. https://github.com/opencv/opencv/issues/21060
        if zoom < 1:
            img = cv2.resize(self.composite_image, dsize=None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
            self.pixmap = QGraphicsPixmapItem(QPixmap(np_to_qimage(img)))
            self.pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.scene().clear()
            self.scene().addItem(self.pixmap)
            t = (QTransform()
                .translate(view_w / 2, view_h / 2)
                .translate(view_x, view_y)
                .rotate(self.rotation)
                .translate(-img.shape[1] / 2, -img.shape[0] / 2)
            )
            self.pixmap.setTransform(t)
        else:
            target_buffer = np.empty((view_h, view_w, 3), dtype=np.ubyte)
            matrix = multiply(
                translate(-image_x / 2, -image_y / 2),
                rotate(self.rotation),
                scale(zoom),
                translate(view_x, view_y),
                translate(view_w / 2, view_h / 2),
            )[:2]
            inter = cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_CUBIC
            cv2.warpAffine(self.composite_image, matrix, dsize=(view_w, view_h), dst=target_buffer, flags=inter)
            self.pixmap = QGraphicsPixmapItem(QPixmap(np_to_qimage(target_buffer)))
            self.scene().clear()
            self.scene().addItem(self.pixmap)

    async def reset_viewport(self):
        self.scene().clear()
        canvas = self.canvas_state.get_current()
        # TODO: Test dispatching parallel tile workers.
        def comp_runner():
            size = canvas.size
            offset = (0, 0)
            color = np.zeros(size + (3,), dtype=DTYPE)
            alpha = np.zeros(size + (1,), dtype=DTYPE)
            color, _ = composite.composite(canvas.top_level, offset, (color, alpha))
            color *= 255
            color = color.astype(np.ubyte)
            return color
        self.composite_image = await util.peval(comp_runner)
        self.apply_transform()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.zoom += 1
        else:
            self.zoom -= 1
        self.zoom = max(self.zoom, 0)
        self.zoom = min(self.zoom, len(scroll_zoom_levels)-1)
        self.apply_transform()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_BracketLeft:
            self.rotation -= 15
            self.rotation %= 360
            self.apply_transform()
        elif event.key() == Qt.Key.Key_BracketRight:
            self.rotation += 15
            self.rotation %= 360
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

class StatusBar(QStatusBar):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.system_memory_bar = QProgressBar()
        self.own_memory_bar = QProgressBar()
        self.disk_bar = QProgressBar()
        self.system_memory_bar.setMaximumWidth(100)
        self.own_memory_bar.setMaximumWidth(100)
        self.disk_bar.setMaximumWidth(100)
        self.system_memory_bar.setTextVisible(False)
        self.own_memory_bar.setStyleSheet('background-color: rgba(0,0,0,0);')
        layout = QStackedLayout()
        layout.addWidget(self.system_memory_bar)
        layout.addWidget(self.own_memory_bar)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        stack = QWidget()
        stack.setMaximumWidth(100)
        stack.setLayout(layout)
        self.addPermanentWidget(QLabel(text='Memory Usage'))
        self.addPermanentWidget(stack)
        self.addPermanentWidget(QLabel(text='Disk Usage'))
        self.addPermanentWidget(self.disk_bar)
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(1000)
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start()
        self.update_status()

    def update_status(self):
        stats = util.get_system_stats()
        self.own_memory_bar.setValue(stats.own_memory_usage)
        self.own_memory_bar.setFormat(f'{stats.own_memory_usage}% ({stats.system_memory_usage}%)')
        self.system_memory_bar.setValue(stats.system_memory_usage)
        self.disk_bar.setValue(stats.disk_usage)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CrowPainter')
        self.setGeometry(0, 0, 1000, 1000)
        self.create_menus()
        self.canvases = []
        self.viewports = []
        self.viewport_tab = QTabWidget(self)
        self.viewport_tab.setTabsClosable(True)
        self.viewport_tab.tabCloseRequested.connect(self.on_tab_close_requested)
        self.setCentralWidget(self.viewport_tab)
        self.dock = QDockWidget()
        self.dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        statusbar = StatusBar()
        self.setStatusBar(statusbar)

    def on_tab_close_requested(self, index):
        widget = self.viewport_tab.widget(index)
        widget.close()
        self.viewport_tab.removeTab(index)

    def on_new(self):
        pass

    async def on_open(self):
        image_types = ['png', 'psd', 'psb', 'tif', 'tiff', 'jpg', 'jpeg', 'jpe', 'webp', 'bmp', 'dib', 'jp2', 'pbm', 'pgm', 'ppm', 'pnm']
        image_format_str = ' '.join([f'*.{ext}' for ext in image_types])
        files, _ = QFileDialog.getOpenFileNames(self, caption='Open', filter=f'Images ({image_format_str});;All files (*.*)', dir='.')
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
        #self.viewports.append(viewport)
        index = self.viewport_tab.addTab(viewport, viewport.canvas_state.file_path.name)
        self.viewport_tab.setCurrentIndex(index)

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
