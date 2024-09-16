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

from . import layer_data, composite, util, blendfuncs
from .constants import *
from .file_io import psd, image, native

class ExtendedInfoMessage(QDialog):
    def __init__(self, parent=None, title='', text=''):
        super().__init__(parent)
        self.setWindowTitle(title)
        message = QPlainTextEdit()
        message.setPlainText(text)
        message.setReadOnly(True)
        copy_button = QPushButton(text='Copy text')
        copy_button.clicked.connect(lambda: QGuiApplication.clipboard().setText(text))
        close_button = QDialogButtonBox(QDialogButtonBox.Close)
        close_button.accepted.connect(self.accept)
        close_button.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(message)
        layout.addWidget(copy_button)
        layout.addWidget(close_button)
        self.setLayout(layout)
        self.setModal(True)

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
    h, w, d = img.shape
    match d:
        case 1:
            format = QImage.Format.Format_Grayscale8
        case 3:
            format = QImage.Format.Format_RGB888
        case 4:
            format = QImage.Format.Format_RGBA8888
    return QImage(img, w, h, format)

def make_pyramid(image):
    pyramid = [image]
    while((np.array(pyramid[-1].shape[:2]) > 1).all()):
        pyramid.append(cv2.resize(pyramid[-1], dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return pyramid

# Only allow specific zoom levels, otherwise the view result might look like crap.
scroll_zoom_levels = [2 ** (x / 4) for x in range(-28, 21)]
default_zoom_level = 28

def find_fitting_zoom_level(view, image):
    best_fit = 0
    for index, zoom in zip(range(len(scroll_zoom_levels)), scroll_zoom_levels):
        if image * zoom < view:
            best_fit = index
        else:
            break
    return best_fit

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
    for m in matrices:
        np.matmul(result, m, result)
    return result

def matrix_to_QTransform(matrix:np.ndarray) -> QTransform:
    m = matrix.T
    t = QTransform().setMatrix(m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2], m[2,0], m[2,1], m[2,2])
    return t

async def full_composite(canvas:layer_data.Canvas):
    # TODO: Test dispatching parallel tile workers.
    def comp_runner():
        size = canvas.size
        offset = (0, 0)
        if canvas.background.transparent:
            color = np.zeros(size + (3,), dtype=BLENDING_DTYPE)
            alpha = np.zeros(size + (1,), dtype=BLENDING_DTYPE)
        else:
            color = np.full(size + (3,), dtype=BLENDING_DTYPE, fill_value=np.array(canvas.background.color) / 255.0)
            alpha = np.ones(size + (1,), dtype=BLENDING_DTYPE)
        color, alpha = composite.composite(canvas.top_level, offset, (color, alpha))
        blendfuncs.clip_divide(color, alpha, out=color)
        color *= 255
        color = color.astype(STORAGE_DTYPE)
        alpha *= 255
        alpha = alpha.astype(STORAGE_DTYPE)
        return np.dstack((color, alpha))
    return await util.peval(comp_runner)

def make_checkerboard_texture(check_a, check_b, size):
    check_a = util.clamp(0, 255, check_a)
    check_b = util.clamp(0, 255, check_b)
    arr = np.array([check_a, check_b, check_b, check_a], dtype=STORAGE_DTYPE).reshape((2,2,1))
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST).reshape((size, size, 1))

def make_ideal_checkerboard(value, size):
    a = value * 0.9
    b = value + 0.1
    return make_checkerboard_texture(a * 255, b * 255, size)

class Viewport(QGraphicsView):
    '''Display a canvas and handle input events for it.'''
    def __init__(self, canvas_state:CanvasState, initial_composite=None, parent=None):
        super().__init__(parent)
        self.position = QPointF()
        self.zoom = default_zoom_level
        self.last_zoom = None
        self.rotation = 0.0
        self.canvas_state = canvas_state
        self.composite_image:np.ndarray = initial_composite
        self.last_mouse_pos = QPointF(0, 0)
        self.moving_view = False
        self.canvas_bg_area = QGraphicsPolygonItem()
        self.canvas_pixmap = QGraphicsPixmapItem()
        self.setScene(QGraphicsScene())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalScrollBar().disconnect(self)
        self.verticalScrollBar().disconnect(self)
        self.setMouseTracking(True)
        self.setTabletTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene().addItem(self.canvas_bg_area)
        self.scene().addItem(self.canvas_pixmap)
        self.update_background()
        self.first_show = True

    def apply_transform(self):
        size = self.size()
        self.setSceneRect(self.rect())
        view_w = size.width()
        view_h = size.height()
        view_x = self.position.x()
        view_y = self.position.y()
        image_w = self.composite_image.shape[1]
        image_h = self.composite_image.shape[0]
        zoom = scroll_zoom_levels[self.zoom]
        t = (QTransform()
            .translate(view_w / 2, view_h / 2)
            .rotate(self.rotation)
            .translate(view_x, view_y)
            .translate(-image_w * zoom / 2, -image_h * zoom / 2)
        )
        self.canvas_bg_area.setPolygon(t.mapToPolygon(QRect(0, 0, image_w * zoom, image_h * zoom)))
        if zoom < 1:
            if self.zoom != self.last_zoom:
                img = cv2.resize(self.composite_image, dsize=None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
                self.canvas_pixmap.setPixmap(QPixmap(np_to_qimage(img)))
                self.canvas_pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.canvas_pixmap.setTransform(t)
        else:
            do_ssaa = zoom >= 2 and (self.rotation not in [0, 90, 180, 270] or zoom not in [2, 4, 8, 16, 32])
            scale_factor = 2 if do_ssaa else 1
            matrix = multiply(
                translate(view_w * scale_factor * 0.5, view_h * scale_factor * 0.5),
                rotate(self.rotation),
                translate(view_x * scale_factor, view_y * scale_factor),
                scale(zoom * scale_factor),
                translate(-image_w / 2, -image_h / 2),
            )[:2]
            inter = cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_CUBIC
            target_buffer = cv2.warpAffine(self.composite_image, matrix, dsize=(view_w * scale_factor, view_h * scale_factor), flags=inter)
            if do_ssaa:
                target_buffer = cv2.resize(target_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            self.canvas_pixmap.setPixmap(QPixmap(np_to_qimage(target_buffer)))
            self.canvas_pixmap.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self.canvas_pixmap.resetTransform()
        self.last_zoom = self.zoom

    def update_background(self):
        bg = self.canvas_state.get_current().background
        if bg.transparent and bg.checker:
            self.canvas_bg_area.setBrush(QBrush(np_to_qimage(make_ideal_checkerboard(bg.checker_brightness, 32))))
        else:
            self.canvas_bg_area.setBrush(QBrush(QColor(*bg.color, 255)))

    def fit_canvas_in_view(self):
        self.position = QPointF()
        size = self.size()
        view_h = size.height()
        view_w = size.width()
        image_h, image_w = self.composite_image.shape[:2]
        if image_h < view_h and image_w < view_w:
            self.zoom = default_zoom_level
        else:
            z_h = find_fitting_zoom_level(view_h, image_h)
            z_w = find_fitting_zoom_level(view_w, image_w)
            self.zoom = min(z_h, z_w)
        self.rotation = 0
        self.apply_transform()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_transform()

    def showEvent(self, event):
        super().showEvent(event)
        if self.first_show:
            self.first_show = False
            self.fit_canvas_in_view()

    def wheelEvent(self, event:QWheelEvent) -> None:
        last_zoom = scroll_zoom_levels[self.zoom]
        if event.angleDelta().y() > 0:
            self.zoom += 1
        else:
            self.zoom -= 1
        self.zoom = max(self.zoom, 0)
        self.zoom = min(self.zoom, len(scroll_zoom_levels)-1)
        next_zoom = scroll_zoom_levels[self.zoom]
        delta_zoom = next_zoom / last_zoom
        # TODO Figure out how to apply this to the mouse/pen position
        self.position = QTransform().scale(delta_zoom, delta_zoom).map(self.position)
        self.apply_transform()

    def keyPressEvent(self, event:QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_R:
            self.fit_canvas_in_view()
        if event.key() == Qt.Key.Key_BracketLeft:
            self.rotation -= 15
            self.rotation %= 360
            self.apply_transform()
        if event.key() == Qt.Key.Key_BracketRight:
            self.rotation += 15
            self.rotation %= 360
            self.apply_transform()

    def mousePressEvent(self, event:QMouseEvent) -> None:
        if event.button() == event.button().LeftButton:
            self.moving_view = True

    def mouseReleaseEvent(self, event:QMouseEvent) -> None:
        if event.button() == event.button().LeftButton:
            self.moving_view = False

    def mouseMoveEvent(self, event:QMouseEvent) -> None:
        if self.moving_view:
            delta = event.position() - self.last_mouse_pos
            delta = QTransform().rotate(-self.rotation).map(delta)
            self.position += delta
            self.apply_transform()
        self.last_mouse_pos = event.position()

class LayerTextItem(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        name_font = 'font: 8pt;'
        small_font = 'font: 7pt;'
        self.layer_name = QLabel()
        self.layer_name.setStyleSheet(name_font)
        self.blend_mode = QLabel()
        self.blend_mode.setStyleSheet(small_font)
        self.opacity = QLabel()
        self.opacity.setStyleSheet(small_font)
        layout = QVBoxLayout()
        layout.addWidget(self.layer_name)
        layout.addWidget(self.blend_mode)
        layout.addWidget(self.opacity)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

class LayerItem(QWidget):
    def __init__(self, is_group, parent=None) -> None:
        super().__init__(parent)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.clip_icon = QLabel()
        self.clip_icon.setStyleSheet('background-color: deeppink;')
        self.clip_icon.setMaximumWidth(3)
        self.visible_checkbox = QCheckBox()
        self.visible_checkbox.setStatusTip('Change layer visibility')
        if is_group:
            self.icon = QPushButton()
            self.icon.setCheckable(True)
            self.icon.clicked.connect(self.on_group_icon_clicked)
        else:
            self.icon = QLabel()
        self.text = LayerTextItem()
        layout = QHBoxLayout()
        layout.addWidget(self.clip_icon)
        layout.addWidget(self.visible_checkbox)
        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(3,0,3,0)
        frame.setLayout(layout)

        self.child_list = LayerList()
        vbox = QVBoxLayout()
        vbox.addWidget(frame)
        vbox.addWidget(self.child_list)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0,0,0,0)
        self.setLayout(vbox)

    def on_group_icon_clicked(self, checked):
        self.icon.setChecked(checked)
        if checked:
            self.icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
            self.child_list.show()
        else:
            self.icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
            self.child_list.hide()

class LayerList(QWidget):
    def __init__(self, is_group=True, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(20 if is_group else 0,0,0,0)
        self.setLayout(layout)

    def add_item(self, widget):
        self.layout().insertWidget(0, widget)

def blend_mode_to_str(blend_mode:layer_data.BlendMode):
    return blend_mode.name.replace('_', ' ').title()

def build_layer_list(layer_list:LayerList, layers:layer_data.GroupLayer):
    for layer in layers:
        layer:layer_data.BaseLayer
        is_group = isinstance(layer, layer_data.GroupLayer)
        item = LayerItem(is_group)
        item.layer_id = layer.id
        item.clip_icon.setVisible(layer.clip)
        item.visible_checkbox.setChecked(layer.visible)
        item.text.layer_name.setText(layer.name)
        item.text.blend_mode.setText(blend_mode_to_str(layer.blend_mode))
        item.text.opacity.setText(f'{int(layer.opacity / 255 * 100)}%')
        layer_list.add_item(item)
        if is_group:
            build_layer_list(item.child_list, layer)
            item.on_group_icon_clicked(layer.folder_open)

class LayerControlPanel(QDockWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

class StatusBar(QStatusBar):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.system_memory_bar = QProgressBar()
        self.system_memory_bar.setMaximumWidth(100)
        self.system_memory_bar.setTextVisible(False)
        self.own_memory_bar = QProgressBar()
        self.own_memory_bar.setMaximumWidth(100)
        self.own_memory_bar.setStyleSheet('background-color: transparent;')
        self.disk_bar = QProgressBar()
        self.disk_bar.setMaximumWidth(100)
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
        self.viewport_tab.currentChanged.connect(self.on_tab_selected)
        self.setCentralWidget(self.viewport_tab)
        statusbar = StatusBar()
        self.setStatusBar(statusbar)

        self.layer_panel_dock = QDockWidget()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_panel_dock)

    def on_tab_close_requested(self, index):
        widget = self.viewport_tab.widget(index)
        widget.close()
        self.viewport_tab.removeTab(index)

    def on_tab_selected(self, index):
        if index == -1:
            self.layer_panel_dock.setWidget(None)
            return
        viewport:Viewport = self.viewport_tab.widget(index)
        canvas_state = viewport.canvas_state.get_current()
        list = LayerList(is_group=False)
        build_layer_list(list, canvas_state.top_level)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(list)
        self.layer_panel_dock.setWidget(scroll_area)

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
            show_error_message(traceback.format_exc())
            return

        canvas_state = CanvasState(
            initial_state=canvas,
            file_path=file_path,
            on_filesystem=True
        )
        composite_image = await full_composite(canvas)
        viewport = Viewport(canvas_state=canvas_state, initial_composite=composite_image)
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
