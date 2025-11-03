import sys
from pathlib import Path
import asyncio
from collections import deque
import logging
import traceback
import time
import threading
import contextlib

import cv2
import numpy as np
from pyrsistent import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import PySide6.QtAsyncio as QtAsyncio
import psutil
from pyqttoast import Toast, ToastPreset

from . import layer_data, composite, util, blendfuncs
from .constants import *
from .file_io import psd, image, native

@contextlib.contextmanager
def timeit(message):
    start = time.monotonic()
    try:
        yield
    finally:
        logging.info(f'{message}: {time.monotonic() - start}')

class ExtendedInfoMessage(QDialog):
    def __init__(self, parent=None, title='', text=''):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
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

def dispatch_to_main_thread(callback):
    timer = QTimer()
    timer.moveToThread(QApplication.instance().thread())
    def wrapper():
        try:
            callback()
        finally:
            timer.deleteLater()
    timer.timeout.connect(wrapper)
    QMetaObject.invokeMethod(timer, 'start', Q_ARG(int, 0))

def error_toast(text):
    toast = Toast()
    toast.setDuration(5000)
    toast.setText(text)
    toast.applyPreset(ToastPreset.ERROR_DARK)
    toast.setFadeInDuration(0)
    toast.setFadeOutDuration(0)
    toast.show()

def show_error_toast(text):
    dispatch_to_main_thread(lambda: error_toast(text))

def show_error_message(text):
    dispatch_to_main_thread(lambda: ExtendedInfoMessage(title='Error', text=text).exec())

class CanvasState():
    def __init__(self, initial_state:layer_data.Canvas, file_path:Path):
        self.file_path = file_path
        self.current_index = 0
        self.saved_state = initial_state
        self.selected_objects = []
        self.states = deque([initial_state])

    def get_current(self):
        return self.states[self.current_index]

    def append(self, state, state_limit:int|None=None):
        self.states = self.states[:self.current_index + 1]
        self.states.append(state)
        if state_limit is not None and len(self.states) > state_limit:
            delta = len(self.states) - state_limit
            if delta == 1:
                self.states.popleft()
            else:
                self.states = self.states[delta:]
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

    def set_saved(self, state=None):
        if state is None:
            state = self.get_current()
        self.saved_state = state

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

def do_nothing(*args, **kwargs):
    pass

async def parallel_composite(canvas:layer_data.Canvas, size:layer_data.IVec2=None, offset:layer_data.IVec2=(0, 0), progress_callback=do_nothing):
    if size is None:
        size = canvas.size
    if canvas.background.transparent:
        backdrop = np.zeros(size + (4,), dtype=blendfuncs.dtype)
    else:
        backdrop = np.empty(size + (4,), dtype=blendfuncs.dtype)
        util.get_color(backdrop)[:] = blendfuncs.from_bytes(np.uint8(canvas.background.color))
        util.get_alpha(backdrop)[:] = blendfuncs.get_max()

    lock = threading.Lock()
    tiles = list(util.generate_tiles(size, TILE_SIZE))
    progress_count = 0
    progress_total = len(tiles)

    def tile_task(size, offset):
        nonlocal progress_count
        tile = util.get_overlap_view(backdrop, size, offset)
        composite.composite(canvas.top_level, offset, tile)
        with lock:
            progress_count += 1
            progress_callback(progress_count / progress_total)

    tasks = []
    for (tile_size, tile_offset) in tiles:
        tasks.append(util.peval(tile_task, tile_size, tile_offset))
    await asyncio.gather(*tasks)

    def final():
        nonlocal backdrop
        color = util.get_color(backdrop)
        alpha = util.get_alpha(backdrop)
        blendfuncs.clip_divide(color, alpha, out=color)
        return blendfuncs.to_bytes(backdrop)
    return await util.peval(final)

def make_checkerboard_texture(check_a, check_b, size):
    check_a = util.clamp(0, 255, check_a)
    check_b = util.clamp(0, 255, check_b)
    arr = np.array([check_a, check_b, check_b, check_a], dtype=np.uint8).reshape((2,2,1))
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST).reshape((size, size, 1))

def make_ideal_checkerboard(value, size):
    a = value * 0.9
    b = value + 0.1
    return make_checkerboard_texture(a * 255, b * 255, size)

class Viewport(QGraphicsView):
    '''Display a canvas and handle input events for it.'''
    def __init__(self, parent=None, canvas_state:CanvasState=None, initial_composite=None):
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
            color = blendfuncs.to_bytes(bg.color)
            self.canvas_bg_area.setBrush(QBrush(QColor(*color, 255)))

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
    def __init__(self, parent=None, is_group=True) -> None:
        super().__init__(parent)

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
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.addWidget(self.clip_icon)
        layout.addWidget(self.visible_checkbox)
        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(3,0,3,0)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(layout)

        self.child_list = LayerList(frame)
        vbox = QVBoxLayout(self)
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
    def __init__(self, parent=None, is_group=True) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(20 if is_group else 0,0,0,0)
        self.setLayout(layout)

    def addWidget(self, widget):
        self.layout().insertWidget(0, widget)

def blend_mode_to_str(blend_mode:layer_data.BlendMode):
    return blend_mode.name.replace('_', ' ').title()

def build_layer_list(layer_list:LayerList, layers:layer_data.GroupLayer):
    for layer in layers:
        layer:layer_data.BaseLayer
        is_group = isinstance(layer, layer_data.GroupLayer)
        item = LayerItem(layer_list, is_group)
        item.layer_id = layer.id
        item.clip_icon.setVisible(layer.clip)
        item.visible_checkbox.setChecked(layer.visible)
        item.text.layer_name.setText(layer.name)
        item.text.blend_mode.setText(blend_mode_to_str(layer.blend_mode))
        item.text.opacity.setText(f'{int(float(layer.opacity) / blendfuncs.get_max() * 100)}%')
        layer_list.addWidget(item)
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

image_types = [
    ('Crowpainter', 'crow'),
    ('PNG', 'png'),
    ('Photoshop', 'psd', 'psb'),
    ('JPEG', 'jpg', 'jpeg', 'jpe'),
    ('WebP', 'webp'),
]

def make_filter(name, exts):
    exts = ' '.join(f'*.{ext}' for ext in exts)
    return f'{name} ({exts}) ({exts})'

all_images_filter = [make_filter('Image Files', [ext for image_type in image_types for ext in image_type[1:]])]
sub_images_filter = [make_filter(image_type[0], image_type[1:]) for image_type in image_types]
image_filter = ';;'.join(all_images_filter + sub_images_filter)

class JpgWriterDialog(QDialog):
    pass

class SignalProgressBar(QProgressBar):
    update_progress = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_progress.connect(lambda value: self.setValue(int(value * 100)))

    # Normalized value (0 to 1); Thread safe.
    def update_value(self, value):
        self.update_progress.emit(value)

class ProgressDialog(QDialog):
    def __init__(self, parent=None, text='', cancellable=False):
        super().__init__(parent)
        self.setMinimumWidth(250)
        self.text = QLabel(text=text)
        self.progress_bar = SignalProgressBar()
        self.cancel_button = QDialogButtonBox(standardButtons=QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.rejected.connect(self.reject)
        self.cancel_button.setVisible(cancellable)
        layout = QVBoxLayout()
        layout.addWidget(self.text)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)

    def update_value(self, value):
        self.progress_bar.update_value(value)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings()
        self.setWindowTitle('CrowPainter')
        self.setGeometry(self.settings.value('window/geometry', type=QRect, defaultValue=QRect(0, 0, 1000, 1000)))

        self.viewport_tab = QTabWidget(self)
        self.viewport_tab.setTabsClosable(True)
        self.viewport_tab.tabCloseRequested.connect(self.on_tab_close_requested)
        self.viewport_tab.currentChanged.connect(self.on_tab_selected)
        self.setCentralWidget(self.viewport_tab)
        self.setStatusBar(StatusBar(self))

        self.layer_panel_dock = QDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_panel_dock)
        self.scroll_area = QScrollArea(self.layer_panel_dock)
        self.scroll_area.setWidgetResizable(True)
        self.layer_panel_dock.setWidget(self.scroll_area)

        self.create_menus()
        self.open_lock = asyncio.Lock()

    def closeEvent(self, event):
        self.settings.setValue('window/geometry', self.geometry())
        return super().closeEvent(event)

    def on_tab_close_requested(self, index):
        widget = self.viewport_tab.widget(index)
        widget.deleteLater()

    def on_tab_selected(self, index):
        if self.scroll_area.widget() is not None:
            self.scroll_area.widget().deleteLater()
        if index == -1:
            return
        viewport:Viewport = self.viewport_tab.widget(index)
        canvas_state = viewport.canvas_state.get_current()
        layer_list = LayerList(self.scroll_area, is_group=False)
        build_layer_list(layer_list, canvas_state.top_level)
        self.scroll_area.setWidget(layer_list)

    def on_new(self):
        pass

    def on_open(self):
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter(image_filter)
        dialog.setDirectory(self.settings.value('lastdir', '.'))
        dialog.setModal(True)
        dialog.filesSelected.connect(lambda files: asyncio.ensure_future(self.open_files(files)))
        dialog.show()

    async def open_files(self, files):
        for file_path in files:
            # TODO check if file is already open and ask to reopen without saving.
            await self.open(file_path)

    def on_save(self):
        pass

    @contextlib.contextmanager
    def progress_status_bar(self):
        try:
            prog = SignalProgressBar()
            self.statusBar().addWidget(prog)
            yield prog
        finally:
            self.statusBar().removeWidget(prog)
            prog.deleteLater()

    @contextlib.contextmanager
    def progress_dialog(self, text):
        try:
            prog = ProgressDialog(self, text)
            prog.show()
            yield prog
        finally:
            prog.close()
            prog.deleteLater()

    def on_save_as(self):
        viewport:Viewport = self.viewport_tab.currentWidget()
        if viewport is not None:
            dialog = QFileDialog(self)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setNameFilter(image_filter)
            dialog.setDirectory(self.settings.value('lastdir', '.', type=str))
            dialog.setModal(True)
            dialog.fileSelected.connect(lambda file: asyncio.ensure_future(self._save_file(file, viewport)))
            dialog.show()

    async def _save_file(self, file_path, viewport, modal=False):
        if file_path == '':
            return
        current_canvas = viewport.canvas_state.get_current()
        current_composite = viewport.composite_image.copy()
        self.settings.setValue('lastdir', str(Path(file_path).parent))
        if modal:
            progress = self.progress_dialog(f'Saving {current_canvas.name}')
        else:
            progress = self.progress_status_bar()
        with progress as progress_widget, timeit(f'save {file_path}'):
            result = await util.peval(self.save, current_canvas, current_composite, file_path, progress_widget.update_value)
        viewport.canvas_state.set_saved(current_canvas)

    def on_close(self):
        pass

    async def open(self, file_path):
        file_path = Path(file_path)
        self.settings.setValue('lastdir', str(file_path.parent))
        current_task = None
        with self.progress_dialog(text=f'Opening {file_path.name}') as progress:
            def cancel_open():
                if current_task is not None:
                    current_task.cancel('Open cancelled')
            progress.rejected.connect(cancel_open)
            try:
                with timeit(f'open {file_path}'):
                    with timeit(f'open file read {file_path}'):
                        current_task = asyncio.create_task(open_file(file_path))
                        canvas = await current_task
                    composite_image = None
                    if isinstance(canvas, tuple):
                        canvas, composite_image = canvas
                    canvas_state = CanvasState(
                        initial_state=canvas,
                        file_path=file_path,
                    )
                    if composite_image is None:
                        with timeit(f'open composite {file_path}'):
                            current_task = asyncio.create_task(parallel_composite(canvas, progress_callback=progress.update_value))
                            composite_image = await current_task
                    viewport = Viewport(canvas_state=canvas_state, initial_composite=composite_image)
                    index = self.viewport_tab.addTab(viewport, viewport.canvas_state.file_path.name)
                    self.viewport_tab.setCurrentIndex(index)
            except asyncio.CancelledError:
                pass

    def save(self, canvas:layer_data.Canvas, composite:np.ndarray, file_path, progress_callback):
        file_path = Path(file_path)
        file_type = file_path.suffix
        if file_type == '.crow':
            native.write(canvas, composite, file_path, progress_callback)
        if file_type in ['.webp', '.png']:
            image.write(composite, file_path, [])
        return True

    def create_menu_action(self, menu:QMenu, text:str, callback, enabled=True):
        action = QAction(text=text, parent=self)
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def create_menu_widget_toggle_action(self, menu, widget, text):
        def callback():
            widget.setHidden(not widget.isHidden())
            set_menu_item_text()
        action = self.create_menu_action(menu, text, callback)
        def set_menu_item_text():
            prefix = 'Show' if widget.isHidden() else 'Hide'
            action.setText(f'{prefix} {text}')
        set_menu_item_text()
        return action

    def create_menus(self):
        menu = self.menuBar()
        menu_file = menu.addMenu('&File')

        self.action_new = self.create_menu_action(menu_file, '&New ...', self.on_new)
        self.action_open = self.create_menu_action(menu_file, '&Open ...', self.on_open)
        self.action_save = self.create_menu_action(menu_file, '&Save ...', self.on_save)
        self.action_save_as = self.create_menu_action(menu_file, 'Save &As ...', self.on_save_as)
        self.action_close = self.create_menu_action(menu_file, '&Close ...', self.on_close)
        self.action_mem_stats = self.create_menu_action(menu_file, '&Print Memory Stats', util.update_memory_tracking)

        menu_edit = menu.addMenu('&Edit')
        menu_canvas = menu.addMenu('&Canvas')
        menu_layer = menu.addMenu('&Layer')
        menu_selection = menu.addMenu('&Selection')
        self.menu_view = menu.addMenu('&View')
        menu_window = menu.addMenu('&Window')

        self.action_show_layer_panel = self.create_menu_widget_toggle_action(menu_window, self.layer_panel_dock, '&Layer Panel')

async def open_file(file_path:Path):
    file_type = file_path.suffix
    if file_type in ['.psd', '.psb']:
        return await util.peval(psd.read, file_path)
    elif file_type == '.crow':
        return await util.peval(native.read, file_path)
    else:
        return await util.peval(image.read, file_path)

def init_logging():
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)s] %(message)s',
        level=logging.INFO
    )

# Exceptions that occur in any thread, outside of the async event loop.
def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    message = "".join(traceback.format_exception_only(exc_type, exc_value))
    logging.exception(tb)
    show_error_toast(message)

# Exceptions that occur in the async event loop.
def async_exception_handler(context):
    tb = context.get('traceback')
    message = repr(context.get('exception'))
    logging.exception(tb)
    show_error_toast(message)

async def async_main():
    init_logging()
    asyncio.get_event_loop().set_exception_handler(async_exception_handler)
    main_window = MainWindow()
    main_window.show()
    Toast.setPositionRelativeToWidget(main_window)
    sys.excepthook = excepthook

def main():
    app = QApplication(sys.argv)
    app.setOrganizationName('crowpainter')
    app.setApplicationName('crowpainter')
    app.setStyle('fusion')
    QtAsyncio.run(async_main())

# Lets this run in the vscode debugger.
if __name__ == '__main__':
    main()
