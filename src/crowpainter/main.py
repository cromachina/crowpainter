import sys
from pathlib import Path

import cv2
import numpy as np
from pyrsistent import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .constants import BlendMode

dtype = np.float64

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def get_overlap_tiles(dst, src, offset):
    ox, oy = offset
    dx, dy = dst.shape[:2]
    sx, sy = src.shape[:2]
    d_min_x = clamp(0, dx, ox)
    d_min_y = clamp(0, dy, oy)
    d_max_x = clamp(0, dx, ox + sx)
    d_max_y = clamp(0, dy, oy + sy)
    s_min_x = clamp(0, sx, -ox)
    s_min_y = clamp(0, sy, -oy)
    s_max_x = clamp(0, sx, dx - ox)
    s_max_y = clamp(0, sy, dy - oy)
    return dst[d_min_x:d_max_x, d_min_y:d_max_y], src[s_min_x:s_max_x, s_min_y:s_max_y]

def blit(dst, src, offset):
    dst, src = get_overlap_tiles(dst, src, offset)
    np.copyto(dst, src)

def generate_tiles(size, tile_size):
    height, width = size
    tile_height, tile_width = tile_size
    y = 0
    while y < height:
        x = 0
        while x < width:
            size_y = min(tile_height, height - y)
            size_x = min(tile_width, width - x)
            yield ((size_y, size_x), (y, x))
            x += tile_width
        y += tile_height

TILE_SIZE = (256, 256)
Vec2 = tuple[int, int]

class Tile(PClass):
    color = field(initial=np.zeros(TILE_SIZE + (3,), dtype=dtype))
    alpha = field(initial=np.zeros(TILE_SIZE + (1,), dtype=dtype))

class MaskTile(PClass):
    alpha = field(initial=np.zeros(TILE_SIZE + (1,), dtype=dtype))

class Mask(PClass):
    visible = field(initial=True)
    position = field(initial=(0.0, 0.0))
    tiles = field(initial=pmap())

class BaseLayer(PClass):
    name = field(initial="")
    blend_mode = field(initial=BlendMode.NORMAL)
    opacity = field(initial=255)
    lock_alpha = field(initial=False)
    lock_draw = field(initial=False)
    lock_move = field(initial=False)
    lock_all = field(initial=False)
    clip = field(initial=False)
    mask = field(initial=None)

class PixelLayer(BaseLayer):
    position = field(initial=(0.0, 0.0))
    tiles = field(initial=pmap())

class FillLayer(BaseLayer):
    color = field(initial=np.ones(4, dtype=dtype))

class GroupLayer(BaseLayer):
    layers = field(initial=pvector())

class Selection(PClass):
    mask_position = field(initial=(0.0, 0.0))
    mask_tiles = field(initial=pmap())

class Canvas(PClass):
    selection = field(type=optional(Selection), initial=None)
    top_level = field(initial=GroupLayer())
    size = field(initial=(0, 0))

class CanvasState():
    def __init__(self, initial_state, file_name, file_dir=None):
        self.file_name = file_name
        self.file_dir = file_dir
        self.current_index = 0
        self.saved_state = initial_state
        self.states = pdeque([initial_state])

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

class Viewport(QGraphicsView):
    '''Display a canvas and handle input events for it.'''
    def __init__(self, canvas_state:CanvasState, parent=None):
        super().__init__(parent)
        self.position = QPointF(0.0, 0.0)
        self.zoom = 1.0
        self.last_zoom = self.zoom
        self.rotation = 0.0
        self.canvas_state:CanvasState = canvas_state
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

        self.reset_viewport()

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

    def reset_viewport(self):
        self.scene().clear()
        canvas = self.canvas_state.get_current()
        h, w = canvas.size
        self.image = canvas.top_level.layers[0].tiles

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

def pixel_data_to_tiles(data:np.ndarray):
    tiles = []
    for (size, offset) in generate_tiles(data.shape[:2], TILE_SIZE):
        tile = np.zeros(shape=size + data.shape[2:])
        blit(tile, data, -np.array(offset))
        tiles.append((offset, tile))
    return tiles

def scalar_to_tiles(value, shape):
    tiles = []
    for (size, offset) in generate_tiles(shape, TILE_SIZE):
        tiles.append((offset, value))
    return tiles

def color_alpha_to_tiles(color:np.ndarray, alpha):
    color_tiles = pixel_data_to_tiles(color)
    if np.isscalar(alpha):
        alpha_tiles = scalar_to_tiles(alpha, color.shape[:2])
    else:
        alpha_tiles = pixel_data_to_tiles(alpha)
    tiles = {}
    for (offset, c), (_, a) in zip(color_tiles, alpha_tiles):
        tiles[offset] = Tile(color=c, alpha=a)
    return pmap(tiles)

def open_file(file_name:Path):
    data = cv2.imread(str(file_name), cv2.IMREAD_UNCHANGED)
    if data is None:
        return None

    if data.shape[2] == 3:
        color = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(dtype) / 255.0
        alpha = 1.0
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA).astype(dtype) / 255.0
        color, alpha = np.split(data, [2], axis=2)
    data = None

    p = PixelLayer(name="Layer1", tiles=color_alpha_to_tiles(color, alpha))
    top_level = GroupLayer(layers=pvector([p]))
    canvas = Canvas(top_level=top_level, size=color.shape[:2])
    return CanvasState(canvas, file_name.name, file_name.parent)

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

    def on_open(self):
        files, filter = QFileDialog.getOpenFileNames(self, caption='Open', filter='All files (*.*)', dir='.')
        for file in files:
            # TODO check if file is already open and ask to reopen.

            canvas_state = open_file(Path(file))
            if canvas_state is None:
                QErrorMessage(self).showMessage(f'Could not open file: {file}')
            else:
                viewport = Viewport(canvas_state)
                self.viewports.append(viewport)
                self.viewport_tab.addTab(viewport, viewport.canvas_state.file_name)

    def create_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu('&File')

        new_action = QAction(text='&New ...', parent=self)
        new_action.triggered.connect(self.on_new)
        file_menu.addAction(new_action)

        open_action = QAction(text='&Open ...', parent=self)
        open_action.triggered.connect(self.on_open)
        file_menu.addAction(open_action)

def main():
    app = QApplication(sys.argv)

    widget = MainWindow()
    widget.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()