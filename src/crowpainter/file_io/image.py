from pathlib import Path

import cv2

from ..layer_data import *

def read(file_path:Path) -> Canvas:
    data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise Exception('Error opening file', file_path)

    if data.shape[2] == 3:
        color = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        alpha = 255
        bg = BackgroundSettings()
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
        color, alpha = np.split(data, [3], axis=2)
        bg = BackgroundSettings(transparent=True)
    data = None

    color_tiles, alpha_tiles = color_alpha_to_tiles(color, alpha)

    layer = PixelLayer(
        name="Layer1",
        color=color_tiles,
        alpha=alpha_tiles,
    )
    return Canvas(
        top_level=GroupLayer(layers=pvector([layer])),
        size=color.shape[:2],
        background=bg,
    )

# TODO extra params: jpg quality, png compression, etc.
def write(canvas:Canvas, file_name:Path):
    raise NotImplementedError()
