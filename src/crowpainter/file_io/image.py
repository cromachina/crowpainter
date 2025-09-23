from pathlib import Path

import cv2

from ..layer_data import *

def read(file_path:Path) -> Canvas:
    data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise Exception('Error opening file', file_path)
    if data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)
        bg = BackgroundSettings()
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
        bg = BackgroundSettings(transparent=True)
    data = util.to_storage_dtype(data)
    tiles = pixel_data_to_tiles(data)
    layer = PixelLayer(
        name="Layer1",
        color=tiles,
    )
    return Canvas(
        top_level=GroupLayer(layers=pvector([layer])),
        size=data.shape[:2],
        background=bg,
    )

def is_grayscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return (r == g).all() and (g == b).all()

def write(image:np.ndarray, file_path:Path, params):
    if (image[:,:,3] == 255).all():
        if is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(file_path), image, params)
