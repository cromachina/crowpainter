import json
from pathlib import Path
import zipfile
import tempfile

import numpy as np

from ..layer_data import *

# Similar in concept to OpenRaster, except JSON for the index and tiles as raw byte arrays.
VERSION = 0

_layer_types = { t.__name__: t for t in [PixelLayer, GroupLayer] }
_tile_types = { t.__name__: t for t in [ColorTile, AlphaTile, FillTile] }

def _id_generator(init=0):
    while True:
        yield str(init)
        init += 1

def _serialize_numpy_number(obj):
    if  np.issubdtype(np.array(obj).dtype.type, np.integer):
        return obj.item()
    elif  np.issubdtype(np.array(obj).dtype.type, np.floating):
        return obj.item()
    else:
        raise TypeError(obj)

def _read_ndarray(array, zip:zipfile.ZipFile):
    bytes = zip.read(array['data_path'])
    return np.frombuffer(bytes, dtype=STORAGE_DTYPE).reshape(array['shape'])

def _write_ndarray(array:np.ndarray, zip:zipfile.ZipFile, id_gen):
    data_path = next(id_gen)
    array_data = {
        'shape': array.shape,
        'data_path': data_path
    }
    zip.writestr(data_path, array.tobytes())
    return array_data

def _read_tile_data(tiles, zip:zipfile.ZipFile):
    tile_map_data = {}
    for tile in tiles:
        tile_constructor = _tile_types[tile['type']]
        if tile_constructor is FillTile:
            tile_args = {}
            tile_args['size'] = tile['size']
            tile_args['value'] = tile['value']
            tile_data = tile_constructor(**tile_args)
        else:
            tile_data = tile_constructor.from_data(_read_ndarray(tile['data'], zip))
        tile_map_data[tuple(tile['index'])] = tile_data
    return pmap(tile_map_data)

def _write_tile_data(tiles:PMap[IVec2, BaseArrayTile | FillTile], zip:zipfile.ZipFile, id_gen):
    tile_map_data = []
    for index, tile in tiles.items():
        tile_data = {
            'type': type(tile).__name__,
            'index': index,
        }
        if isinstance(tile, FillTile):
            tile_data['size'] = tile.size
            tile_data['value'] = tile.value
        else:
            tile_data['data'] = _write_ndarray(tile.data, zip, id_gen)
        tile_map_data.append(tile_data)
    return tile_map_data

def _read_mask(mask, zip:zipfile.ZipFile):
    if mask is None:
        return None
    return Mask(
        position=mask['position'],
        alpha=_read_tile_data(mask['alpha'], zip),
        visible=mask['visible'],
        background_color=mask['background_color'],
    )

def _write_mask(mask:Mask | None, zip:zipfile.ZipFile, id_gen):
    if mask is None:
        return None
    return {
        'position': mask.position,
        'alpha': _write_tile_data(mask.alpha, zip, id_gen),
        'visible': mask.visible,
        'background_color': mask.background_color,
    }

def _read_sublayers(layers, zip:zipfile.ZipFile):
    layer_data = []
    for sublayer in layers:
        layer_constructor = _layer_types[sublayer['type']]
        layer_args = {
            'name': sublayer['name'],
            'blend_mode': BlendMode(sublayer['blend_mode']),
            'visible': sublayer['visible'],
            'opacity': sublayer['opacity'],
            'lock_alpha': sublayer['lock_alpha'],
            'lock_draw': sublayer['lock_draw'],
            'lock_move': sublayer['lock_move'],
            'lock_all': sublayer['lock_all'],
            'clip': sublayer['clip'],
            'id': sublayer['id'],
            'mask': _read_mask(sublayer['mask'], zip)
        }
        if layer_constructor is PixelLayer:
            layer_args['color'] = _read_tile_data(sublayer['color'], zip)
            layer_args['alpha'] = _read_tile_data(sublayer['alpha'], zip)
            layer_args['position'] = sublayer['position']
        elif layer_constructor is GroupLayer:
            layer_args['layers'] = _read_sublayers(sublayer['layers'], zip)
            layer_args['folder_open'] = sublayer['folder_open']
        layer_data.append(layer_constructor(**layer_args))
    return pvector(layer_data)

def _write_sublayers(layers:GroupLayer | list[BaseLayer], zip:zipfile.ZipFile, id_gen):
    layers_data = []
    for sublayer in layers:
        sublayer_data = {
            'type': type(sublayer).__name__,
            'name': sublayer.name,
            'blend_mode': sublayer.blend_mode.value,
            'visible': sublayer.visible,
            'opacity': sublayer.opacity,
            'lock_alpha': sublayer.lock_alpha,
            'lock_draw': sublayer.lock_draw,
            'lock_move': sublayer.lock_move,
            'lock_all': sublayer.lock_all,
            'clip': sublayer.clip,
            'id': sublayer.id,
            'mask': _write_mask(sublayer.mask, zip, id_gen)
        }
        if isinstance(sublayer, PixelLayer):
            sublayer_data['color'] = _write_tile_data(sublayer.color, zip, id_gen)
            sublayer_data['alpha'] = _write_tile_data(sublayer.alpha, zip, id_gen)
            sublayer_data['position'] = sublayer.position
        elif isinstance(sublayer, GroupLayer):
            sublayer_data['layers'] = _write_sublayers(sublayer.layers, zip, id_gen)
            sublayer_data['folder_open'] = sublayer.folder_open
        layers_data.append(sublayer_data)
    return layers_data

def read(file_path:Path) -> Canvas:
    with zipfile.ZipFile(file_path, 'r') as zip:
        canvas_data = json.loads(zip.read('canvas').decode())
        return Canvas(
            size = tuple(canvas_data['size']),
            top_level=_read_sublayers(canvas_data['top_level'], zip),
            background=BackgroundSettings(**canvas_data['background']),
            selection=_read_mask(canvas_data['selection'], zip)
        )

def write(canvas:Canvas, file_path:Path):
    with tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=file_path.name, delete=False, delete_on_close=False) as temp:
        try:
            zip = zipfile.ZipFile(temp, 'w', compresslevel=9, compression=zipfile.ZIP_DEFLATED)
            id_gen = _id_generator()
            canvas_data = {
                'version': VERSION,
                'size': canvas.size,
                'top_level': _write_sublayers(canvas.top_level, zip, id_gen),
                'background': {
                    'color': canvas.background.color,
                    'transparent': canvas.background.transparent,
                    'checker': canvas.background.checker,
                    'checker_brightness': canvas.background.checker_brightness,
                },
                'selection': _write_mask(canvas.selection, zip, id_gen),
            }
            zip.writestr('canvas', json.dumps(canvas_data, default=_serialize_numpy_number))
            zip.close()
            Path(temp.name).rename(file_path)
        except Exception as ex:
            temp.close()
            Path(temp.name).unlink(True)
            raise ex
