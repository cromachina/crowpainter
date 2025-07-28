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

def count_tiles(layers):
    count = 0
    for layer in layers:
        if isinstance(layer, PixelLayer):
            count += len(layer.color) + len(layer.alpha)
        if isinstance(layer, GroupLayer):
            count += count_tiles(layer.layers)
        if layer.mask is not None:
            count += len(layer.mask.alpha)
    return count

class _SerializeConfig():
    def __init__(self, zip_file:zipfile.ZipFile, progress_count:int, progress_callback) -> None:
        self.zip_file = zip_file
        self.current_count = 0
        self.progress_count = progress_count
        self.progress_callback = progress_callback
        self.id_gen = _id_generator()

    def progress_update(self):
        if self.progress_callback is not None:
            self.current_count += 1
            self.progress_callback(self.current_count / self.progress_count)

    def next_id(self):
        return next(self.id_gen)

def _read_ndarray(data_path, zip:zipfile.ZipFile):
    with zip.open(data_path, 'r') as fp:
        return util.to_storage_dtype(np.load(fp, allow_pickle=False))

def _write_ndarray(array:np.ndarray, config:_SerializeConfig):
    data_path = config.next_id()
    with config.zip_file.open(data_path, 'w') as fp:
        np.save(fp, array, allow_pickle=False)
    return data_path

def _read_tile_data(tiles, zip:zipfile.ZipFile):
    tile_map_data = {}
    for tile in tiles:
        tile_constructor = _tile_types[tile['type']]
        if tile_constructor is FillTile:
            tile_args = {}
            tile_args['size'] = tile['size']
            tile_args['value'] = _read_ndarray(tile['value'], zip)
            tile_data = tile_constructor(**tile_args)
        else:
            tile_data = tile_constructor.from_data(_read_ndarray(tile['data'], zip))
        tile_map_data[tuple(tile['index'])] = tile_data
    return pmap(tile_map_data)

def _write_tile_data(tiles:PMap[IVec2, BaseArrayTile | FillTile], config:_SerializeConfig):
    tile_map_data = []
    for index, tile in tiles.items():
        tile_data = {
            'type': type(tile).__name__,
            'index': index,
        }
        if isinstance(tile, FillTile):
            tile_data['size'] = tile.size
            tile_data['value'] = _write_ndarray(tile.value, config)
        else:
            tile_data['data'] = _write_ndarray(tile.data, config)
        tile_map_data.append(tile_data)
        config.progress_update()
    return tile_map_data

def _read_mask(mask, zip:zipfile.ZipFile):
    if mask is None:
        return None
    return Mask(
        position=mask['position'],
        alpha=_read_tile_data(mask['alpha'], zip),
        visible=mask['visible'],
        background_color=_read_ndarray(mask['background_color'], zip),
    )

def _write_mask(mask:Mask | None, config:_SerializeConfig):
    if mask is None:
        return None
    return {
        'position': mask.position,
        'alpha': _write_tile_data(mask.alpha, config),
        'visible': mask.visible,
        'background_color': _write_ndarray(mask.background_color, config),
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

def _write_sublayers(layers:GroupLayer | list[BaseLayer], config):
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
            'mask': _write_mask(sublayer.mask, config)
        }
        if isinstance(sublayer, PixelLayer):
            sublayer_data['color'] = _write_tile_data(sublayer.color, config)
            sublayer_data['alpha'] = _write_tile_data(sublayer.alpha, config)
            sublayer_data['position'] = sublayer.position
        elif isinstance(sublayer, GroupLayer):
            sublayer_data['layers'] = _write_sublayers(sublayer.layers, config)
            sublayer_data['folder_open'] = sublayer.folder_open
        layers_data.append(sublayer_data)
    return layers_data

def read(file_path:Path) -> Canvas:
    with zipfile.ZipFile(file_path, 'r') as zip:
        canvas_data = json.loads(zip.read('canvas').decode())
        background = canvas_data['background']
        background['color'] = _read_ndarray(background['color'], zip)
        return Canvas(
            size = tuple(canvas_data['size']),
            top_level=_read_sublayers(canvas_data['top_level'], zip),
            background=BackgroundSettings(**background),
            selection=_read_mask(canvas_data['selection'], zip)
        )

def write(canvas:Canvas, file_path:Path, progress_callback=None):
    with tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=file_path.name, delete=False, delete_on_close=False) as temp:
        try:
            config = _SerializeConfig(
                zip_file=zipfile.ZipFile(temp, 'w', compresslevel=5, compression=zipfile.ZIP_DEFLATED),
                progress_count=count_tiles(canvas.top_level),
                progress_callback=progress_callback,
            )
            canvas_data = {
                'version': VERSION,
                'size': canvas.size,
                'top_level': _write_sublayers(canvas.top_level, config),
                'background': {
                    'color': _write_ndarray(canvas.background.color, config),
                    'transparent': canvas.background.transparent,
                    'checker': canvas.background.checker,
                    'checker_brightness': canvas.background.checker_brightness,
                },
                'selection': _write_mask(canvas.selection, config),
            }
            config.zip_file.writestr('canvas', json.dumps(canvas_data, default=_serialize_numpy_number))
            config.zip_file.close()
            Path(temp.name).rename(file_path)
        except Exception as ex:
            temp.close()
            Path(temp.name).unlink(True)
            raise ex
