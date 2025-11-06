import json
from pathlib import Path
import zipfile
import tempfile

import numpy as np
from pyrsistent import *

from . import rle
from .. import blendfuncs, constants, layer_data, util

# Similar in concept to OpenRaster, except JSON for the index and tiles as raw byte arrays.
VERSION = 0

_layer_types = { t.__name__: t for t in [layer_data.PixelLayer, layer_data.GroupLayer] }
_tile_types = { t.__name__: t for t in [layer_data.PixelTile, layer_data.FillTile] }

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
        if isinstance(layer, layer_data.PixelLayer):
            count += len(layer.data)
        if isinstance(layer, layer_data.GroupLayer):
            count += count_tiles(layer.layers)
        if layer.mask is not None:
            count += len(layer.mask.data)
    return count

class _SerializeConfig:
    def __init__(self, zip_file:zipfile.ZipFile, progress_count:int, progress) -> None:
        self.zip_file = zip_file
        self.progress = progress
        self.id_gen = _id_generator()

    def next_id(self):
        return next(self.id_gen)

def _to_rle(array:np.ndarray, is_bytes=False) -> tuple[list[str], list[np.ndarray]]:
    if np.isscalar(array) or len(array.shape) <= 1:
        return [('RAW', array)]
    dims = (array.shape[0] * array.shape[1],)
    data = []
    if not is_bytes:
        array = blendfuncs.to_bytes(array)
    for channel in range(array.shape[2]):
        dst = np.empty_like(array, shape=dims)
        src = array[:,:,channel]
        size = rle.encode(dst, src.reshape(dims))
        if size == 0:
            data.append(('RAW', src))
        else:
            data.append(('RLE', dst[:size]))
    return data

def _read_ndarray(array_info, zfile:zipfile.ZipFile, is_bytes=False):
    path = array_info['path']
    data = array_info['data']
    shape = array_info['shape']
    with zfile.open(path, 'r') as fp:
        if len(shape) <= 1:
            return np.frombuffer(fp.read(), dtype=np.uint8 if is_bytes else blendfuncs.dtype)
        array = np.empty(shape, dtype=np.uint8)
        for data, channel in zip(data, range(array.shape[2])):
            tag = data['tag']
            encoded = np.frombuffer(fp.read(data['size']), dtype=np.uint8)
            if tag == 'RAW':
                array[:,:,channel] = encoded.reshape(shape[:2])[:]
            else:
                dims = (array.shape[0] * array.shape[1],)
                rle.decode(array[:,:,channel].reshape(dims), encoded)
        if is_bytes:
            return array
        else:
            return blendfuncs.from_bytes(array)

def _write_ndarray(array:np.ndarray, config:_SerializeConfig, is_bytes=False):
    data = []
    path = str(Path('data') / config.next_id())
    rle_result = _to_rle(array, is_bytes=is_bytes)
    with config.zip_file.open(path, 'w') as fp:
        for tag, subarray in rle_result:
            data.append({
                'tag': tag,
                'size': subarray.nbytes,
            })
            fp.write(subarray.tobytes())
    return {
        'path': path,
        'data': data,
        'shape': array.shape,
    }

def _read_tile_data(tile_info_list, zfile:zipfile.ZipFile):
    def task():
        tile_map_data = {}
        for tile_info in tile_info_list:
            data = _read_ndarray(tile_info['data'], zfile)
            tile_constructor = _tile_types[tile_info['type']]
            if tile_constructor is layer_data.FillTile:
                tile_args = {}
                tile_args['size'] = tile_info['size']
                tile_args['value'] = data
                tile = tile_constructor(**tile_args)
            else:
                tile = tile_constructor.from_data(data)
            tile_map_data[tuple(tile_info['index'])] = tile
        return pmap(tile_map_data)
    return util.pool.submit(task)

def _write_tile_data(tiles:PMap[layer_data.IVec2, layer_data.PixelTile | layer_data.FillTile], config:_SerializeConfig):
    tile_map_data = []
    for index, tile in tiles.items():
        tile_data = {
            'type': type(tile).__name__,
            'index': index,
        }
        if isinstance(tile, layer_data.FillTile):
            tile_data['size'] = tile.size
            tile_data['data'] = _write_ndarray(tile.value, config)
        else:
            tile_data['data'] = _write_ndarray(tile.data, config)
        tile_map_data.append(tile_data)
        config.progress.update()
    return tile_map_data

def _read_mask(mask, zfile:zipfile.ZipFile):
    if mask is None:
        return None
    return layer_data.Mask(
        position=mask['position'],
        data=_read_tile_data(mask['data'], zfile),
        visible=mask['visible'],
        background_color=_read_ndarray(mask['background_color'], zfile),
    )

def _write_mask(mask:layer_data.Mask | None, config:_SerializeConfig):
    if mask is None:
        return None
    return {
        'position': mask.position,
        'data': _write_tile_data(mask.data, config),
        'visible': mask.visible,
        'background_color': _write_ndarray(mask.background_color, config),
    }

def _read_sublayers(layers, zfile:zipfile.ZipFile):
    layers_list = []
    for sublayer in layers:
        layer_constructor = _layer_types[sublayer['type']]
        layer_args = {
            'name': sublayer['name'],
            'blend_mode': constants.BlendMode(sublayer['blend_mode']),
            'visible': sublayer['visible'],
            'opacity': sublayer['opacity'],
            'lock_alpha': sublayer['lock_alpha'],
            'lock_draw': sublayer['lock_draw'],
            'lock_move': sublayer['lock_move'],
            'lock_all': sublayer['lock_all'],
            'clip': sublayer['clip'],
            'id': sublayer['id'],
            'mask': _read_mask(sublayer['mask'], zfile)
        }
        if layer_constructor is layer_data.PixelLayer:
            layer_args['data'] = _read_tile_data(sublayer['data'], zfile)
            layer_args['position'] = sublayer['position']
        elif layer_constructor is layer_data.GroupLayer:
            layer_args['layers'] = _read_sublayers(sublayer['layers'], zfile)
            layer_args['folder_open'] = sublayer['folder_open']
        layers_list.append(layer_constructor(**layer_args))
    return pvector(layers_list)

def _write_sublayers(layers:layer_data.GroupLayer | list[layer_data.BaseLayer], config):
    layers_list = []
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
        if isinstance(sublayer, layer_data.PixelLayer):
            sublayer_data['data'] = _write_tile_data(sublayer.data, config)
            sublayer_data['position'] = sublayer.position
        elif isinstance(sublayer, layer_data.GroupLayer):
            sublayer_data['layers'] = _write_sublayers(sublayer.layers, config)
            sublayer_data['folder_open'] = sublayer.folder_open
        layers_list.append(sublayer_data)
    return layers_list

def read(file_path:Path) -> layer_data.Canvas:
    with zipfile.ZipFile(file_path, 'r') as zfile:
        canvas_data = json.loads(zfile.read('canvas.json').decode())
        background = canvas_data['background']
        background['color'] = _read_ndarray(background['color'], zfile)
        composite = util.pool.submit(lambda: _read_ndarray(canvas_data['composite'], zfile, is_bytes=True))
        return layer_data.reify_canvas_futures(layer_data.Canvas(
            size=tuple(canvas_data['size']),
            top_level=_read_sublayers(canvas_data['top_level'], zfile),
            background=layer_data.BackgroundSettings(**background),
            selection=_read_mask(canvas_data['selection'], zfile)
        )), composite.result()

def write(canvas:layer_data.Canvas, composite_image:np.ndarray, file_path:Path, progress_callback=None):
    with (tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=file_path.name, delete=False, delete_on_close=False) as temp,
        zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as zfile):
        try:
            config = _SerializeConfig(
                zip_file=zfile,
                progress=util.ProgressCounter(count_tiles(canvas.top_level), progress_callback)
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
                'composite': _write_ndarray(composite_image, config, is_bytes=True)
            }
            config.zip_file.writestr('canvas.json', json.dumps(canvas_data, default=_serialize_numpy_number))
            config.zip_file.close()
            Path(temp.name).rename(file_path)
        except Exception as ex:
            zfile.close()
            temp.close()
            Path(temp.name).unlink(True)
            raise ex
