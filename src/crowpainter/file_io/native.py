import json
from pathlib import Path
import zipfile
import tempfile
import itertools

import numpy as np

from ..layer_data import *
from . import rle_native

# Similar in concept to OpenRaster, except JSON for the index and tiles as raw byte arrays.
VERSION = 0

_layer_types = { t.__name__: t for t in [PixelLayer, GroupLayer] }
_tile_types = { t.__name__: t for t in [ColorTile, AlphaTile, FillTile] }

def decode(dst:np.ndarray, src:np.ndarray):
    return rle_native.decode(dst, src)

def encode(dst:np.ndarray, src:np.ndarray):
    return rle_native.encode(dst, src)

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
            count += len(layer.color)
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

def _to_rle(array:np.ndarray) -> (list[str], list[np.ndarray]):
    if np.isscalar(array) or len(array.shape) <= 1:
        return [('RAW', array)]
    dims = (array.shape[0] * array.shape[1],)
    data = []
    for channel in range(array.shape[2]):
        dst = np.empty_like(array, shape=dims)
        src = array[:,:,channel]
        size = encode(dst, src.reshape(dims))
        if size == 0:
            data.append(('RAW', src))
        else:
            data.append(('RLE', dst[:size]))
    return data

def _read_ndarray(array_info, zfile:zipfile.ZipFile):
    path = array_info['path']
    data = array_info['data']
    shape = array_info['shape']
    with zfile.open(path, 'r') as fp:
        if len(shape) <= 1:
            return np.frombuffer(fp.read(), dtype=STORAGE_DTYPE)
        array = np.empty(shape, dtype=STORAGE_DTYPE)
        for data, channel in zip(data, range(array.shape[2])):
            tag = data['tag']
            encoded = np.frombuffer(fp.read(data['size']), dtype=STORAGE_DTYPE)
            if tag == 'RAW':
                array[:,:,channel] = encoded.reshape(shape[:2])[:]
            else:
                dims = (array.shape[0] * array.shape[1],)
                decode(array[:,:,channel].reshape(dims), encoded)
        return array

def _write_ndarray(array:np.ndarray, config:_SerializeConfig):
    data = []
    path = str(Path('data') / config.next_id())
    rle_result = _to_rle(array)
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

def _read_tile_data(tile_info_list, tile_data):
    tile_map_data = {}
    for tile_info in tile_info_list:
        tile_constructor = _tile_types[tile_info['type']]
        if tile_constructor is FillTile:
            tile_args = {}
            tile_args['size'] = tile_info['size']
            tile_args['value'] = tile_data[tile_info['data']['path']]
            tile = tile_constructor(**tile_args)
        else:
            tile = tile_constructor.from_data(tile_data[tile_info['data']['path']])
        tile_map_data[tuple(tile_info['index'])] = tile
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
            tile_data['data'] = _write_ndarray(tile.value, config)
        else:
            tile_data['data'] = _write_ndarray(tile.data, config)
        tile_map_data.append(tile_data)
        config.progress_update()
    return tile_map_data

def _read_mask(mask, tile_data):
    if mask is None:
        return None
    return Mask(
        position=mask['position'],
        alpha=_read_tile_data(mask['alpha'], tile_data),
        visible=mask['visible'],
        background_color=tile_data[mask['background_color']['path']],
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

def _read_sublayers(layers, tile_data):
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
            'mask': _read_mask(sublayer['mask'], tile_data)
        }
        if layer_constructor is PixelLayer:
            layer_args['color'] = _read_tile_data(sublayer['color'], tile_data)
            layer_args['position'] = sublayer['position']
        elif layer_constructor is GroupLayer:
            layer_args['layers'] = _read_sublayers(sublayer['layers'], tile_data)
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
            sublayer_data['position'] = sublayer.position
        elif isinstance(sublayer, GroupLayer):
            sublayer_data['layers'] = _write_sublayers(sublayer.layers, config)
            sublayer_data['folder_open'] = sublayer.folder_open
        layers_data.append(sublayer_data)
    return layers_data

def _gather_tiles(tiles):
    for tile in tiles:
        yield tile['data']

def _gather_mask(mask):
    if mask is not None:
        yield from _gather_tiles(mask['alpha'])
        yield mask['background_color']

def _gather_layers(layers):
    for sublayer in layers:
        yield from _gather_mask(sublayer['mask'])
        if sublayer['type'] == 'PixelLayer':
            yield from _gather_tiles(sublayer['color'])
        elif sublayer['type'] == 'GroupLayer':
            yield from _gather_layers(sublayer['layers'])

def _read_all_tiles(canvas_data, zfile:zipfile.ZipFile):
    data = {}
    for tile_data in itertools.chain(
            _gather_layers(canvas_data['top_level']),
            [canvas_data['background']['color']],
            _gather_mask(canvas_data['selection'])):
        data[tile_data['path']] = _read_ndarray(tile_data, zfile)
    return data

def read(file_path:Path) -> Canvas:
    with zipfile.ZipFile(file_path, 'r') as zfile:
        canvas_data = json.loads(zfile.read('canvas.json').decode())
        tile_data = _read_all_tiles(canvas_data, zfile)
        background = canvas_data['background']
        background['color'] = tile_data[background['color']['path']]
        return Canvas(
            size=tuple(canvas_data['size']),
            top_level=_read_sublayers(canvas_data['top_level'], tile_data),
            background=BackgroundSettings(**background),
            selection=_read_mask(canvas_data['selection'], tile_data)
        )

def write(canvas:Canvas, composite_image:np.ndarray, file_path:Path, progress_callback=None):
    with (tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=file_path.name, delete=False, delete_on_close=False) as temp,
        zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as zfile):
        try:
            config = _SerializeConfig(
                zip_file=zfile,
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
                #'composite': _write_ndarray(composite_image, config)
            }
            config.zip_file.writestr('canvas.json', json.dumps(canvas_data, default=_serialize_numpy_number))
            config.zip_file.close()
            Path(temp.name).rename(file_path)
        except Exception as ex:
            zfile.close()
            temp.close()
            Path(temp.name).unlink(True)
            raise ex
