from pathlib import Path
import struct
import io
import tempfile

import psd_tools
import psd_tools.constants as psdc
import psd_tools.api.layers as psdl

from . import rle
from .. import util
from ..layer_data import *
from ..constants import *
from .. import blendfuncs

_from_psd_blendmode = {
    psdc.BlendMode.PASS_THROUGH: BlendMode.PASS,
    psdc.BlendMode.NORMAL: BlendMode.NORMAL,
    psdc.BlendMode.MULTIPLY: BlendMode.MULTIPLY,
    psdc.BlendMode.SCREEN: BlendMode.SCREEN,
    psdc.BlendMode.OVERLAY: BlendMode.OVERLAY,
    psdc.BlendMode.LINEAR_BURN: BlendMode.TS_LINEAR_BURN,
    psdc.BlendMode.LINEAR_DODGE: BlendMode.TS_LINEAR_DODGE,
    psdc.BlendMode.LINEAR_LIGHT: BlendMode.TS_LINEAR_LIGHT,
    psdc.BlendMode.COLOR_BURN: BlendMode.TS_COLOR_BURN,
    psdc.BlendMode.COLOR_DODGE: BlendMode.TS_COLOR_DODGE,
    psdc.BlendMode.VIVID_LIGHT: BlendMode.TS_VIVID_LIGHT,
    psdc.BlendMode.HARD_LIGHT: BlendMode.HARD_LIGHT,
    psdc.BlendMode.SOFT_LIGHT: BlendMode.SOFT_LIGHT,
    psdc.BlendMode.PIN_LIGHT: BlendMode.PIN_LIGHT,
    psdc.BlendMode.HARD_MIX: BlendMode.TS_HARD_MIX,
    psdc.BlendMode.DARKEN: BlendMode.DARKEN,
    psdc.BlendMode.LIGHTEN: BlendMode.LIGHTEN,
    psdc.BlendMode.DARKER_COLOR: BlendMode.DARKEN_COLOR,
    psdc.BlendMode.LIGHTER_COLOR: BlendMode.LIGHTEN_COLOR,
    psdc.BlendMode.DIFFERENCE: BlendMode.TS_DIFFERENCE,
    psdc.BlendMode.EXCLUSION: BlendMode.EXCLUDE,
    psdc.BlendMode.SUBTRACT: BlendMode.SUBTRACT,
    psdc.BlendMode.DIVIDE: BlendMode.DIVIDE,
    psdc.BlendMode.HUE: BlendMode.HUE,
    psdc.BlendMode.SATURATION: BlendMode.SATURATION,
    psdc.BlendMode.COLOR: BlendMode.COLOR,
    psdc.BlendMode.LUMINOSITY: BlendMode.LUMINOSITY,
}
_from_psd_special = {
    psdc.BlendMode.LINEAR_BURN: BlendMode.LINEAR_BURN,
    psdc.BlendMode.LINEAR_DODGE: BlendMode.LINEAR_DODGE,
    psdc.BlendMode.LINEAR_LIGHT: BlendMode.LINEAR_LIGHT,
    psdc.BlendMode.COLOR_BURN: BlendMode.COLOR_BURN,
    psdc.BlendMode.COLOR_DODGE: BlendMode.COLOR_DODGE,
    psdc.BlendMode.VIVID_LIGHT: BlendMode.VIVID_LIGHT,
    psdc.BlendMode.HARD_MIX: BlendMode.HARD_MIX,
    psdc.BlendMode.DIFFERENCE: BlendMode.DIFFERENCE,
}

_to_psd_blendmode = { v:k for k,v in _from_psd_blendmode.items() }
_to_psd_special = { v:k for k,v in _from_psd_special.items() }

def _channel_matches(layer, channel, info):
    if channel == 'color':
        return info.id >= 0
    if channel == 'shape':
        return info.id == psdc.ChannelID.TRANSPARENCY_MASK
    if channel == 'mask':
        if not layer.mask:
            return False
        if layer.mask._has_real():
            return info.id == psdc.ChannelID.REAL_USER_LAYER_MASK
        else:
            return info.id == psdc.ChannelID.USER_LAYER_MASK
    else:
        raise ValueError(f'Unknown channel type: {channel}')

def _layer_numpy(layer:psdl.Layer, channel_name=None):
    if channel_name == 'mask' and (not layer.mask or layer.mask.size == (0, 0)):
        return None

    is_color = channel_name == 'color'
    depth = layer._psd.depth
    version = layer._psd.version

    all_channels = list(zip(layer._channels, layer._record.channel_info))
    channels = [channel for channel, info in all_channels if _channel_matches(layer, channel_name, info)]
    has_alpha = False
    if is_color:
        for channel, info in all_channels:
            if _channel_matches(layer, 'shape', info):
                channels.append(channel)
                has_alpha = True
                break

    if len(channels) == 0:
        return None

    # Use the psd-tools path if can't decode everything with RLE.
    if not all([channel.compression == psdc.Compression.RLE for channel in channels]):
        color = blendfuncs.from_floats(layer.numpy(channel))
        if is_color:
            alpha = blendfuncs.from_floats(layer.numpy('shape'))
            color = np.stack((color, alpha), axis=1).reshape((height, width, -1))
        return color

    if channel_name == 'mask':
        width, height = layer.mask.width, layer.mask.height
    else:
        width, height = layer.width, layer.height

    decoded = [blendfuncs.parse_array(rle.decode_psd(channel.data, width, height, depth, version), depth) for channel in channels]
    if is_color and not has_alpha:
        decoded.append(np.full_like(decoded[0], blendfuncs.get_max()))

    return np.stack(decoded, axis=1).reshape((height, width, -1))

def _get_sai_special_mode_opacity(layer:psdl.Layer):
    blocks = layer.tagged_blocks
    tsly = blocks.get(psdc.Tag.TRANSPARENCY_SHAPES_LAYER, None)
    iOpa = blocks.get(psdc.Tag.BLEND_FILL_OPACITY, None)
    if tsly and iOpa and tsly.data == 0:
        opacity, blend_mode = iOpa.data, _from_psd_special.get(layer.blend_mode, BlendMode.NORMAL)
    else:
        opacity, blend_mode = layer.opacity, _from_psd_blendmode.get(layer.blend_mode, BlendMode.NORMAL)
    return blendfuncs.from_bytes(np.uint8(opacity)), blend_mode

def _get_group_folder_settings(layer:psdl.Layer):
    blocks = layer.tagged_blocks
    lsct = blocks.get(psdc.Tag.SECTION_DIVIDER_SETTING, None)
    return lsct.data.kind == psdc.SectionDivider.OPEN_FOLDER

def _get_layer_channel(layer:psdl.Layer, channel, prune=False):
    def task():
        data = pixel_data_to_tiles(_layer_numpy(layer, channel))
        if prune:
            data = prune_tiles(data)
        return data
    return util.pool.submit(task)

def _get_mask(layer:psdl.Layer):
    if layer.mask:
        return Mask(
            position=(layer.mask.top, layer.mask.left),
            data=_get_layer_channel(layer, 'mask'),
            visible=not layer.mask.disabled,
            background_color=blendfuncs.from_bytes(np.uint8(layer.mask.background_color)),
        )
    else:
        return None

def debug_layer(layer:psdl.Layer):
    print(layer.name)
    print(' is group:', layer.is_group())
    print(' extents:', (layer.bbox[1], layer.bbox[0], layer.bbox[3], layer.bbox[2]))
    print(' channels:', len(layer._channels))
    print(' opacity:', layer.opacity)
    print(' blendmode:', layer.blend_mode.name)
    print(' flags transparency protected:', layer._record.flags.transparency_protected)
    print(' flags visible:', layer._record.flags.visible)
    print(' visible:', layer.visible)
    if layer.locks is not None:
        if layer.locks.transparency: print(' lock alpha')
        if layer.locks.composite: print(' lock draw')
        if layer.locks.position: print(' lock move')
        if layer.locks.complete: print(' lock all')
    if layer.clipping_layer: print(' is clipping')
    if layer.has_mask(): print(' has mask')
    print(' records:')
    for key in layer._record.tagged_blocks.keys():
        data = layer._record.tagged_blocks.get_data(key)
        if key == psdc.Tag.SECTION_DIVIDER_SETTING:
            data = data.kind.name
        print(f'  {key.name}:', data)

def _get_base_layer_properties(layer:psdl.Layer):
    #debug_layer(layer)
    opacity, blend_mode = _get_sai_special_mode_opacity(layer)
    props = {
        'name': layer.name,
        'blend_mode': blend_mode,
        'visible': layer.visible,
        'opacity': opacity,
        'clip': layer.clipping_layer,
        'mask': _get_mask(layer),
        'id': layer.layer_id,
    }
    if layer.locks is not None:
        props |= {
            'lock_alpha': layer.locks.transparency,
            'lock_draw': layer.locks.composite,
            'lock_move': layer.locks.position,
            'lock_all': layer.locks.composite,
        }
    return props

def _get_group_layer_properties(layer:psdl.Layer):
    return {
        'layers': _build_sublayers(layer),
        'folder_open': _get_group_folder_settings(layer),
    }

def _get_pixel_layer_properties(layer:psdl.Layer):
    return {
        'data': _get_layer_channel(layer, 'color', True),
        'position': (layer.top, layer.left),
    }

def _build_sublayers(psd_group) -> GroupLayer:
    layers = []
    for psd_sublayer in psd_group:
        args = _get_base_layer_properties(psd_sublayer)
        if psd_sublayer.is_group():
            layers.append(GroupLayer(**(args | _get_group_layer_properties(psd_sublayer))))
        else:
            layers.append(PixelLayer(**(args | _get_pixel_layer_properties(psd_sublayer))))
    return pvector(layers)

def _is_pure_background(layer:psdl.Layer):
    alpha = _layer_numpy(layer, 'shape')
    alpha_all_1 = True if alpha is None else (alpha == blendfuncs.get_max()).all()
    if alpha_all_1:
        color = util.get_color(_layer_numpy(layer, 'color'))
        full_color = color[0, 0]
        color_all_eq = (color == full_color).all()
        return color_all_eq, np.array(full_color, dtype=blendfuncs.dtype)
    return False, None

def debug_psd(psd:psd_tools.PSDImage):
    # for k in psd_file.image_resources.keys():
    #     print(psdc.Resource(k).name)
    #     print(psd_file.image_resources.get_data(k))
    for record, channels in psd._record._iter_layers():
        blocks = record.tagged_blocks
        divider = blocks.get_data(psdc.Tag.SECTION_DIVIDER_SETTING, None)
        divider = blocks.get_data(psdc.Tag.NESTED_SECTION_DIVIDER_SETTING, divider)
        divider = '' if divider is None else divider.kind.name
        print(record.name, divider)

def read(file_path:Path) -> Canvas:
    psd_file = psd_tools.PSDImage.open(str(file_path))
    bg = BackgroundSettings(transparent=True)
    if len(psd_file) > 0:
        bg_layer = psd_file[0]
        if (bg_layer.name == 'Background'
            and bg_layer.kind == 'pixel'
            and bg_layer.mask is None
            and psd_file.size == bg_layer.size):
            pure, full_color = _is_pure_background(bg_layer)
            if pure:
                bg = BackgroundSettings(color=full_color)
                psd_file._layers.pop(0)
    return reify_canvas_futures(Canvas(
        size=(psd_file.height, psd_file.width),
        top_level=_build_sublayers(psd_file),
        background=bg,
    ))

BIT_DEPTH = 8
CHANNELS = 4
SIGNATURE = b'8BIM'
COLOR_CHANNELS = (psdc.ChannelID.CHANNEL_0, psdc.ChannelID.CHANNEL_1, psdc.ChannelID.CHANNEL_2, psdc.ChannelID.TRANSPARENCY_MASK)
MASK_CHANNELS = (psdc.ChannelID.USER_LAYER_MASK,)
RLE_HEADER = struct.pack('>H', psdc.Compression.RLE)

class _SerializeConfig:
    def __init__(self, canvas:Canvas, version:int, progress_callback):
        self.canvas = canvas
        self.version = version
        self.layer_data = []
        self.current_count = 0
        self.progress_count = canvas.count_layers()
        self.progress_callback = progress_callback

    def progress_update(self):
        if self.progress_callback is not None:
            self.current_count += 1
            self.progress_callback(self.current_count / self.progress_count)

    def vselect(self, v1, v2):
        return v1 if self.version == 1 else v2

def _get_pad(size, pad):
    mod = size % pad
    if mod > 0:
        return b'\0' * (pad - mod)
    else:
        return b''

def _to_rle(array:np.ndarray, version):
    result = []
    for channel in range(array.shape[2]):
        src = array[:,:,channel]
        result.append([r.tobytes() for r in rle.encode_psd(src.reshape(src.shape[:2]), version)])
    return result

def _collect_layer_data(layer:BaseLayer, config:_SerializeConfig, group_end=False, background=False):
    layer_record = io.BytesIO()
    channel_data = io.BytesIO()
    config.layer_data.append((layer_record, channel_data))

    # Collect and compress channel planes
    planes = []
    if isinstance(layer, PixelLayer):
        if background:
            color = tuple(blendfuncs.to_bytes(config.canvas.background.color))# + (np.uint8(0xff),)
            pixel_data = np.full(shape=config.canvas.size + (3,), fill_value=color, dtype=np.uint8)
            extents = (0, 0) + config.canvas.size
        else:
            extents, pixel_data = tiles_to_pixel_data(layer)
        print(layer.name, extents)
        planes.extend(zip(_to_rle(pixel_data, config.version), COLOR_CHANNELS))
    else:
        extents = (0, 0, 0, 0)
        planes.extend(zip([[]] * 4, COLOR_CHANNELS))
    if layer.mask is not None:
        mask_extents, pixel_data = tiles_to_pixel_data(layer.mask)
        planes.extend(zip(_to_rle(pixel_data, config.version), MASK_CHANNELS))

    # Collect channel info data
    channel_info = []
    for data, channel_id in planes:
        sub_size = 0
        for sub in [RLE_HEADER] + data:
            sub_size += channel_data.write(sub)
        channel_info.append(struct.pack(config.vselect('>hI', '>hQ'), channel_id, sub_size))

    # Get blend mode
    if layer.blend_mode in _to_psd_special:
        special_mode = True
        blend_mode = _to_psd_special.get(layer.blend_mode)
    else:
        special_mode = False
        blend_mode = _to_psd_blendmode.get(layer.blend_mode)

    records = []

    # Mask record
    if layer.mask is None:
        records.append(struct.pack('>I', 0))
    else:
        records.append(struct.pack(
            '>I4iBB2x',
            20, # Length
            *mask_extents,
            blendfuncs.to_bytes(layer.mask.background_color),
            util.set_bits([
                (1, not layer.mask.visible)
            ])
        ))

    # Blending ranges record
    records.append(struct.pack('>I', 0))

    # Name record
    layer_name = struct.pack('>p', layer.name.encode('macroman', 'replace'))
    records.append(layer_name)
    records.append(_get_pad(len(layer_name), 4))

    # Group start/end
    if isinstance(layer, GroupLayer):
        if group_end:
            divider_setting = psdc.SectionDivider.BOUNDING_SECTION_DIVIDER
        else:
            divider_setting = psdc.SectionDivider.OPEN_FOLDER if layer.folder_open else psdc.SectionDivider.CLOSED_FOLDER
        records.append(struct.pack(
            '>4s4sII',
            SIGNATURE,
            psdc.Tag.SECTION_DIVIDER_SETTING,
            4,
            divider_setting
        ))

    if not group_end:
        # Protection flags
        records.append(struct.pack(
            '>4s4sII',
            SIGNATURE,
            psdc.Tag.PROTECTED_SETTING,
            4,
            util.set_bits([
                (0, layer.lock_alpha),
                (1, layer.lock_draw),
                (2, layer.lock_move),
                (32, layer.lock_all),
            ])
        ))

        # Layer ID
        records.append(struct.pack(
            '>4s4sII',
            SIGNATURE,
            psdc.Tag.LAYER_ID,
            4,
            layer.id
        ))

        # SAI special mode
        if special_mode:
            records.append(struct.pack(
                '>4s4sIB3x',
                SIGNATURE,
                psdc.Tag.TRANSPARENCY_SHAPES_LAYER,
                4,
                0,
            ))

            records.append(struct.pack(
                '>4s4sIB3x',
                SIGNATURE,
                psdc.Tag.BLEND_FILL_OPACITY,
                4,
                blendfuncs.to_bytes(layer.opacity)
            ))

        # Unicode layer name
        layer_unicode_name = layer.name.encode('utf-16-be') + b'\0\0'
        records.append(struct.pack(
            '>4s4sII',
            SIGNATURE,
            psdc.Tag.UNICODE_LAYER_NAME,
            4 + len(layer_unicode_name),
            len(layer.name),
        ))
        records.append(layer_unicode_name)

    # Write everything to file
    # Layer record header
    layer_record.write(struct.pack(
        '>4iH',
        *extents,
        len(channel_info)
    ))
    for info in channel_info:
        layer_record.write(info)
    layer_record.write(struct.pack(
        '>4s4sBBBxI',
        SIGNATURE,
        blend_mode,
        255 if special_mode else blendfuncs.to_bytes(layer.opacity),
        psdc.Clipping.NON_BASE if layer.clip else psdc.Clipping.BASE,
        util.set_bits([
            (0, layer.lock_alpha),
            (1, not layer.visible),
        ]),
        sum(len(record) for record in records)
    ))
    for record in records:
        layer_record.write(record)

def _collect_layers(layer, config):
    if isinstance(layer, PixelLayer):
        _collect_layer_data(layer, config)
        config.progress_update()

    elif isinstance(layer, GroupLayer):
        _collect_layer_data(GroupLayer(name='</Layer group>'), config, group_end=True)
        for sublayer in layer:
            _collect_layers(sublayer, config)
        _collect_layer_data(layer, config)
        config.progress_update()

    elif isinstance(layer, Canvas):
        if not layer.background.transparent:
            _collect_layer_data(PixelLayer(name='Background'), config, background=True)

        for sublayer in layer:
            _collect_layers(sublayer, config)

def write(canvas:Canvas, composite_image:np.ndarray, file_path:Path, progress_callback=None):
    with tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=file_path.name, delete=False, delete_on_close=False) as fp:
        try:
            version = 1 if file_path.suffix.lower() == '.psd' else 2
            config = _SerializeConfig(canvas, version, progress_callback)

            fp.write(struct.pack('>4sH6xHIIHHII',
                b'8BPS',
                config.version,
                CHANNELS,
                canvas.size[0],
                canvas.size[1],
                BIT_DEPTH,
                psdc.ColorMode.RGB,
                0, # Color mode section size
                0, # Image resources section size
            ))

            # Placeholder layer and mask info header
            layer_info_pos = fp.tell()
            fp.write(struct.pack(
                config.vselect('>IIh', '>QQh'),
                0,
                0,
                0,
            ))

            _collect_layers(canvas, config)

            layer_info_size = 0
            for layer_record, _ in config.layer_data:
                layer_info_size += fp.write(layer_record.getbuffer())

            for _, channel_data in config.layer_data:
                layer_info_size += fp.write(channel_data.getbuffer())

            layer_info_size += fp.write(_get_pad(layer_info_size, 2))

            # Global layer mask info.
            global_layer_mask_size = fp.write(struct.pack('>I', 0))

            # Global tagged blocks (empty)

            # Rewrite layer and mask info header now that we know sizes
            fp.seek(layer_info_pos, io.SEEK_SET)
            fp.write(struct.pack(
                config.vselect('>IIh', '>QQh'),
                layer_info_size + global_layer_mask_size,
                layer_info_size,
                -len(config.layer_data),
            ))
            fp.seek(0, io.SEEK_END)

            # Image data section (composite image)
            planes = _to_rle(composite_image, config.version)
            fp.write(RLE_HEADER)
            for plane in planes:
                for sub in plane:
                    fp.write(sub)

            fp.close()
            Path(fp.name).rename(file_path)
        except Exception as ex:
            fp.close()
            Path(fp.name).unlink(True)
            raise ex
