from pathlib import Path

import psd_tools
import psd_tools.constants as psdc
import psd_tools.api.layers as psdl

from .. import util
from ..layer_data import *
from ..constants import *

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

def _get_sai_special_mode_opacity(layer:psdl.Layer):
    blocks = layer.tagged_blocks
    tsly = blocks.get(psdc.Tag.TRANSPARENCY_SHAPES_LAYER, None)
    iOpa = blocks.get(psdc.Tag.BLEND_FILL_OPACITY, None)
    if tsly and iOpa and tsly.data == 0:
        return STORAGE_DTYPE(iOpa.data), _from_psd_special.get(layer.blend_mode, BlendMode.NORMAL)
    return STORAGE_DTYPE(layer.opacity), _from_psd_blendmode.get(layer.blend_mode, BlendMode.NORMAL)

def _get_protection_settings(layer:psdl.Layer):
    blocks = layer.tagged_blocks
    lspf = blocks.get(psdc.Tag.PROTECTED_SETTING, None)
    data = lspf.data if lspf else 0
    return tuple((util.bit(data, bit) for bit in [0, 1, 2, 32]))

def _get_group_folder_settings(layer:psdl.Layer):
    blocks = layer.tagged_blocks
    lsct = blocks.get(psdc.Tag.SECTION_DIVIDER_SETTING, None)
    return lsct.data.kind == psdc.SectionDivider.OPEN_FOLDER

def _get_layer_channel(layer:psdl.Layer, channel):
    data = util.layer_numpy(layer, channel)
    tile_type = ColorTile if channel == 'color' else AlphaTile
    return pixel_data_to_tiles(data, tile_type)

def _get_mask(layer:psdl.Layer):
    if layer.mask:
        return Mask(
            position=(layer.mask.top, layer.mask.left),
            alpha=_get_layer_channel(layer, 'mask'),
            visible=not layer.mask.disabled,
            background_color=layer.mask.background_color,
        )
    else:
        return None

def _get_base_layer_properties(layer:psdl.Layer):
    opacity, blend_mode = _get_sai_special_mode_opacity(layer)
    lock_alpha, lock_draw, lock_move, lock_all = _get_protection_settings(layer)
    return {
        'name': layer.name,
        'blend_mode': blend_mode,
        'visible': layer.visible,
        'opacity': opacity,
        'lock_alpha': lock_alpha,
        'lock_draw': lock_draw,
        'lock_move': lock_move,
        'lock_all': lock_all,
        'clip': layer.clipping_layer,
        'mask': _get_mask(layer),
        'id': layer.layer_id,
    }

def _get_group_layer_properties(layer:psdl.Layer):
    return {
        'layers': _build_sublayers(layer),
        'folder_open': _get_group_folder_settings(layer),
    }

def _get_pixel_layer_properties(layer:psdl.Layer):
    color, alpha = prune_tiles(
        _get_layer_channel(layer, 'color'),
        _get_layer_channel(layer, 'shape'))
    return {
        'color': color,
        'alpha': alpha,
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
    alpha = util.layer_numpy(layer, 'shape')
    alpha_all_1 = True if alpha is None else (alpha == 255).all()
    if alpha_all_1:
        color = util.layer_numpy(layer, 'color')
        full_color = color[0, 0]
        color_all_eq = (color == full_color).all()
        return color_all_eq, tuple(full_color)
    return False, None

def read(file_path:Path) -> Canvas:
    psd_file = psd_tools.PSDImage.open(str(file_path))
    bg = BackgroundSettings(
        transparent=True
    )
    if len(psd_file) > 0:
        bg_layer = psd_file[0]
        if (bg_layer.name == 'Background'
            and bg_layer.kind == 'pixel'
            and bg_layer.mask is None
            and psd_file.size == bg_layer.size):
            pure, full_color = _is_pure_background(bg_layer)
            if pure:
                bg = BackgroundSettings(
                    color=full_color,
                )
                psd_file._layers.pop(0)
    return Canvas(
        size=(psd_file.height, psd_file.width),
        top_level=_build_sublayers(psd_file),
        background=bg,
    )

# TODO: psd-tools says it doesn't support adding layers to a PSD
# but I think I can hack it to make it work, which would probably
# save a lot of pain trying to write a PSD exporter from scratch.
def write(canvas:Canvas, file_path:Path):
    raise NotImplementedError()
