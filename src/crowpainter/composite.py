# Time optimized compositing of a change from the current active layer:
# - All layers successive composite results are cached, so each layer
#   contains x2 pixel data per tile. Call this the lower cache.
# - Above the current active layer, every run of normal blend mode
#   layers is composited into one cached layer, but every other blend mode layer is not.
#   Call this the upper cache.
# - Composite the current layer on top of the immediately below lower cache.
# - Then composite this result with the next successive upper caches until finished.
# - The amount of layer blends that occur in the middle of the stack of many layers
#   is 1 + number of upper caches
# - It might be possible that the upper cache could contain runs of other blend modes
#   that are associative operations, such as multiply or add.

# Alternatively, just do it the dumb way since most brush strokes are probably small enough for it

from . import blendfuncs, util
from .util import peval
from .constants import *
from .layer_data import *

def check_lock(arr:np.ndarray):
    return arr if arr.flags.writeable else None

async def composite(layer:GroupLayer | list[BaseLayer], offset:IVec2, backdrop:tuple[np.ndarray, np.ndarray]):
    color_dst, alpha_dst = backdrop

    for sublayer in layer:
        if not sublayer.visible:
            continue
        blend_mode = sublayer.blend_mode

        if isinstance(sublayer, GroupLayer):
            if blend_mode == BlendMode.PASS_THROUGH:
                next_backdrop = (color_dst, alpha_dst)
            else:
                next_color = np.zeros_like(color_dst)
                next_alpha = np.zeros_like(alpha_dst)
                next_backdrop = (next_color, next_alpha)
            color_src, alpha_src = await composite(sublayer, offset, next_backdrop)
            pixel_srcs = { offset: ((color_dst, color_src), (alpha_dst, alpha_src)) }
        elif isinstance(sublayer, PixelLayer):
            pixel_srcs = sublayer.get_pixel_data(color_dst, alpha_dst, offset)

        opacity = sublayer.opacity / 100.0

        for (sub_offset, ((sub_color_dst, sub_color_src), (sub_alpha_dst, sub_alpha_src))) in pixel_srcs.items():
            sub_masks = sublayer.get_mask_data(sub_alpha_dst, sub_offset)
            if sub_masks is None:
                mask_src = None
            else:
                mask_src = sub_masks.get(sub_offset)

            # A pass-through layer has already been blended, so just lerp instead.
            # NOTE: Clipping layers do not apply to pass layers, as if clipping were simply disabled.
            if blend_mode == BlendMode.PASS_THROUGH:
                if mask_src is None:
                    mask_src = opacity
                else:
                    mask_src = np.multiply(mask_src, opacity)
                if np.isscalar(mask_src) and mask_src == 1.0:
                    np.copyto(sub_color_dst, sub_color_src)
                    np.copyto(sub_alpha_dst, sub_alpha_src)
                else:
                    blendfuncs.lerp(sub_color_dst, sub_color_src, mask_src, out=sub_color_dst)
                    blendfuncs.lerp(sub_alpha_dst, sub_alpha_src, mask_src, out=sub_alpha_dst)
            else:
                if isinstance(sublayer, GroupLayer):
                    # Un-multiply group composites so that we can multiply group opacity correctly
                    sub_color_src = blendfuncs.clip_divide(sub_color_src, sub_alpha_src, out=check_lock(sub_color_src))

                if sublayer.clip_layers:
                    # Composite the clip layers now. This basically overwrites just the color by blending onto it without
                    # alpha blending it first. For whatever reason, applying a large root to the alpha source before passing
                    # it to clip compositing fixes brightening that can occur with certain blend modes (like multiply).
                    corrected_alpha = sub_alpha_src ** 0.0001
                    clip_src, _ = composite(sublayer.clip_layers, offset, (sub_color_src, corrected_alpha))
                    if clip_src is not None:
                        sub_color_src = clip_src

                # Apply opacity (fill) before blending otherwise premultiplied blending of special modes will not work correctly.
                sub_alpha_src = np.multiply(sub_alpha_src, opacity, out=check_lock(sub_alpha_src))

                # Now we can 'premultiply' the color_src for the main blend operation.
                sub_color_src = np.multiply(sub_color_src, sub_alpha_src, out=check_lock(color_src))

                # Run the blend operation.
                blend_func = blendfuncs.get_blend_func(blend_mode)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sub_color_src = blend_func(sub_color_dst, sub_color_src, sub_alpha_dst, sub_alpha_src, out=check_lock(sub_color_src))

                # Premultiplied blending may cause out-of-range values, so it must be clipped.
                if blend_mode != BlendMode.NORMAL:
                    sub_color_src = blendfuncs.clip(sub_color_src, out=check_lock(sub_color_src))

                # We apply the mask last and LERP the blended result onto the destination.
                # Why? Because this is how Photoshop and SAI do it. Applying the mask before blending
                # will yield a slightly different result from those programs.
                if mask_src is not None:
                    blendfuncs.lerp(sub_color_dst, sub_color_src, mask_src, out=sub_color_dst)
                else:
                    np.copyto(sub_color_dst, sub_color_src)

                # Finally we can intersect the mask with the alpha_src and blend the alpha_dst together.
                if mask_src is not None:
                    sub_alpha_src = np.multiply(sub_alpha_src, mask_src, out=check_lock(sub_alpha_src))
                blendfuncs.normal_alpha(sub_alpha_dst, sub_alpha_src, out=sub_alpha_dst)

    return color_dst, alpha_dst
