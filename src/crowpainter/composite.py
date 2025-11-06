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

import numpy as np

from . import blendfuncs, constants, layer_data, util

def check_lock(arr):
    return arr if isinstance(arr, np.ndarray) and arr.flags.writeable else None

def get_layer_and_clip_groupings(layers:layer_data.GroupLayer | list [layer_data.BaseLayer]):
    grouped_layers = []
    clip_stack = []
    for layer in reversed(layers):
        if layer.clip:
            clip_stack.append(layer)
        else:
            clip_stack.reverse()
            if layer.blend_mode == constants.BlendMode.PASS:
                for sublayer in clip_stack:
                    grouped_layers.append((sublayer, []))
                grouped_layers.append((layer, []))
            else:
                grouped_layers.append((layer, clip_stack))
            clip_stack = []
    for sublayer in clip_stack:
        grouped_layers.append((sublayer, []))
    grouped_layers.reverse()
    return grouped_layers

def composite(layer:layer_data.GroupLayer | list[layer_data.BaseLayer], offset:layer_data.IVec2, backdrop:np.ndarray):
    for sublayer, clip_layers in get_layer_and_clip_groupings(layer):
        if not sublayer.visible:
            continue
        blend_mode = sublayer.blend_mode

        if isinstance(sublayer, layer_data.GroupLayer):
            if blend_mode == constants.BlendMode.PASS:
                next_backdrop = backdrop.copy()
            else:
                next_backdrop = np.zeros_like(backdrop)
            color_src = composite(sublayer, offset, next_backdrop)
            pixel_srcs = { offset: (backdrop, color_src) }
        elif isinstance(sublayer, layer_data.PixelLayer):
            pixel_srcs = sublayer.get_pixel_data(backdrop, offset)

        opacity = sublayer.opacity

        for (sub_offset, (sub_color_dst, sub_color_src)) in pixel_srcs.items():
            mask_src = sublayer.get_mask_data(sub_color_src.shape[:2], sub_offset)

            # A pass-through layer has already been blended, so just lerp instead.
            # NOTE: Clipping layers do not apply to pass layers, as if clipping were simply disabled.
            if blend_mode == constants.BlendMode.PASS:
                if mask_src is None:
                    mask_src = opacity
                else:
                    blendfuncs.mul(mask_src, opacity, out=mask_src)
                blendfuncs.lerp(sub_color_dst, sub_color_src, mask_src, out=sub_color_dst)
            else:
                color_src = util.get_color(sub_color_src)
                alpha_src = util.get_alpha(sub_color_src)

                if isinstance(sublayer, layer_data.GroupLayer):
                    # Un-multiply group composites so that we can multiply group opacity correctly
                    color_src = blendfuncs.clip_divide(color_src, alpha_src, out=check_lock(color_src))

                if clip_layers:
                    # Composite the clip layers now. This basically overwrites just the color by blending onto it without
                    # alpha blending it first.
                    color_src_copy = sub_color_src.copy()
                    alpha_src_copy = util.get_alpha(color_src_copy)
                    blendfuncs.threshold(alpha_src_copy, out=alpha_src_copy)
                    sub_color_src = composite(clip_layers, sub_offset, color_src_copy)
                    np.copyto(util.get_alpha(sub_color_src), alpha_src)
                    color_src = util.get_color(sub_color_src)
                    alpha_src = util.get_alpha(sub_color_src)

                color_dst = util.get_color(sub_color_dst)
                alpha_dst = util.get_alpha(sub_color_dst)

                # Apply opacity (fill) before blending otherwise premultiplied blending of special modes will not work correctly.
                alpha_src = blendfuncs.mul(alpha_src, opacity, out=check_lock(alpha_src))

                # Now we can 'premultiply' the color_src for the main blend operation.
                color_src = blendfuncs.mul(color_src, alpha_src, out=check_lock(color_src))

                # Run the blend operation.
                blend_func = blendfuncs.get_blend_func(blend_mode)
                with np.errstate(divide='ignore', invalid='ignore'):
                    color_src = blend_func(color_dst, color_src, alpha_dst, alpha_src, out=check_lock(color_src))

                # Premultiplied blending may cause out-of-range values, so it must be clipped.
                if blend_mode != constants.BlendMode.NORMAL:
                    color_src = blendfuncs.clip(color_src, out=check_lock(color_src))

                # We apply the mask last and LERP the blended result onto the destination.
                # Why? Because this is how Photoshop and SAI do it. Applying the mask before blending
                # will yield a slightly different result from those programs.
                if mask_src is not None:
                    blendfuncs.lerp(color_dst, color_src, mask_src, out=color_dst)
                else:
                    np.copyto(color_dst, color_src)

                # Finally we can intersect the mask with the alpha_src and blend the alpha_dst together.
                if mask_src is not None:
                    alpha_src = blendfuncs.mul(alpha_src, mask_src, out=check_lock(alpha_src))
                blendfuncs.normal_alpha(alpha_dst, alpha_src, out=alpha_dst)

    return backdrop
