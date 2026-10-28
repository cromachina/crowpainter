## Compositing functions.
import std/algorithm
import std/options
import fusion/matching

import parazoa

import util
import canvas_data
import blendfuncs

func get_layer_and_clip_groupings*(canvas: Canvas, layers: Vec[LayerId]): seq[(LayerId, seq[LayerId])] =
    var grouped_layers: seq[(LayerId, seq[LayerId])] = @[]
    var clip_stack: seq[LayerId] = @[]
    var empty: seq[LayerId] = @[]
    for layer_id in layers:
        let layer = canvas.layer_lookup[layer_id]
        if layer.clip:
            clip_stack.add(layer_id)
        else:
            if layer.blend_mode == BlendMode.Pass:
                for sublayer_id in clip_stack:
                    grouped_layers.add((sublayer_id, empty))
                grouped_layers.add((layer_id, empty))
            else:
                grouped_layers.add((layer_id, clip_stack))
            clip_stack = @[]
    for sublayer_id in clip_stack:
        grouped_layers.add((sublayer_id, empty))
    return grouped_layers

func init_blank_mask(): MaskPixelArray =
    for i in 0 ..< result.len:
        result[i] = blendfuncs.range_max

const blank_mask = init_blank_mask()

proc composite_tile*(canvas: Canvas, layers: Vec[LayerId], timeline_pos: TimelinePos, offset: VectorI, backdrop: ref ColorTile): ref ColorTile =
    for (sublayer_id, clip_layers) in get_layer_and_clip_groupings(canvas, layers):
        let sublayer = canvas.layer_lookup[sublayer_id]
        if not sublayer.visible:
            continue

        var color_src: ref ColorTile

        if sublayer of GroupLayer:
            var next_backdrop: ref ColorTile
            if sublayer.blend_mode == BlendMode.Pass:
                next_backdrop[] = backdrop[]
            else:
                next_backdrop = ColorTile.new()
            color_src = composite_tile(canvas, GroupLayer(sublayer).layers, timeline_pos, offset, next_backdrop)

        let mask_frame = get_mask_frame(canvas, sublayer, timeline_pos)
        var mask_src: MaskPixelArray
        case mask_frame:
        of Some(@frame):
            mask_src = get_temp_tile(frame, get_tiles_size(canvas.size.x), offset)
        of None():
            mask_src = blank_mask

        # if blend_mode == BlendMode.Pass:
        #     case mask_src:
        #     of @Some(mask_val):
        #         case mask_val
        #         of @MaskFillTile(fill):
        #             mask_src = blendfuncs.mul(mask_src, opacity)

    return backdrop