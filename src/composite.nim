## Compositing functions.
import std/algorithm

import parazoa

import util
import canvas_data
import blendfuncs
import option

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

func init_blank_color: ColorPixelArray =
    result

func init_blank_mask(): MaskPixelArray =
    for i in 0 ..< result.len:
        result[i] = blendfuncs.range_max

const blank_color = init_blank_color()
const blank_mask = init_blank_mask()

proc composite*(canvas: Canvas, layers: Vec[LayerId], timeline_pos: TimelinePos, offset: VectorI, backdrop: ref seq[ColorPixelArray]) =
    for (sublayer_id, clip_layers) in get_layer_and_clip_groupings(canvas, layers):
        let sublayer = canvas.layer_lookup[sublayer_id]
        if not sublayer.visible:
            continue

        var color_src: ref seq[ColorPixelArray]

        if sublayer of GroupLayer:
            var group = GroupLayer(sublayer)
            if sublayer.blend_mode == BlendMode.Pass:
                group.temp[] = backdrop[]
            else:
                for i in 0 ..< group.temp[].len:
                    group.temp[][i] = blank_color
            composite(canvas, GroupLayer(sublayer).layers, timeline_pos, offset, group.temp)

        let mask_frame = get_mask_frame(canvas, sublayer, timeline_pos)
        var mask_src: MaskPixelArray
        case mask_frame:
        of Some(@frame):
            mask_src = get_tile_data(frame, get_tiles_size(canvas.size.x), offset)
        of None():
            mask_src = blank_mask

        if sublayer.blend_mode == BlendMode.Pass:
            discard
        else:
            discard
    # return backdrop
