## Compositing functions.
import std/algorithm
import std/options
import fusion/matching

import parazoa

import util
import canvas_data
import blendfuncs

func get_layer_and_clip_groupings*(layers: Vec[Layer]): seq[(Layer, seq[Layer])] =
    var grouped_layers: seq[(Layer, seq[Layer])] = @[]
    var clip_stack: seq[Layer] = @[]
    var empty: seq[Layer] = @[]
    for layer in util.reversed(layers):
        if layer.clip:
            clip_stack.add(layer)
        else:
            clip_stack.reverse()
            if layer.blend_mode == BlendMode.Pass:
                for sublayer in clip_stack:
                    grouped_layers.add((sublayer, empty))
                grouped_layers.add((layer, empty))
            else:
                grouped_layers.add((layer, clip_stack))
            clip_stack = @[]
    for sublayer in clip_stack:
        grouped_layers.add((sublayer, empty))
    grouped_layers.reverse()
    return grouped_layers

proc composite_tile*(canvas: Canvas, layers: Vec[Layer], frame_id: FrameId, offset: Vector, backdrop: ref ColorTile): ref ColorTile =
    for sublayer, clip_layers in get_layer_and_clip_groupings(layers).items:
        if not sublayer.visible:
            continue

        var color_src: ref ColorTile

        if sublayer of GroupLayer:
            var next_backdrop: ref ColorTile
            if sublayer.blend_mode == BlendMode.Pass:
                next_backdrop[] = backdrop[]
            else:
                next_backdrop = ColorTile.new()
            color_src = composite_tile(canvas, GroupLayer(sublayer).layers, frame_id, offset, next_backdrop)

        let mask_src:Option[(MaskFrame, MaskTile)] = get_mask_tile(canvas, sublayer, frame_id, offset)

    return backdrop