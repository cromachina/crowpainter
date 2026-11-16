## Types that describe the canvas document structure, layers, tiles, etc.
import parazoa
import bbtree

import std/math
import std/sugar
import std/tables
import std/strformat
import std/strutils

import util
import option
import blendfuncs

def_numerical(FrameId, int)
def_numerical(LayerId, int)
def_numerical(FrameChangeId, int)
def_numerical(LayerChangeId, int)
def_numerical(CanvasChangeId, int)
def_numerical(MaskFrameId, int)
def_numerical(TimelinePos, int)

type
    ColorTile* = object
        case
        of Pixel: pixels*: ColorPixelArray
        of Fill: fill*: ColorPixel

    MaskTile* = object
        case
        of Pixel: pixels*: MaskPixelArray
        of Fill: fill*: MaskPixel

    Frame* = object of RootObj
        id* = 0.FrameId
        change_id* = 0.FrameChangeId
        change_map*: Map[VectorI, uint]
        offset*: VectorI
        pixel_size*: VectorI
        tile_size*: VectorI

    ColorFrame* = object of Frame
        tiles*: Map[VectorI, ColorTile]
        # Data that is written to when painting.
        # Updated parts gets copied to tiles after the action is finished.
        data*: ref seq[ColorPixelArray]

    MaskFrame* = object of Frame
        tiles*: Map[VectorI, MaskTile]
        data*: ref seq[MaskPixelArray]

    GroupObj* = object
        group_open*: bool = true
        last_layer*: Option[LayerId]
        first_layer*: Option[LayerId]
        # Temp buffer for compositing.
        temp*: ref seq[ColorPixelArray]

    Layer* = object
        id* = 0.LayerId
        parent_layer*: Option[LayerId]
        # Next and prev layers in current group, composite order.
        next_layer*: Option[LayerId]
        prev_layer*: Option[LayerId]
        change_id* = 0.LayerChangeId
        name*: string = ""
        blend_mode* = BlendMode.Normal
        opacity*: int16 = range_max
        visible*: bool = true
        mask_visible*: bool = false
        lock_alpha*: bool = false
        lock_draw*: bool = false
        lock_move*: bool = false
        lock_all*: bool = false
        clip*: bool = false
        masks*: BBTree[TimelinePos, MaskFrameId]
        case
        of Raster:
            frames*: BBTree[TimelinePos, FrameId]
        of Group:
            group*: GroupObj

    Canvas* = object
        next_layer_id* = 0.LayerId
        next_frame_id* = 0.FrameId
        # Composite order.
        first_layer*: Option[LayerId]
        last_layer*: Option[LayerId]
        change_id* = 0.CanvasChangeId
        # Recursive data is referenced indirectly through IDs instead of refs to simplfy immutable undo states.
        layer_lookup*: Map[LayerId, Layer]
        frame_lookup*: Map[FrameId, ColorFrame]
        mask_frame_lookup*: Map[MaskFrameId, MaskFrame]
        size*: VectorI
        has_selection*: bool
        selection*: MaskFrame
        background_color*: ColorRgbPixel
        background_transparent*: bool = false
        selected_frame*: Option[FrameId]
        multi_selected_frames*: Set[FrameId]
        selected_layer*: Option[LayerId]
        multi_selected_layers*: Set[LayerId]
        mask_selected*: bool
        fps*: int = 24
        start_frame* = 0.TimelinePos
        end_frame* = 24.TimelinePos

func get_blend_mode_name*(blend_mode: BlendMode): string =
    case blend_mode
    of BlendMode.Pass: "Pass"
    of BlendMode.Normal: "Normal"
    of BlendMode.Multiply: "Multiply"
    of BlendMode.Screen: "Screen"
    of BlendMode.Overlay: "Overlay"
    of BlendMode.LinearBurn: "Linear Burn"
    of BlendMode.LinearDodge: "Linear Dodge"
    of BlendMode.LinearLight: "Linear Light"
    of BlendMode.ColorBurn: "Color Burn"
    of BlendMode.ColorDodge: "Color Dodge"
    of BlendMode.VividLight: "Vivid Light"
    of BlendMode.SoftLight: "Soft Light"
    of BlendMode.HardLight: "Hard Light"
    of BlendMode.PinLight: "Pin Light"
    of BlendMode.HardMix: "Hard Mix"
    of BlendMode.Darken: "Darken"
    of BlendMode.Lighten: "Lighten"
    of BlendMode.DarkenColor: "Darken Color"
    of BlendMode.LightenColor: "Lighten Color"
    of BlendMode.Difference: "Difference"
    of BlendMode.Exclude: "Exclude"
    of BlendMode.Subtract: "Subtract"
    of BlendMode.Divide: "Divide"
    of BlendMode.Hue: "Hue"
    of BlendMode.Saturation: "Saturation"
    of BlendMode.Color: "Color"
    of BlendMode.Luminosity: "Luminosity"
    of BlendMode.TsLinearBurn: "TS Linear Burn"
    of BlendMode.TsLinearDodge: "TS Linear Dodge"
    of BlendMode.TsLinearLight: "TS Linear Light"
    of BlendMode.TsColorBurn: "TS Color Burn"
    of BlendMode.TsColorDodge: "TS Color Dodge"
    of BlendMode.TsVividLight: "TS Vivid Light"
    of BlendMode.TsHardMix: "TS Hard Mix"
    of BlendMode.TsDifference: "TS Difference"

func is_group*(layer: Layer): bool =
    case layer
    of Group: true
    else: false

func get*(canvas: Canvas, id: LayerId): Layer =
    ## Used to retrieve a known layer from layer_lookup.
    canvas.layer_lookup.get(id)

func get*(canvas: Canvas, id: Option[LayerId]): Option[Layer] =
    ## Used to retrieve layers from links.
    case id
    of Some(layer_id): some(canvas.get(layer_id))
    of None: None()

func update*(canvas: var Canvas, layer: Layer) =
    ## Convenience function to update a canvas layer reference.
    canvas.layer_lookup = canvas.layer_lookup.add(layer.id, layer)

func all_equal*[T](s: array[TILE_DATA_SIZE, T]): bool =
    let compared = s[0]
    for value in s:
        if value != compared:
            return false
    return true

func get_mask_frame*(canvas: Canvas, layer: Layer, timeline_pos: TimelinePos): Option[MaskFrame] =
    case layer.masks.get_or_prev(timeline_pos)
    of Some(mask_id): some(canvas.mask_frame_lookup.get(mask_id))
    of None: None()

func get_tile_data*[T](frame: T, row_size:int, tile_index: VectorI): MaskPixelArray =
    frame.data[get_linear_index(row_size, tile_index)]

iterator group_layers_iter*(canvas: Canvas, layer_id: Option[LayerId]): auto =
    ## Iterate children of this group in composite order.
    var layer_id = layer_id
    while true:
        case canvas.get(layer_id)
        of Some(layer):
            yield layer
            layer_id = layer.next_layer
        else:
            break

iterator reverse_group_layers_iter*(canvas: Canvas, layer_id: Option[LayerId]): auto =
    ## Iterate children of this group in list view order (reverse composite order).
    var layer_id = layer_id
    while true:
        case canvas.get(layer_id)
        of Some(layer):
            yield layer
            layer_id = layer.prev_layer
        else:
            break

iterator reverse_group_layers_iter*(canvas: Canvas, group: GroupObj): auto =
    for layer in reverse_group_layers_iter(canvas, group.last_layer):
        yield layer

iterator reverse_group_layers_iter*(canvas: Canvas): auto =
    for layer in reverse_group_layers_iter(canvas, canvas.last_layer):
        yield layer

iterator layers_iter(canvas: Canvas, layer_id: Option[LayerId]): auto {.closure.} =
    var layer_id = layer_id
    while true:
        case canvas.get(layer_id)
        of Some(layer):
            yield layer
            case layer:
            of Group(group):
                for sublayer in layers_iter(canvas, group.first_layer):
                    yield sublayer
            of Raster: discard
            layer_id = layer.next_layer
        of None:
            break

iterator layers_iter*(canvas: Canvas): auto =
    ## Iterate all layers in composite order.
    for layer in layers_iter(canvas, canvas.first_layer):
        yield layer

iterator reverse_layers_iter(canvas: Canvas, layer_id: Option[LayerId]): auto {.closure.} =
    var layer_id = layer_id
    while true:
        case canvas.get(layer_id)
        of Some(layer):
            yield layer
            case layer:
            of Group(group):
                for sublayer in reverse_layers_iter(canvas, group.last_layer):
                    yield sublayer
            of Raster: discard
            layer_id = layer.prev_layer
        of None:
            break

iterator reverse_layers_iter*(canvas: Canvas): auto =
    ## Iterate all layers in list view order (reverse composite order).
    for layer in reverse_layers_iter(canvas, canvas.last_layer):
        yield layer

func get_referenced_layers(canvas: Canvas, layer_id: Option[LayerId]): Set[LayerId] =
    for layer in layers_iter(canvas, layer_id):
        result = result.incl(layer.id)

func get_referenced_frames(canvas: Canvas): Set[FrameId] =
    for layer in canvas.layer_lookup.values:
        case layer
        of Raster(frames):
            for (_, frame_id) in frames.inorder:
                result = result.incl(frame_id)
        else: discard

func get_referenced_mask_frames(canvas: Canvas): Set[MaskFrameId] =
    for layer in canvas.layer_lookup.values:
        for (_, frame_id) in layer.masks.inorder:
            result = result.incl(frame_id)

func clean_map[K, V](m: Map[K, V], s: Set[K]): Map[K, V] =
    result = m
    for key in m.keys:
        if not s.contains(key):
            result = result.del(key)

func cleanup_references(canvas: Canvas): Canvas =
    result = canvas
    result.layer_lookup = clean_map(result.layer_lookup, get_referenced_layers(result, result.first_layer))
    result.frame_lookup = clean_map(result.frame_lookup, get_referenced_frames(result))
    result.mask_frame_lookup = clean_map(result.mask_frame_lookup, get_referenced_mask_frames(result))

func update_parent_links*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Update the parent Group's links to its first and last children of the given child layer.
    result = canvas
    let target_layer = result.get(target_layer_id)
    var maybe_parent = result.get(target_layer.parent_layer)
    case maybe_parent
    of Some(parent_layer):
        case parent_layer
        of Group(group):
            if target_layer.next_layer.is_none:
                group.last_layer = some(target_layer_id)
            if target_layer.prev_layer.is_none:
                group.first_layer = some(target_layer_id)
            result.update(parent_layer)
        else:
            assert(false, "Parent layer cannot be Raster")
    of None:
        if target_layer.next_layer.is_none:
            result.last_layer = some(target_layer_id)
        if target_layer.prev_layer.is_none:
            result.first_layer = some(target_layer_id)

func insert_after*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    ## Insert a layer after the target layer.
    result = canvas
    var inserting_layer = inserting_layer
    var target_layer = result.get(target_layer_id)
    var maybe_next_layer = result.get(target_layer.next_layer)
    case maybe_next_layer
    of Some(next_layer):
        next_layer.prev_layer = some(inserting_layer.id)
        inserting_layer.next_layer = some(next_layer.id)
        result.update(next_layer)
    of None:
        inserting_layer.next_layer = None()
    target_layer.next_layer = some(inserting_layer.id)
    inserting_layer.prev_layer = some(target_layer.id)
    inserting_layer.parent_layer = target_layer.parent_layer
    result.update(target_layer)
    result.update(inserting_layer)
    return update_parent_links(result, inserting_layer.id)

func insert_before*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    ## Insert a layer before the target layer.
    result = canvas
    var inserting_layer = inserting_layer
    var target_layer = result.get(target_layer_id)
    var maybe_prev_layer = result.get(target_layer.prev_layer)
    case maybe_prev_layer
    of Some(prev_layer):
        prev_layer.next_layer = some(inserting_layer.id)
        inserting_layer.prev_layer = some(prev_layer.id)
        result.update(prev_layer)
    of None:
        inserting_layer.prev_layer = None()
    target_layer.prev_layer = some(inserting_layer.id)
    inserting_layer.next_layer = some(target_layer.id)
    inserting_layer.parent_layer = target_layer.parent_layer
    result.update(target_layer)
    result.update(inserting_layer)
    return update_parent_links(result, inserting_layer.id)

func insert_into_group*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    ## Insert a layer into the target group in the last position.
    result = canvas
    var target_layer = result.get(target_layer_id)
    case target_layer
    of Group(group):
        case group.last_layer
        of Some(last_layer):
            result = insert_after(result, last_layer, inserting_layer)
        of None:
            group.last_layer = some(inserting_layer.id)
            group.first_layer = some(inserting_layer.id)
            var inserting_layer = inserting_layer
            inserting_layer.parent_layer = some(target_layer_id)
            result.update(target_layer)
            result.update(inserting_layer)
    of Raster:
        # How did this happen? Just insert after.
        result = insert_after(canvas, target_layer_id, inserting_layer)

func insert_new_to_selected_layer*(canvas: Canvas, inserting_layer: Layer): Canvas =
    ## Insert a new layer to the selected layer position.
    ## If the selected layer is a group, then it is placed at the last position (top).
    ## If the selected layer is a raster, then it is placed after the selected layer.
    ## The inserted layer becomes the new selected layer.
    result = canvas
    case result.selected_layer
    of Some(selected_layer_id):
        case result.get(selected_layer_id)
        of Group:
            result = insert_into_group(result, selected_layer_id, inserting_layer)
        of Raster:
            result = insert_after(result, selected_layer_id, inserting_layer)
    # In the "no selection" case, there are also no layers.
    of None:
        case result.last_layer
        # Something weird happened, there is no selection but layers do exist.
        # Try to insert after the last layer.
        of Some(last_layer_id):
            result = insert_after(result, last_layer_id, inserting_layer)
        of None:
            result.last_layer = some(inserting_layer.id)
            result.first_layer = some(inserting_layer.id)
    result.selected_layer = some(inserting_layer.id)
    result.multi_selected_layers = parazoa.toSet([inserting_layer.id])

func unlink_layer*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    # Removes the linkage of a layer from other layers.
    result = canvas
    var layer = result.get(target_layer_id)
    var maybe_next = result.get(layer.next_layer)
    case maybe_next
    of Some(next_layer):
        next_layer.prev_layer = layer.prev_layer
        result.update(next_layer)
    of None:
        var maybe_parent = result.get(layer.parent_layer)
        case maybe_parent
        of Some(parent):
            case parent
            of Group(group):
                group.last_layer = layer.prev_layer
                result.update(parent)
            else: discard
        of None:
            result.last_layer = layer.prev_layer

    var maybe_prev = result.get(layer.prev_layer)
    case maybe_prev
    of Some(prev_layer):
        prev_layer.next_layer = layer.next_layer
        result.update(prev_layer)
    of None:
        var maybe_parent = result.get(layer.parent_layer)
        case maybe_parent
        of Some(parent):
            case parent
            of Group(group):
                group.first_layer = layer.next_layer
                result.update(parent)
            else: discard
        of None:
            result.first_layer = layer.next_layer

func find_selected_ancenstor*(canvas: Canvas, target_layer_id: LayerId): LayerId =
    result = target_layer_id
    while true:
        var layer = canvas.get(result)
        var maybe_parent = canvas.get(layer.parent_layer)
        case maybe_parent
        of Some(parent):
            if parent.id in canvas.multi_selected_layers:
                result = parent.id
        of None:
            break

func filter_highest_selected*(canvas: Canvas): Set[LayerId] =
    for layer_id in canvas.multi_selected_layers:
        result = result.incl(find_selected_ancenstor(canvas, layer_id))

func sort_sub(canvas: Canvas, layers: Set[LayerId], current_layer: Option[LayerId]): Vec[LayerId] =
    var current_layer = current_layer
    while true:
        case canvas.get(current_layer)
        of Some(layer):
            if layer.id in layers:
                result = result.add(layer.id)
            else:
                case layer
                of Group(group):
                    result.add(sort_sub(canvas, layers, group.first_layer))
                of Raster: discard
            current_layer = layer.next_layer
        of None:
            break

func sorted_selection*(canvas: Canvas): Vec[LayerId] =
    sort_sub(canvas, filter_highest_selected(canvas), canvas.first_layer)

func get_next_available_selection*(canvas: Canvas, layers: Vec[LayerId]): Option[LayerId] =
    if layers.len == 0:
        return None()
    let last = canvas.get(layers.get(layers.len - 1))
    let first = canvas.get(layers.get(0))
    if last.next_layer.is_some:
        return last.next_layer
    if first.prev_layer.is_some:
        return first.prev_layer
    return last.parent_layer

func delete_layers*(canvas: Canvas): Canvas =
    # Unlink all selected layers and cleanup references.
    result = canvas
    let sorted_layers = sorted_selection(result)
    let next_selection = get_next_available_selection(result, sorted_layers)
    for layer in sorted_layers:
        result = unlink_layer(result, layer)
    result.selected_layer = next_selection
    case next_selection
    of Some(next_selection):
        result.multi_selected_layers = parazoa.toSet([next_selection])
    of None: discard
    return cleanup_references(result)

func move_layers*(canvas: Canvas, target_layer_id: LayerId, insert_op: ((Canvas, LayerId, Layer) {.noSideEffect.} -> Canvas)): Canvas =
    result = canvas
    let sorted_layers = sorted_selection(result)
    for layer_id in sorted_layers:
        let layer = result.get(layer_id)
        result = unlink_layer(result, layer_id)
        result = insert_op(result, target_layer_id, layer)

func move_layers_to_group*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection into a target group.
    move_layers(canvas, target_layer_id, insert_into_group)

func move_layers_after*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection to after the target layer.
    move_layers(canvas, target_layer_id, insert_after)

func move_layers_before*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection to before the target layer.
    move_layers(canvas, target_layer_id, insert_before)

func new_layer*(canvas: Canvas): Canvas =
    result = canvas
    let frame = ColorFrame(
        id: result.next_frame_id,
        data: new_tile_array[ColorPixelArray](canvas.size),
    )
    var layer = Raster(frames: singleton(0.TimelinePos, frame.id))
    layer.id = result.next_layer_id
    layer.name = fmt"Layer {layer.id.int}"
    result.next_frame_id += 1.FrameId
    result.next_layer_id += 1.LayerId
    result.frame_lookup = result.frame_lookup.add(frame.id, frame)
    result.layer_lookup = result.layer_lookup.add(layer.id, layer)
    result.selected_frame = Some(val: frame.id)
    return insert_new_to_selected_layer(result, layer)

func new_group*(canvas: Canvas): Canvas =
    result = canvas
    let frame = ColorFrame(
        id: result.next_frame_id,
        data: new_tile_array[ColorPixelArray](canvas.size),
    )
    var layer = Group(group: GroupObj(temp: new_tile_array[ColorPixelArray](canvas.size)))
    layer.id = result.next_layer_id
    layer.name = fmt"Group {layer.id.int}"
    result.next_frame_id += 1.FrameId
    result.next_layer_id += 1.LayerId
    result.frame_lookup = result.frame_lookup.add(frame.id, frame)
    result.layer_lookup = result.layer_lookup.add(layer.id, layer)
    result.selected_frame = Some(val: frame.id)
    return insert_new_to_selected_layer(result, layer)

func new_blank_canvas*(size: VectorI): Canvas =
    result.size = size
    return new_layer(result)

func debug_print_layer(layer: Layer, depth: int) =
    debugEcho indent(fmt"{layer.name} {get_blend_mode_name(layer.blend_mode)}", depth * 2)

func debug_print_layers_sub(canvas: Canvas, layer_id: Option[LayerId], depth: int) =
    var layer_id = layer_id
    while true:
        case canvas.get(layer_id)
        of Some(layer):
            debug_print_layer(layer, depth)
            case layer:
            of Group(group):
                debug_print_layers_sub(canvas, group.last_layer, depth + 1)
            of Raster: discard
            layer_id = layer.prev_layer
        of None:
            break

func debug_print_layers*(canvas: Canvas) =
    debugEcho "Canvas:"
    debug_print_layers_sub(canvas, canvas.last_layer, 1)
