## Types that describe the canvas document structure, layers, tiles, etc.
import parazoa
import bbtree

import util
import std/math

import option

def_numerical(LayerId, int)
def_numerical(FrameId, int)
def_numerical(MaskFrameId, int)
def_numerical(TimelinePos, int)

const TILE_SIZE* = 32
const TILE_DATA_SIZE* = TILE_SIZE * TILE_SIZE

type
    BlendMode* {.pure.} = enum
        Pass,
        Normal,
        Multiply,
        Screen,
        Overlay,
        LinearBurn,
        LinearDodge,
        LinearLight,
        ColorBurn,
        ColorDodge,
        VividLight,
        SoftLight,
        HardLight,
        PinLight,
        HardMix,
        Darken,
        Lighten,
        DarkenColor,
        LightenColor,
        Difference,
        Exclude,
        Subtract,
        Divide,
        Hue,
        Saturation,
        Color,
        Luminosity,
        TsLinearBurn,
        TsLinearDodge,
        TsLinearLight,
        TsColorBurn,
        TsColorDodge,
        TsVividLight,
        TsHardMix,
        TsDifference,

    ColorPixelArray* = array[TILE_DATA_SIZE, ColorPixel]
    MaskPixelArray* = array[TILE_DATA_SIZE, MaskPixel]

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
        change_id*: uint = 0
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

    Layer* = object
        id* = 0.LayerId
        parent_layer*: Option[LayerId]
        # Next and prev layers in current group
        next_layer*: Option[LayerId]
        prev_layer*: Option[LayerId]
        change_id*: uint = 0
        name*: string = ""
        blend_mode* = BlendMode.Normal
        opacity*: uint8 = 255
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
            group_open*: bool = true
            last_layer*: Option[LayerId]
            first_layer*: Option[LayerId]
            # Temp buffer for compositing.
            temp*: ref seq[ColorPixelArray]

    BackgroundSettings* = object
        color*: ColorPixel
        transparent*: bool = false
        checker*: bool = true
        checker_brightness*: uint8 = 127

    # When frames are used as clones, keeping track of ref counts
    # will help clean them up from the lookup tables when they are deleted.
    RefCounted*[T] = object
        count*: int = 1
        value*: T

    Canvas* = object
        next_layer_id* = 0.LayerId
        next_frame_id* = 0.FrameId
        last_toplevel_layer*: Option[LayerId]
        first_toplevel_layer*: Option[LayerId]
        change_id*: uint = 0
        # Recursive data is referenced indirectly through IDs instead of refs to simplfy undo states.
        layer_lookup*: Map[LayerId, Layer]
        frame_lookup*: Map[FrameId, RefCounted[ColorFrame]]
        mask_frame_lookup*: Map[MaskFrameId, RefCounted[MaskFrame]]
        size*: VectorI
        has_selection*: bool
        selection*: MaskFrame
        background_settings*: BackgroundSettings
        selected_frame*: Option[FrameId]
        multi_selected_frames*: seq[FrameId]
        selected_layer*: Option[LayerId]
        multi_selected_layers*: seq[LayerId]
        mask_selected*: bool
        fps: int = 24
        start_frame: int = 0
        end_frame: int = 24

func get*(canvas: Canvas, id: LayerId): Layer =
    canvas.layer_lookup.get(id)

func get*(canvas: Canvas, id: Option[LayerId]): Option[Layer] =
    case id
    of Some(layer_id): Some(val: canvas.get(layer_id))
    of None: None()

func update*(canvas: var Canvas, layer: Layer) =
    canvas.layer_lookup = canvas.layer_lookup.add(layer.id, layer)

func update*[K, V](m: Map[K, RefCounted[V]], k: K, v: V): Map[K, RefCounted[V]] =
    case m.get_maybe(k)
    of Some(val):
        var nval = val
        nval.value = v
        m.add(k, nval)
    of None: m.add(k, RefCounted[V](value: v))

func incr_ref*[K, V](m: Map[K, RefCounted[V]], k: K): Map[K, RefCounted[V]] =
    var val = m.get(k)
    val.count += 1
    m.add(k, val)

func decr_ref*[K, V](map: Map[K, RefCounted[V]], k: K): Map[K, RefCounted[V]] =
    var val = map.get(k)
    val.count -= 1
    if val.count <= 0:
        map.del(k)
    else:
        map.add(k, val)

func get_value*[K, V](map: Map[K, RefCounted[V]], k: K): V =
    map.get(k).value

func all_equal*[T](s: array[TILE_DATA_SIZE, T]): bool =
    let compared = s[0]
    for value in s:
        if value != compared:
            return false
    return true

func get_mask_frame*(canvas: Canvas, layer: Layer, timeline_pos: TimelinePos): Option[MaskFrame] =
    case layer.masks.get_or_prev(timeline_pos)
    of Some(mask_id): Some(val: canvas.mask_frame_lookup.get(mask_id).value)
    of None: None()

func get_index(row_size: int, tile_index: VectorI): int =
    tile_index.y * row_size + tile_index.x

func get_tile_data*[T](frame: T, row_size:int, tile_index: VectorI): MaskPixelArray =
    frame.data[get_index(row_size, tile_index)]

func get_tiles_size*(pixel_width: int): int =
    let (res, rem) = divmod(pixel_width, TILE_SIZE)
    if rem > 0: res + 1 else: res

func get_tiles_size*(pixel_size: VectorI): VectorI =
    VectorI(x:get_tiles_size(pixel_size.x), y:get_tiles_size(pixel_size.y))

func get_pixels_size*(tile_size: VectorI): VectorI =
    tile_size * TILE_SIZE

func new_tile_array*[T](pixel_size: VectorI): ref seq[T] =
    let tiles = get_tiles_size(pixel_size)
    new_ref_seq[T](tiles.x * tiles.y)

func update_parent_links*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    result = canvas
    var target_layer = result.get(target_layer_id)
    var maybe_parent = result.get(target_layer.parent_layer)
    case maybe_parent
    of Some(parent_layer):
        case parent_layer
        of Group(_, last_layer, first_layer):
            if target_layer.next_layer.is_none:
                last_layer = Some(val: target_layer_id)
            if target_layer.prev_layer.is_none:
                first_layer = Some(val: target_layer_id)
            result.update(parent_layer)
        else:
            assert(false, "Parent layer cannot be Raster")
    of None:
        if target_layer.next_layer.is_none:
            result.last_toplevel_layer = Some(val: target_layer_id)
        if target_layer.prev_layer.is_none:
            result.first_toplevel_layer = Some(val: target_layer_id)

func insert_after*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    result = canvas
    var inserting_layer = inserting_layer
    var target_layer = result.get(target_layer_id)
    var maybe_next_layer = result.get(target_layer.next_layer)
    case maybe_next_layer
    of Some(next_layer):
        next_layer.prev_layer = Some(val: inserting_layer.id)
        inserting_layer.next_layer = Some(val: next_layer.id)
        result.update(next_layer)
    of None:
        inserting_layer.next_layer = None()
    target_layer.next_layer = Some(val: inserting_layer.id)
    inserting_layer.prev_layer = Some(val: target_layer.id)
    inserting_layer.parent_layer = target_layer.parent_layer
    result.update(target_layer)
    result.update(inserting_layer)
    return update_parent_links(result, inserting_layer.id)

func insert_before*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    result = canvas
    var inserting_layer = inserting_layer
    var target_layer = result.get(target_layer_id)
    var maybe_prev_layer = result.get(target_layer.prev_layer)
    case maybe_prev_layer
    of Some(prev_layer):
        prev_layer.next_layer = Some(val: inserting_layer.id)
        inserting_layer.prev_layer = Some(val: prev_layer.id)
        result.update(prev_layer)
    of None:
        inserting_layer.prev_layer = None()
    target_layer.prev_layer = Some(val: inserting_layer.id)
    inserting_layer.next_layer = Some(val: target_layer.id)
    inserting_layer.parent_layer = target_layer.parent_layer
    result.update(target_layer)
    result.update(inserting_layer)
    return update_parent_links(result, inserting_layer.id)

func insert_into_group*(canvas: Canvas, target_layer_id: LayerId, inserting_layer: Layer): Canvas =
    var target_layer = canvas.get(target_layer_id)
    case target_layer
    of Group(_, last_layer, first_layer):
        case last_layer
        of Some(last_layer):
            result = insert_after(canvas, last_layer, inserting_layer)
        of None:
            result = canvas
            last_layer = Some(val: inserting_layer.id)
            first_layer = Some(val: inserting_layer.id)
            var inserting_layer = inserting_layer
            inserting_layer.parent_layer = Some(val: target_layer_id)
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
    case canvas.selected_layer
    of Some(selected_layer_id):
        var selected_layer = canvas.get(selected_layer_id)
        case selected_layer
        of Group():
            result = insert_into_group(canvas, selected_layer_id, inserting_layer)
        of Raster():
            result = insert_after(canvas, selected_layer_id, inserting_layer)
    # In the "no selection" case, there are also no layers.
    of None:
        case result.last_toplevel_layer
        # Something weird happened, there is no selection but layers do exist.
        # Try to insert after the last layer.
        of Some(last_layer_id):
            result = insert_after(canvas, last_layer_id, inserting_layer)
        of None:
            result = canvas
            result.last_toplevel_layer = Some(val: inserting_layer.id)
            result.first_toplevel_layer = Some(val: inserting_layer.id)
    result.selected_layer = Some(val: inserting_layer.id)
    result.multi_selected_layers.setLen(0)

func move_layers_to_group*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection into a target group.
    discard

func move_layers_after*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection to after the target layer.
    discard

func move_layers_before*(canvas: Canvas, target_layer_id: LayerId): Canvas =
    ## Move the current selection to before the target layer.
    discard