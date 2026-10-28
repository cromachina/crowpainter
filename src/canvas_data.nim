## Types that describe the canvas document structure, layers, tiles, etc.
import fusion/matching
import std/options

import parazoa
import bbtree

import util
import std/math

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

    Tile* = object of RootObj
    ColorTile* = object of Tile
    ColorPixelTile* = object of ColorTile
        pixels*: ColorPixelArray
    ColorFillTile = object of ColorTile
        fill*: ColorPixel

    MaskTile* = object of Tile
    MaskPixelTile = object of MaskTile
        pixels*: MaskPixelArray
    MaskFillTile = object of MaskTile
        fill*: MaskPixel

    Frame* = object of RootObj
        id* = 0.FrameId
        change_id*: uint = 0
    ColorFrame* = object of Frame
        tiles*: Map[VectorI, ColorTile]
        temp*: ref seq[ColorPixelArray]
    MaskFrame* = object of Frame
        tiles*: Map[VectorI, MaskTile]
        temp*: ref seq[MaskPixelArray]

    Layer* = object of RootObj
        id* = 0.LayerId
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
    RasterLayer* = object of Layer
        frames*: BBTree[TimelinePos, FrameId]
    GroupLayer* = object of Layer
        group_open*: bool = true
        layers*: Vec[LayerId]

    BackgroundSettings* = object
        color*: ColorPixel
        transparent*: bool = false
        checker*: bool = true
        checker_brightness*: uint8 = 127

    Canvas* = object
        next_layer_id* = 0.LayerId
        next_frame_id* = 0.FrameId
        change_id*: uint = 0
        layer_lookup*: Map[LayerId, Layer]
        frame_lookup*: Map[FrameId, ColorFrame]
        mask_frame_lookup*: Map[MaskFrameId, MaskFrame]
        size*: VectorI
        top_level*: Vec[LayerId]
        selection*: Option[Map[VectorI, MaskTile]]
        background_settings*: BackgroundSettings
        selected_frame*: FrameId
        multi_selected_frames*: seq[FrameId]
        selected_layer*: LayerId
        multi_selected_layers*: seq[LayerId]
        mask_selected*: bool
        fps: int = 24
        start_frame: int = 0
        end_frame: int = 24

func all_equal*[T](s: array[TILE_DATA_SIZE, T]): bool =
    let compared = s[0]
    for value in s:
        if value != compared:
            return false
    return true

func get_mask_frame*(canvas: Canvas, layer: Layer, timeline_pos: TimelinePos): Option[MaskFrame] =
    case layer.masks.get_or_prev(timeline_pos):
    of Some(@mask_id):
        return some(canvas.mask_frame_lookup.get(mask_id))

func get_index(row_size: int, tile_index: VectorI): int =
    tile_index.y * row_size + tile_index.x

func get_temp_tile*[T](frame: T, row_size:int, tile_index: VectorI): MaskPixelArray =
    frame.temp[get_index(row_size, tile_index)]

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
