## Types that describe the canvas document structure, layers, tiles, etc.
import fusion/matching
import std/options

import parazoa

import util

def_numerical(LayerId, uint)
def_numerical(FrameId, uint)
def_numerical(TimelinePos, uint)

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

    ChannelType* = int16

    ColorPixel* = Color[ChannelType]
    MaskPixel* = ChannelType

    ColorPixelArray* = array[TILE_SIZE, array[TILE_SIZE, ColorPixel]]
    MaskPixelArray* = array[TILE_SIZE, array[TILE_SIZE, MaskPixel]]

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
        timeline_pos* = 0.TimelinePos
    ColorFrame* = object of Frame
        tiles*: Map[Vector, ColorTile]
        temp*: ref seq[ColorPixel]
    MaskFrame* = object of Frame
        bg_white*: bool
        tiles*: Map[Vector, MaskTile]
        temp*: ref seq[MaskPixel]

    Layer* = object of RootObj
        id* = 0.LayerId
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
        masks*: Vec[FrameId]
    RasterLayer* = object of Layer
        frames*: Vec[FrameId]
    GroupLayer* = object of Layer
        group_open*: bool = true
        layers*: Vec[Layer]

    BackgroundSettings* = object
        color*: ColorPixel
        transparent*: bool = false
        checker*: bool = true
        checker_brightness*: uint8 = 127

    Canvas* = object
        frame_lookup*: Map[FrameId, ColorFrame]
        mask_frame_lookup*: Map[FrameId, MaskFrame]
        size*: Vector
        top_level*: Vec[Layer]
        selection*: Option[Map[Vector, MaskTile]]
        background_settings*: BackgroundSettings

func get_mask_tile*(canvas: Canvas, layer: Layer, frame_id: FrameId, offset: Vector): Option[(MaskFrame, MaskTile)] =
    case canvas.mask_frame_lookup.get_maybe(frame_id):
    of Some(@frame):
        case frame.tiles.get_maybe(offset):
        of Some(@tile):
            return some((frame, tile))

func all_equal*[T](s: seq[T]): bool =
    let compared = s[0]
    for value in s:
        if value != compared:
            return false
    return true

func get_tile_slice(canvas_size: Vector, tile_index: Vector): HSlice[int, int] =
    let index = tile_index.x * TILE_SIZE + tile_index.y * canvas_size.x
    let index_end = index + TILE_DATA_SIZE
    return index..index_end

func tile_from_temp*(frame: ColorFrame, canvas_size: Vector, tile_index: Vector): ColorTile =
    let chunk = frame.temp[get_tile_slice(canvas_size, tile_index)]
    return if all_equal(chunk):
        ColorFillTile(fill: chunk[0])
    else:
        ColorPixelTile(pixels: cast[ColorPixelArray](chunk))

func tile_from_temp*(frame: MaskFrame, canvas_size: Vector, tile_index: Vector): MaskTile =
    let chunk = frame.temp[get_tile_slice(canvas_size, tile_index)]
    return if all_equal(chunk):
        MaskFillTile(fill: chunk[0])
    else:
        MaskPixelTile(pixels: cast[MaskPixelArray](chunk))
