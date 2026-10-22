# Reference: https://gitlab.com/define-private-public/stb_image-Nim
{.compile: "lib/stb_image.c".}

import util


{.push header: "../lib/stb_image.h".}

proc stbi_failure_reason(): cstring
proc stbi_load(filename: cstring, x, y, channels_in_file: var cint, desired_channels: cint): ptr byte
proc stbi_image_free(retval_from_stbi_load: ptr byte): void

{.pop.}

{.push header: "../lib/stb_image_write.h".}

# bool (default: true)
var stbi_write_tga_with_rle: cint
# 0 .. unbounded (default: 8)
var stbi_write_png_compression_level: cint
# -1 .. 5 (default: -1)
var stbi_write_force_png_filter:cint

proc stbi_write_png(filename: cstring, w, h, comp: cint, data: pointer, stride_in_bytes: cint): cint
proc stbi_write_bmp(filename: cstring, w, h, comp: cint, data: pointer): cint
proc stbi_write_tga(filename: cstring, w, h, comp: cint, data: pointer): cint
proc stbi_write_hdr(filename: cstring, w, h, comp: cint, data: ptr cfloat): cint
proc stbi_write_jpg(filename: cstring, w, h, comp: cint, data: pointer, quality: cint): cint

{.pop.}

type
    STBIException* = object of ValueError

    # Image = concept i, type T
    #     i.size is VectorI
    #     i.data is seq[T]

    ImageGray* = ref object
        size*: VectorI
        data*: ref seq[byte]
    ImageGrayAlpha* = ref object
        size*: VectorI
        data*: ref seq[ColorGrayAlpha[byte]]
    ImageRgb* = ref object
        size*: VectorI
        data*: ref seq[ColorRgb[byte]]
    ImageRgba* = ref object
        size*: VectorI
        data*: ref seq[Color[byte]]

func new_image*[T](width, height: int): T =
    result.new
    result.size.x = width.int
    result.size.y = height.int
    result.data.new
    result.data[].setLenUninit(width * height)

func get_channels*(t: typedesc[ImageGray]): int = 1
func get_channels*(t: typedesc[ImageGrayAlpha]): int = 2
func get_channels*(t: typedesc[ImageRgb]): int = 3
func get_channels*(t: typedesc[ImageRgba]): int = 4

func get_index*[T](image: T, x, y: int): int =
    x + (y * image.size.x)

func `[]`*[T](image: T, x, y: int): T =
    image.data[image.get_index(x, y)]

func `[]=`*[T](image: T, x, y: int, value: T) =
    image.data[image.get_index(x, y)] = value

proc failure_reason(): string =
    $stbi_failure_reason()

proc load*[T](filename: string): T =
    var
        width, height, found_channels_unused: cint
        data = stbi_load(filename, width, height, found_channels_unused, get_channels(T).cint)
    if data == nil:
        raise newException(STBIException, failure_reason())
    defer: stbi_image_free(data)
    var image = new_image[T](width, height)
    copyMem(image.data[][0].addr, data, image.data[].len * sizeof(image.data[][0]))
    return image

proc write_png*(filename: string, width, height, channels: int, data: seq[byte], compression_level:int = 8, png_filter:int = -1, stride_in_bytes: int = 0): bool =
    stbi_write_png_compression_level = compression_level.cint
    stbi_write_force_png_filter = png_filter.cint
    stbi_write_png(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr, stride_in_bytes.cint).bool

proc write_bmp*(filename: string, width, height, channels: int, data: seq[byte]): bool =
    stbi_write_bmp(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr).bool

proc write_tga*(filename: string, width, height, channels: int, data: seq[byte]): bool =
    stbi_write_tga(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr).bool

proc write_jpg*(filename: string, width, height, channels: int, data: seq[byte], quality: int = 90): bool =
    stbi_write_jpg(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr, quality.cint).bool

proc write_png*[T](image: T, filename: string, compression_level:int = 8, png_filter:int = -1, stride_in_bytes: int = 0): bool =
    write_png(filename, image.size.x, image.size.y, get_channels(T), cast[seq[byte]](image.data[]), compression_level, png_filter, stride_in_bytes)

proc write_bmp*[T](image: T, filename: string): bool =
    write_bmp(filename, image.size.x, image.size.y, get_channels(T), cast[seq[byte]](image.data[]))

proc write_tga*[T](image: T, filename: string): bool =
    write_tga(filename, image.size.x, image.size.y, get_channels(T), cast[seq[byte]](image.data[]))

proc write_jpg*[T](image: T, filename: string, quality: int = 90): bool =
    write_jpg(filename, image.size.x, image.size.y, get_channels(T), cast[seq[byte]](image.data[]), quality)
