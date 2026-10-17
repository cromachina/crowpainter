# Reference: https://gitlab.com/define-private-public/stb_image-Nim
{.compile: "lib/stb_image.c".}

type
    ChannelType* {.size: sizeof(cint), pure.} = enum
        Default = 0,
        Gray = 1,
        GrayAlpha = 2,
        Rgb = 3,
        Rgba = 4,

{.push header: "../lib/stb_image.h".}

proc stbi_failure_reason(): cstring
proc stbi_load(filename: cstring, x, y, channels_in_file: var cint, desired_channels: ChannelType): ptr byte
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

    Image* = ref object
        width*: uint
        height*: uint
        channels*: uint
        data*: seq[byte]

func new_image*(width, height, channels: uint): Image =
    var image = Image(width: width, height:height, channels: channels)
    newSeq(image.data, width * height * channels)
    return image

func get_index(image: Image, x, y, channel: uint): uint =
    channel + (x * image.channels) + (y * image.width * image.channels)

func `[]`*(image: Image, x, y, channel: uint): byte =
    image.data[image.get_index(x, y, channel)]

func `[]=`*(image: Image, x, y, channel: uint, value: byte) =
    image.data[image.get_index(x, y, channel)] = value

proc failure_reason(): string =
    $stbi_failure_reason()

proc load*(filename: string, desired_channels: ChannelType): Image =
    var
        width, height, found_channels: cint
        data = stbi_load(filename, width, height, found_channels, desired_channels)
    if data == nil:
        raise newException(STBIException, failure_reason())
    defer: stbi_image_free(data)
    let channels = if desired_channels == ChannelType.Default: found_channels.uint else: desired_channels.uint
    var image = Image(width: width.uint, height: height.uint, channels: channels)
    newSeq(image.data, image.width * image.height * image.channels)
    copyMem(image.data[0].addr, data, image.data.len)
    return image

proc write_png*(filename: string, width, height, channels: uint, data: seq[byte], compression_level:int = 8, png_filter:int = -1, stride_in_bytes: int = 0): bool =
    stbi_write_png_compression_level = compression_level.cint
    stbi_write_force_png_filter = png_filter.cint
    stbi_write_png(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr, stride_in_bytes.cint).bool

proc write_png*(image: Image, filename: string, compression_level:int = 8, png_filter:int = -1, stride_in_bytes: int = 0): bool =
    write_png(filename, image.width, image.height, image.channels, image.data, compression_level, png_filter, stride_in_bytes)

proc write_bmp*(filename: string, width, height, channels: uint, data: seq[byte]): bool =
    stbi_write_bmp(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr).bool

proc write_bmp*(image: Image, filename: string): bool =
    write_bmp(filename, image.width, image.height, image.channels, image.data)

proc write_tga*(filename: string, width, height, channels: uint, data: seq[byte]): bool =
    stbi_write_tga(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr).bool

proc write_tga*(image: Image, filename: string): bool =
    write_tga(filename, image.width, image.height, image.channels, image.data)

proc write_jpg*(filename: string, width, height, channels: uint, data: seq[byte], quality: uint = 90): bool =
    stbi_write_jpg(filename, width.cint, height.cint, channels.cint, data[0].unsafeAddr, quality.cint).bool

proc write_jpg*(image: Image, filename: string, quality: uint = 90): bool =
    write_jpg(filename, image.width, image.height, image.channels, image.data, quality)
