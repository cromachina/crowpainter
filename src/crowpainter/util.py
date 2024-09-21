import asyncio
from concurrent.futures import ThreadPoolExecutor

import psutil
import numpy as np
from pyrsistent import *
import psd_tools.constants as ptc
from psd_tools.api.layers import Layer

from .file_io import rle

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='WorkerThread')

def peval(func):
    return asyncio.get_running_loop().run_in_executor(pool, func)

def generate_tiles(size, tile_size):
    height, width = size
    tile_height, tile_width = tile_size
    y = 0
    while y < height:
        x = 0
        while x < width:
            size_y = min(tile_height, height - y)
            size_x = min(tile_width, width - x)
            yield ((size_y, size_x), (y, x))
            x += tile_width
        y += tile_height

def generate_points(size, tile_size):
    s = set()
    for (size_y, size_x), (y, x) in generate_tiles(size, tile_size):
        yy = y + size_y
        xx = x + size_x
        s.add((y, x))
        s.add((yy, x))
        s.add((y, xx))
        s.add((yy, xx))
    return s

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def get_overlap_tiles(dst, src, offset):
    oy, ox = offset
    dy, dx = dst.shape[:2]
    sy, sx = src.shape[:2]
    d_min_x = clamp(0, dx, ox)
    d_min_y = clamp(0, dy, oy)
    d_max_x = clamp(0, dx, ox + sx)
    d_max_y = clamp(0, dy, oy + sy)
    s_min_x = clamp(0, sx, -ox)
    s_min_y = clamp(0, sy, -oy)
    s_max_x = clamp(0, sx, dx - ox)
    s_max_y = clamp(0, sy, dy - oy)
    return dst[d_min_y:d_max_y, d_min_x:d_max_x], src[s_min_y:s_max_y, s_min_x:s_max_x]

def get_overlap_view(arr, size, offset):
    oy, ox = offset
    dy, dx = arr.shape[:2]
    sy, sx = size
    min_x = clamp(0, dx, ox)
    min_y = clamp(0, dy, oy)
    max_x = clamp(0, dx, ox + sx)
    max_y = clamp(0, dy, oy + sy)
    return arr[min_y:max_y, min_x:max_x]

def blit(dst, src, offset):
    dst, src = get_overlap_tiles(dst, src, offset)
    np.copyto(dst, src)

def bit(number, bit):
    return bool(number & (1 << bit))

class SystemStats(PRecord):
    own_memory_usage:int = field()
    system_memory_usage:int = field()
    disk_usage:int = field()

def get_system_stats():
    mem_stats = psutil.virtual_memory()
    system_total = mem_stats.total
    disk_total = 1
    disk_used = 0
    for mountpoint in {p.device:p.mountpoint for p in psutil.disk_partitions()}.values():
        usage = psutil.disk_usage(mountpoint)
        disk_total += usage.total
        disk_used += usage.used
    return SystemStats(
        own_memory_usage = int(psutil.Process().memory_info().rss / system_total * 100),
        system_memory_usage = int(mem_stats.percent),
        disk_usage = int(disk_used / disk_total * 100),
    )

def _parse_array(data, depth, lut=None):
    if depth == 8:
        parsed = np.frombuffer(data, ">u1")
        if lut is not None:
            parsed = lut[parsed]
        return parsed
    elif depth == 16:
        return (np.frombuffer(data, ">u2") / 256).astype(np.uint8)
    elif depth == 32:
        return (np.frombuffer(data, ">f4") * 256).astype(np.uint8)
    elif depth == 1:
        return np.unpackbits(np.frombuffer(data, np.uint8))
    else:
        raise ValueError("Unsupported depth: %g" % depth)

def layer_numpy(layer:Layer, channel=None):
    if channel == 'mask' and (not layer.mask or layer.mask.size != (0, 0)):
        return None

    depth = layer._psd.depth
    version = layer._psd.version

    def channel_matches(info):
        if channel == 'color':
            return info.id >= 0
        if channel == 'shape':
            return info.id == ptc.ChannelID.TRANSPARENCY_MASK
        if channel == 'mask':
            if not layer.mask:
                return False
            if layer.mask._has_real():
                return info.id == ptc.ChannelID.REAL_USER_LAYER_MASK
            else:
                return info.id == ptc.ChannelID.USER_LAYER_MASK
        else:
            raise ValueError(f'Unknown channel type: {channel}')

    channels = zip(layer._channels, layer._record.channel_info)
    channels = [channel for channel, info in channels if channel_matches(info)]

    if len(channels) == 0:
        return None

    # Use the psd-tools path if we are not decoding RLE
    if not all([channel.compression == ptc.Compression.RLE for channel in channels]):
        data = layer.numpy(channel)
        if data is not None:
            data = (data * 255.0).astype(np.uint8)
        return data

    if channel == 'mask':
        width, height = layer.mask.width, layer.mask.height
    else:
        width, height = layer.width, layer.height

    decoded = []
    for channel in channels:
        decoded.append(_parse_array(rle.decode_rle(channel.data, width, height, depth, version), depth))

    return np.stack(decoded, axis=1).reshape((height, width, -1))
