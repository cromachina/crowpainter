import psutil
import numpy as np
from pyrsistent import *
from . import constants

def is_subtype(value, dtype):
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, dtype)
    if np.isscalar(value):
        return np.issubdtype(np.array(value).dtype, dtype)
    if isinstance(value, type):
        return np.issubdtype(value, dtype)
    return False

def is_integer(value):
    return is_subtype(value, np.integer)

def is_floating(value):
    return is_subtype(value, np.floating)

# Convert to dtype with normalizing
def convert(value, dtype):
    if value is None:
        return None
    value_dtype = np.array(value).dtype
    if value_dtype == dtype:
        return value
    if is_integer(value) and is_integer(dtype):
        a = np.iinfo(value_dtype).max
        b = np.iinfo(dtype).max
        if b > a:
            return dtype(value) * dtype(b // a)
        else:
            return dtype(value // (a // b))
    if is_floating(value) and is_integer(dtype):
        return dtype(value * np.iinfo(dtype).max)
    if is_integer(value) and is_floating(dtype):
        return dtype(value) / np.iinfo(value_dtype).max
    if is_floating(value) and is_floating(dtype):
        return dtype(value)
    raise ValueError(f'Type conversion not implemented from {value_dtype} to {dtype}')

def to_display_dtype(x):
    return convert(x, constants.DISPLAY_DTYPE)

def to_storage_dtype(x):
    return convert(x, constants.STORAGE_DTYPE)

def to_blending_dtype(x):
    return convert(x, constants.BLENDING_DTYPE)

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
