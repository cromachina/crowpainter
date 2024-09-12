import asyncio
from concurrent.futures import ThreadPoolExecutor

import psutil
import numpy as np

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
