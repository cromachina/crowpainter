import psutil
import weakref
import gc
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyrsistent import *

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(worker_count)

def peval(func, *args, **kwargs):
    return asyncio.get_running_loop().run_in_executor(pool, lambda: func(*args, **kwargs))

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

def set_bits(bit_values):
    number = 0
    for bit, value in bit_values:
        b = 1 if value else 0
        number |= (b << bit)
    return number

def get_color(arr:np.ndarray):
    return arr[:,:,:3]

def get_alpha(arr:np.ndarray):
    return arr[:,:,3:]

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

def trackfinal(obj, message):
    weakref.finalize(obj, print, "finalized:", message)

def memory_usage():
    return psutil.Process().memory_info().rss

def get_gc_object_stats():
    stats = Counter()
    for obj in gc.get_objects():
        stats[type(obj)] += 1
    return stats

def get_changed_stats(stats_old, stats_new):
    all_stats = (stats_old | stats_new).keys()
    stats = {}
    for stat in all_stats:
        delta = stats_new[stat] - stats_old[stat]
        current = stats_new[stat]
        stats[stat] = (delta, current)
    return stats

def print_gc_object_stats(stats):
    print('Objects:')
    for stat, count in sorted(stats.items(), key=lambda x: str(x[0])):
        if count[0] != 0:
            print(f'  ({count[0]:+}, {count[1]}): {stat}')

_current_gc_stats = get_gc_object_stats()

def update_memory_tracking():
    global _current_gc_stats
    old_stats = _current_gc_stats
    _current_gc_stats = get_gc_object_stats()
    print_gc_object_stats(get_changed_stats(old_stats, _current_gc_stats))
    print(f'Memory usage: {memory_usage()}')

class ProgressCounter:
    def __init__(self, total, progress_callback):
        self.count = 0
        self.total = total
        self.callback = progress_callback

    def update(self):
        if self.callback is not None:
            self.count += 1
            self.callback(self.count / self.total)
