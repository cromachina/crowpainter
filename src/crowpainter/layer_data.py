from __future__ import annotations
from typing import Self

from pyrsistent import *
import numpy as np

from .constants import *
from . import util

IVec2 = tuple[int, int]
DVec4 = tuple[float, float, float, float]

class SelectableObject(PClass):
    id:int = field()

class BaseTile(PClass):
    data:np.ndarray = field(mandatory=True)

    @classmethod
    def from_data(cls, data, lock=True):
        tile = cls(data, lock)
        if lock:
            tile.lock()
        return tile

    def lock(self):
        self.data.flags.writeable=False

class ColorTile(BaseTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (3,), dtype=STORAGE_DTYPE), lock)

class AlphaTile(BaseTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (1,), dtype=STORAGE_DTYPE), lock)

class Mask(SelectableObject):
    position:IVec2 = field(initial=(0, 0))
    alpha:PMap[IVec2, AlphaTile] = field(initial=pmap())
    visible:bool = field(initial=True)
    background_color:STORAGE_DTYPE = field(initial=0)

    # TODO if mask data partially covers target buffer, then it needs to be blitted onto the background color
    # to completely fill the area of the target buffer.
    def get_mask_data(self, target_alpha_buffer:np.ndarray, target_offset:IVec2):
        return get_overlap_regions(self.alpha, self.position, target_alpha_buffer, target_offset)

    def thaw(self):
        return self.set(alpha=thaw(self.alpha))

class BaseLayer(SelectableObject):
    name:str = field(initial="")
    blend_mode:BlendMode = field(initial=BlendMode.NORMAL)
    visible:bool = field(initial=True)
    opacity:float = field(initial=255)
    lock_alpha:bool = field(initial=False)
    lock_draw:bool = field(initial=False)
    lock_move:bool = field(initial=False)
    lock_all:bool = field(initial=False)
    clip:bool = field(initial=False)
    clip_layers:PVector[Self] | None = field(initial=None)
    mask:Mask | None = field(initial=None)

    def get_mask_data(self, target_alpha_buffer:np.ndarray, target_offset:IVec2):
        return None if self.mask is None else self.mask.get_mask_data(target_alpha_buffer, target_offset)

    def thaw(self):
        return self.set(mask=self.mask.thaw())

class PixelLayer(BaseLayer):
    position:IVec2 = field(initial=(0, 0))
    color:PMap[IVec2, ColorTile] = field(initial=pmap())
    alpha:PMap[IVec2, AlphaTile] = field(initial=pmap())

    def get_pixel_data(self, target_color_buffer:np.ndarray, target_alpha_buffer:np.ndarray, target_offset:IVec2):
        color = get_overlap_regions(self.color, self.position, target_color_buffer, target_offset)
        alpha = get_overlap_regions(self.alpha, self.position, target_alpha_buffer, target_offset)
        d = {}
        for k,c in color.items():
            aa = alpha.get(k)
            if aa is None:
                aa = np.ones_like(c)
            d[k] = (c, aa)
        return d

    def thaw(self):
        return self.set(color=thaw(self.color), alpha=thaw(self.color))

class GroupLayer(BaseLayer):
    layers:PVector[BaseLayer] = field(initial=pvector())

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)

    def thaw(self):
        return self.set(layers=[layer.thaw() for layer in self.layers])

class Canvas(PClass):
    size:IVec2 = field(initial=(0, 0))
    top_level:PVector[BaseLayer] = field(initial=pvector())
    selection:Mask | None = field(initial=None)

    def thaw(self):
        return self.set(
            top_level=[layer.thaw() for layer in self.top_level],
            selection=self.selection.thaw() if self.selection is not None else None
        )

def get_overlap_regions(tiles:PMap[IVec2, BaseTile], tiles_offset:IVec2, target_buffer:np.ndarray, target_offset:IVec2) -> dict[IVec2, tuple[np.ndarray, np.ndarray]]:
    regions = dict()
    relative_offset = np.array(tiles_offset) - np.array(target_offset)
    for point in util.generate_points(target_buffer.shape[:2], TILE_SIZE):
        point_offset = np.array(point) - relative_offset
        tile_index = point_offset // TILE_SIZE
        region_offset = (tile_index * TILE_SIZE) + relative_offset
        region = tiles.get(tuple(tile_index))
        if region is not None:
            overlap_tiles = util.get_overlap_tiles(target_buffer, region.data, tuple(region_offset))
            overlap_shape = overlap_tiles[0].shape[:2]
            if overlap_shape[0] != 0 and overlap_shape[1] != 0:
                regions[tuple(tile_index)] = overlap_tiles
    return regions

def pixel_data_to_tiles(data:np.ndarray, tile_constructor):
    if data is None:
        return pmap()
    tiles = {}
    for (size, offset) in util.generate_tiles(data.shape[:2], TILE_SIZE):
        offset = np.array(offset)
        tile = np.zeros(shape=size + data.shape[2:], dtype=STORAGE_DTYPE)
        util.blit(tile, data, -offset)
        index = tuple(offset // TILE_SIZE)
        tiles[index] = tile_constructor(data=tile)
    return pmap(tiles)

def scalar_to_tiles(value, shape, tile_constructor):
    tiles = {}
    for (size, offset) in util.generate_tiles(shape[:2], TILE_SIZE):
        tile = np.full(shape=size + shape[2:], dtype=STORAGE_DTYPE, fill_value=value)
        index = tuple(np.array(offset) // TILE_SIZE)
        tiles[index] = tile_constructor(data=tile)
    return pmap(tiles)

def color_alpha_to_tiles(color:np.ndarray, alpha:float | np.ndarray):
    color_tiles = pixel_data_to_tiles(color, ColorTile)
    if np.isscalar(alpha):
        alpha_tiles = scalar_to_tiles(alpha, color.shape[:2] + (1,), AlphaTile)
    else:
        alpha_tiles = pixel_data_to_tiles(alpha, AlphaTile)
    return color_tiles, alpha_tiles
