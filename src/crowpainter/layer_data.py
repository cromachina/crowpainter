from __future__ import annotations
from typing import Self

from pyrsistent import *
import numpy as np

from .constants import *
from . import util

IVec2 = tuple[int, int]
DVec4 = tuple[float, float, float, float]

class SelectableObject(PClass):
    id:int = field(initial=0)

class BaseArrayTile(PClass):
    data:np.ndarray = field(mandatory=True)

    @classmethod
    def from_data(cls, data, lock=True):
        tile = cls(data, lock)
        if lock:
            tile.lock()
        return tile

    def lock(self):
        self.data.flags.writeable=False

class ColorTile(BaseArrayTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (3,), dtype=STORAGE_DTYPE), lock)

class AlphaTile(BaseArrayTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (1,), dtype=STORAGE_DTYPE), lock)

class FillTile(PClass):
    size:IVec2 = field()
    value:tuple | STORAGE_DTYPE = field()

class Mask(SelectableObject):
    position:IVec2 = field(initial=(0, 0))
    alpha:PMap[IVec2, AlphaTile] = field(initial=pmap())
    visible:bool = field(initial=True)
    background_color:STORAGE_DTYPE = field(initial=0)

    def get_mask_data(self, target_alpha_buffer:np.ndarray, target_offset:IVec2):
        mask = np.full(target_alpha_buffer.shape, self.background_color)
        for region in get_overlap_regions(self.alpha, self.position, mask, target_offset).values():
            np.copyto(*region)
        return mask

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
    mask:Mask | None = field(initial=None)

    def get_mask_data(self, target_alpha_buffer:np.ndarray, target_offset:IVec2):
        return None if self.mask is None else self.mask.get_mask_data(target_alpha_buffer, target_offset)

    def thaw(self):
        return self.set(mask=self.mask.thaw())

class PixelLayer(BaseLayer):
    position:IVec2 = field(initial=(0, 0))
    color:PMap[IVec2, ColorTile] = field(initial=pmap())
    alpha:PMap[IVec2, AlphaTile | FillTile] = field(initial=pmap())

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
    folder_open:bool = field(initial=True)

    def __iter__(self):
        return iter(self.layers)

    def __reversed__(self):
        return reversed(self.layers)

    def thaw(self):
        return self.set(layers=[layer.thaw() for layer in self.layers])

class BackgroundSettings(PClass):
    color:tuple = field(initial=(255, 255, 255))
    transparent:bool = field(initial=False)
    checker:bool = field(initial=True)
    checker_brightness:float = field(initial=0.5)

class Canvas(PClass):
    size:IVec2 = field(initial=(0, 0))
    top_level:PVector[BaseLayer] = field(initial=pvector())
    selection:Mask | None = field(initial=None)
    background:BackgroundSettings = field(initial=BackgroundSettings())

    def thaw(self):
        return self.set(
            top_level=[layer.thaw() for layer in self.top_level],
            selection=self.selection.thaw() if self.selection is not None else None
        )

def get_overlap_regions(tiles:PMap[IVec2, BaseArrayTile | FillTile], tiles_offset:IVec2, target_buffer:np.ndarray, target_offset:IVec2) -> dict[IVec2, tuple[np.ndarray, np.ndarray | STORAGE_DTYPE]]:
    regions = dict()
    relative_offset = np.array(tiles_offset) - np.array(target_offset)
    for point in util.generate_points(target_buffer.shape[:2], TILE_SIZE):
        point_offset = np.array(point) - relative_offset
        tile_index = point_offset // TILE_SIZE
        region_offset = (tile_index * TILE_SIZE) + relative_offset
        region = tiles.get(tuple(tile_index))
        if region is not None:
            if isinstance(region, FillTile):
                overlap_tiles = (util.get_overlap_view(target_buffer, region.size, tuple(region_offset)), region.value)
            else:
                overlap_tiles = util.get_overlap_tiles(target_buffer, region.data, tuple(region_offset))
            overlap_shape = overlap_tiles[0].shape[:2]
            if overlap_shape[0] != 0 and overlap_shape[1] != 0:
                absolute_offset = np.array(target_offset) + region_offset
                regions[tuple(absolute_offset)] = overlap_tiles
    return regions

def pixel_data_to_tiles(data:np.ndarray | None, tile_constructor):
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

def prune_tiles(color_tiles:PMap[IVec2, ColorTile], alpha_tiles:PMap[IVec2, AlphaTile]):
    new_color_tiles = {}
    new_alpha_tiles = {}
    for index,color_tile in color_tiles.items():
        alpha_tile = alpha_tiles.get(index)
        if alpha_tile is None or alpha_tile.data.any():
            new_color_tiles[index] = color_tile
            if alpha_tile is None:
                new_alpha_tiles[index] = FillTile(size=color_tile.data.shape[:2], value=255)
            else:
                new_alpha_tiles[index] = alpha_tile
    return pmap(new_color_tiles), pmap(new_alpha_tiles)

def color_alpha_to_tiles(color:np.ndarray, alpha:STORAGE_DTYPE | np.ndarray):
    color_tiles = pixel_data_to_tiles(color, ColorTile)
    if np.isscalar(alpha):
        alpha_tiles = scalar_to_tiles(alpha, color.shape[:2] + (1,), AlphaTile)
    else:
        alpha_tiles = pixel_data_to_tiles(alpha, AlphaTile)
    return color_tiles, alpha_tiles
