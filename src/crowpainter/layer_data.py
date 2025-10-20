from __future__ import annotations
from typing import Self

from pyrsistent import *
import numpy as np

from .constants import *
from . import util, blendfuncs

IVec2 = tuple[int, int]
DVec4 = tuple[float, float, float, float]

class SelectableObject(PClass):
    id:int = field(initial=0)

class BaseArrayTile(PClass):
    data:np.ndarray = field(mandatory=True)

    @classmethod
    def from_data(cls, data, lock=True):
        tile = cls(data=data)
        if lock:
            tile.lock()
        return tile

    def lock(self):
        self.data.flags.writeable=False

class ColorTile(BaseArrayTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (4,), dtype=blendfuncs.dtype), lock)

class AlphaTile(BaseArrayTile):
    def make(size:IVec2, lock=True):
        return Self.from_data(np.zeros(size + (1,), dtype=blendfuncs.dtype), lock)

class FillTile(PClass):
    size:IVec2 = field()
    value:np.ndarray | blendfuncs.dtype = field()

class Mask(SelectableObject):
    position:IVec2 = field(initial=(0, 0))
    alpha:PMap[IVec2, AlphaTile] = field(initial=pmap())
    visible:bool = field(initial=True)
    background_color:blendfuncs.dtype = field(initial=blendfuncs.dtype(0))

    def get_mask_data(self, size:IVec2, target_offset:IVec2):
        mask = np.full(size + (1,), self.background_color, dtype=blendfuncs.dtype)
        masks = get_overlap_regions(self.alpha, self.position, mask, target_offset)
        if not masks:
            return None
        for region in masks.values():
            np.copyto(*region)
        return mask

    def thaw(self):
        return self.set(alpha=thaw(self.alpha))

class BaseLayer(SelectableObject):
    name:str = field(initial="")
    blend_mode:BlendMode = field(initial=BlendMode.NORMAL)
    visible:bool = field(initial=True)
    opacity:blendfuncs.dtype = field(initial=blendfuncs.get_max())
    lock_alpha:bool = field(initial=False)
    lock_draw:bool = field(initial=False)
    lock_move:bool = field(initial=False)
    lock_all:bool = field(initial=False)
    clip:bool = field(initial=False)
    mask:Mask | None = field(initial=None)

    def get_mask_data(self, size:IVec2, target_offset:IVec2):
        return None if self.mask is None else self.mask.get_mask_data(size, target_offset)

    def thaw(self):
        return self.set(mask=self.mask.thaw())

class PixelLayer(BaseLayer):
    position:IVec2 = field(initial=(0, 0))
    color:PMap[IVec2, ColorTile] = field(initial=pmap())

    def get_pixel_data(self, target_color_buffer:np.ndarray, target_offset:IVec2):
        return get_overlap_regions(self.color, self.position, target_color_buffer, target_offset)

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
    color:tuple = field(initial=np.full(shape=(3,), fill_value=blendfuncs.get_max(), dtype=blendfuncs.dtype))
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

def get_overlap_regions(tiles:PMap[IVec2, BaseArrayTile | FillTile], tiles_offset:IVec2, target_buffer:np.ndarray, target_offset:IVec2) -> dict[IVec2, tuple[np.ndarray, np.ndarray | blendfuncs.dtype]]:
    regions = dict()
    relative_offset = np.array(tiles_offset) - np.array(target_offset)
    for point in util.generate_points(target_buffer.shape[:2], TILE_SIZE):
        point_offset = np.array(point) - relative_offset
        tile_index = point_offset // TILE_SIZE
        tile_offset = (tile_index * TILE_SIZE) + relative_offset
        tile = tiles.get(tuple(tile_index))
        if tile is not None:
            if isinstance(tile, FillTile):
                overlap_tiles = (util.get_overlap_view(target_buffer, tile.size, tuple(tile_offset)), tile.value)
            else:
                overlap_tiles = util.get_overlap_tiles(target_buffer, tile.data, tuple(tile_offset))
            overlap_shape = overlap_tiles[0].shape[:2]
            if overlap_shape[0] != 0 and overlap_shape[1] != 0:
                tile_abs_offset = tile_index * TILE_SIZE + tiles_offset
                absolute_offset = np.maximum(np.array(target_offset), tile_abs_offset)
                regions[tuple(absolute_offset)] = overlap_tiles
    return regions

def pixel_data_to_tiles(data:np.ndarray | None):
    if data is None:
        return pmap()
    tiles = {}
    for (size, offset) in util.generate_tiles(data.shape[:2], TILE_SIZE):
        offset = np.array(offset)
        tile = np.zeros(shape=size + data.shape[2:], dtype=blendfuncs.dtype)
        util.blit(tile, data, -offset)
        index = tuple(offset // TILE_SIZE)
        tiles[index] = ColorTile.from_data(tile)
    return pmap(tiles)

def scalar_to_tiles(value, shape, tile_constructor):
    tiles = {}
    for (size, offset) in util.generate_tiles(shape[:2], TILE_SIZE):
        tile = np.full(shape=size + shape[2:], fill_value=value, dtype=blendfuncs.dtype)
        index = tuple(np.array(offset) // TILE_SIZE)
        tiles[index] = tile_constructor.from_data(tile)
    return pmap(tiles)

def prune_tiles(tiles:PMap[IVec2, ColorTile]):
    new_tiles = {}
    for index, tile in tiles.items():
        alpha = util.get_alpha(tile.data)
        has_alpha = alpha.any()
        if has_alpha:
            new_tiles[index] = tile
    return pmap(new_tiles)
