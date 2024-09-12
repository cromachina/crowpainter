from enum import Enum, auto
import numpy as np

TILE_SIZE = (256, 256)
DTYPE = np.float64

class BlendMode(Enum):
    PASS = auto
    NORMAL = auto
    MULTIPLY = auto
    SCREEN = auto
    OVERLAY = auto
    LINEAR_BURN = auto
    LINEAR_DODGE = auto
    LINEAR_LIGHT = auto
    COLOR_BURN = auto
    COLOR_DODGE = auto
    VIVID_LIGHT = auto
    SOFT_LIGHT = auto
    HARD_LIGHT = auto
    PIN_LIGHT = auto
    HARD_MIX = auto
    DARKEN = auto
    LIGHTEN = auto
    DARKEN_COLOR = auto
    LIGHTEN_COLOR = auto
    DIFFERENCE = auto
    EXCLUDE = auto
    SUBTRACT = auto
    DIVIDE = auto
    HUE = auto
    SATURATION = auto
    COLOR = auto
    LUMINOSITY = auto
    TS_LINEAR_BURN = auto
    TS_LINEAR_DODGE = auto
    TS_LINEAR_LIGHT = auto
    TS_COLOR_BURN = auto
    TS_COLOR_DODGE = auto
    TS_VIVID_LIGHT = auto
    TS_HARD_MIX = auto
    TS_DIFFERENCE = auto
