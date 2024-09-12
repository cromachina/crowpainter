from pathlib import Path

from ..layer_data import *

# TODO Use protocol buffers for a binary format.

def read(file_path:Path) -> Canvas:
    raise NotImplementedError()

def write(canvas:Canvas, file_name:Path):
    raise NotImplementedError()
