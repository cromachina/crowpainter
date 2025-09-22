import pytest
import numpy as np
from crowpainter.file_io import rle_native

def run(data, should_overflow=False):
    array = np.array(data, dtype=np.uint8)
    encoded = np.zeros_like(array)
    decoded = np.zeros_like(array)
    size = rle_native.encode(encoded, array)
    if should_overflow:
        assert size == 0
    else:
        assert size > 0
        print(array)
        decoded_size = rle_native.decode(decoded, encoded[:size])
        print(encoded[:size])
        print(decoded)
        assert decoded_size == array.nbytes
        assert (array == decoded).all()

def test():
    run([0, 254, 254, 254, 254, 254, 254, 254, 162, 254])
    run([0, 0, 87, 254, 254, 254, 254, 254, 254, 254, 162, 254])
    run([0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
    run([1,2,3,4,5,5,5,5,5,5,1,2,3,4,5])
    run([5,5,5,5,5,5,1,2,3,4,5,5,5,5,5])
    run(([1] * 200) + [1,2,3,4,5])
    run(([1] * 200))
    run(list(range(1, 201)) + ([5] * 60) + list(range(1, 201)))
    run(list(range(1, 201)), True)
