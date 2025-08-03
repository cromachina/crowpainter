import pytest
import numpy as np
import crowpainter.file_io.rle_native as rle

def run(data, should_overflow=False):
    a = np.array(data, dtype=np.uint8)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    size = rle.encode(b, a)
    if should_overflow:
        assert size == 0
    else:
        assert size > 0
        assert rle.decode(c, b) > 0
        assert (a == c).all()

def test():
    run([1,2,3,4,5,5,5,5,5,5,1,2,3,4,5])
    run([5,5,5,5,5,5,1,2,3,4,5,5,5,5,5])
    run(([1] * 200) + [1,2,3,4,5])
    run(([1] * 200))
    run(list(range(1, 201)) + [5,5,5,5,5])
    run(list(range(1, 201)), True)
