#cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport *
import numpy as np

# Return the size of the decompressed region, or 0 if reading or writing would exceed the src or dst lengths.
cdef size_t decode_rle(uint8_t[:] dst, const uint8_t[:] src, size_t dst_length, size_t src_length) noexcept nogil:
    cdef size_t src_i, dst_i
    cdef uint8_t length
    src_i = 0
    dst_i = 0
    while dst_i < dst_length and src_i < src_length:
        length = src[src_i]
        src_i += 1
        # No-op
        if length == 128:
            pass
        # RAW
        elif length < 128:
            length += 1
            if dst_i + length > dst_length or src_i + length > src_length:
                return 0
            dst[dst_i:dst_i + length] = src[src_i:src_i + length]
            dst_i += length
            src_i += length
        # RLE
        else:
            length = (length ^ 0xff) + 2
            if dst_i + length > dst_length or src_i > src_length:
                return 0
            dst[dst_i:dst_i + length] = src[src_i]
            dst_i += length
            src_i += 1
    return dst_i

cdef inline bint finish_raw(uint8_t[:] dst, const uint8_t[:] src, size_t* pdst_i, size_t src_i, int count, size_t dst_length) noexcept nogil:
    cdef size_t dst_i = pdst_i[0]
    if count <= 0:
        return True
    if dst_i + count + 2 >= dst_length:
        return False
    dst[dst_i] = count - 1
    dst_i += 1
    dst[dst_i:dst_i + count] = src[src_i:src_i + count]
    dst_i += count
    pdst_i[0] = dst_i
    return True

cdef inline bint finish_rle(uint8_t[:] dst, const uint8_t[:] src, size_t* pdst_i, size_t src_i, int count, size_t dst_length) noexcept nogil:
    cdef size_t dst_i = pdst_i[0]
    cdef uint8_t rle_length
    if count <= 0:
        return True
    if dst_i + 1 >= dst_length:
        return False
    rle_length = (count ^ 0xff) + 2
    dst[dst_i] = rle_length
    dst[dst_i + 1] = src[src_i]
    dst_i += 2
    pdst_i[0] = dst_i
    return True

# Returns the size of the compressed region, or 0 to indicate the pathological case where
# there are too many RAWs and the region would grow instead.
cdef size_t encode_rle(uint8_t[:] dst, const uint8_t[:] src, size_t dst_length, size_t src_length) noexcept nogil:
    cdef size_t src_i = 0
    cdef size_t dst_i = 0
    cdef size_t src_end = src_length - 1
    cdef int count = 0
    cdef uint8_t MAX_LENGTH = 127
    cdef bint raw = True
    while src_i < src_end:
        if src[src_i] == src[src_i + 1]: # Becomes RLE
            if raw:
                if not finish_raw(dst, src, &dst_i, src_i - count, count, dst_length):
                    return 0
                raw = False
                count = 1
            else:
                if count == MAX_LENGTH:
                    if not finish_rle(dst, src, &dst_i, src_i, count, dst_length):
                        return 0
                    count = 0
                count += 1
        else: # Becomes RAW
            if raw:
                if count == MAX_LENGTH:
                    if not finish_raw(dst, src, &dst_i, src_i - count, count, dst_length):
                        return 0
                    count = 0
                count += 1
            else:
                if not finish_rle(dst, src, &dst_i, src_i, count + 1, dst_length):
                    return 0
                raw = True
                count = 0
        src_i += 1
    if raw:
        if not finish_raw(dst, src, &dst_i, src_i - count, count + 1, dst_length):
            return 0
    else:
        if not finish_rle(dst, src, &dst_i, src_i, count + 1, dst_length):
            return 0
    return dst_i

def decode(dst:np.ndarray, src:np.ndarray):
    return decode_rle(dst, src, dst.size, src.size)

def encode(dst:np.ndarray, src:np.ndarray):
    return encode_rle(dst, src, dst.size, src.size)
