#cython: language_level=3, boundscheck=False, wraparound=False
import sys
from libc.string cimport memcpy, memset
from libc.stdint cimport *
import numpy as np

# Return the size of the decompressed region, or 0 if reading or writing would exceed the src or dst lengths.
cdef inline size_t decode_rle(uint8_t[:] dst, const uint8_t[:] src) noexcept nogil:
    cdef size_t dst_length = dst.shape[0]
    cdef size_t src_length = src.shape[0]
    cdef size_t src_i, dst_i
    cdef uint8_t length
    src_i = 0
    dst_i = 0
    while dst_i < dst_length and src_i < src_length:
        length = src[src_i]
        src_i += 1
        # RAW
        if length < 128:
            length += 1
            if dst_i + length > dst_length or src_i + length > src_length:
                return 0
            dst[dst_i:dst_i + length] = src[src_i:src_i + length]
            dst_i += length
            src_i += length
        # RLE
        elif length > 128:
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
cdef size_t encode_rle(uint8_t[:] dst, const uint8_t[:] src) noexcept nogil:
    cdef size_t dst_length = dst.shape[0]
    cdef size_t src_length = src.shape[0]
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

cdef void decode_rle_counts(uint8_t[:,:] dst, const uint32_t[:] src_row_sizes, const uint8_t[:] src) noexcept nogil:
    cdef uint32_t src_row_offset = 0, src_row_length
    cdef const uint8_t[:] src_row
    cdef uint8_t[:] dst_row
    cdef int i

    for i in range(src_row_sizes.shape[0]):
        src_row_length = src_row_sizes[i]
        src_row = src[src_row_offset:src_row_offset + src_row_length]
        src_row_offset += src_row_length
        dst_row = dst[i]
        decode_rle(dst_row, src_row)

cdef void encode_rle_counts(uint8_t[:] dst, uint32_t[:] dst_row_sizes, const uint8_t[:,:]src) noexcept nogil:
    pass

def decode(dst:np.ndarray, src:np.ndarray):
    return decode_rle(dst, src)

def encode(dst:np.ndarray, src:np.ndarray):
    return encode_rle(dst, src)

def decode_counts(data, width, height, depth, version):
    row_size = max(width * depth // 8, 1)
    dtype = (np.uint16, np.uint32)[version - 1]
    counts = np.frombuffer(data, dtype=dtype, count=height).copy()
    if sys.byteorder == 'little':
        counts.byteswap(inplace=True)
    rows = np.frombuffer(data, dtype=np.uint8, offset=counts.nbytes)
    result = np.empty((height, row_size), dtype=np.uint8)
    decode_rle_counts(result, counts.astype(np.uint32), rows)
    return result

def encode_counts():
    pass
