#cython: language_level=3, boundscheck=False, wraparound=False
import sys
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
cdef inline size_t encode_rle(uint8_t[:] dst, const uint8_t[:] src) noexcept nogil:
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
    cdef Py_ssize_t i
    for i in range(src_row_sizes.shape[0]):
        src_row_length = src_row_sizes[i]
        src_row = src[src_row_offset:src_row_offset + src_row_length]
        src_row_offset += src_row_length
        dst_row = dst[i]
        decode_rle(dst_row, src_row)

# Returns final resulting length of dst
cdef uint32_t encode_rle_counts(uint32_t[:] dst_row_sizes, uint8_t[:] dst, const uint8_t[:,:] src) noexcept nogil:
    cdef uint32_t dst_row_offset = 0, dst_row_size
    cdef const uint8_t[:] src_row
    cdef uint8_t[:] dst_row
    cdef Py_ssize_t i
    for i in range(src.shape[0]):
        dst_row = dst[dst_row_offset:]
        src_row = src[i]
        dst_row_size = encode_rle(dst_row, src_row)
        dst_row_sizes[i] = dst_row_size
        dst_row_offset += dst_row_size
    return dst_row_offset

def decode(dst:np.ndarray, src:np.ndarray):
    return decode_rle(dst, src)

def encode(dst:np.ndarray, src:np.ndarray):
    return encode_rle(dst, src)

def decode_psd(src, width, height, depth, version):
    row_size = max(width * depth // 8, 1)
    dtype = (np.uint16, np.uint32)[version - 1]
    src_row_sizes = np.frombuffer(src, dtype=dtype, count=height).copy()
    if sys.byteorder == 'little':
        src_row_sizes.byteswap(inplace=True)
    src_rows = np.frombuffer(src, dtype=np.uint8, offset=src_row_sizes.nbytes)
    dst = np.empty((height, row_size), dtype=np.uint8)
    decode_rle_counts(dst, src_row_sizes.astype(np.uint32), src_rows)
    return dst

def encode_psd(src:np.ndarray, version):
    worst_case_width = int(np.ceil(src.shape[1] / 128)) + src.shape[1]
    dst_row_sizes = np.empty(shape=(src.shape[0],), dtype=np.uint32)
    dst = np.empty(shape=(src.shape[0] * worst_case_width,), dtype=np.uint8)
    final_size = encode_rle_counts(dst_row_sizes, dst, src)
    dst = dst[:final_size]
    dtype = (np.uint16, np.uint32)[version - 1]
    dst_row_sizes = dst_row_sizes.astype(dtype)
    if sys.byteorder == 'little':
        dst_row_sizes.byteswap(inplace=True)
    return dst_row_sizes, dst
