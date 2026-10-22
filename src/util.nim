import parazoa

import std/options
import std/strformat
import std/times

template bin_op*(op: untyped, typ, base: typedesc) =
    func op*(x, y: typ): typ {.borrow.}
    func op*(x: base, y: typ): typ {.borrow.}
    func op*(x: typ, y: base): typ {.borrow.}

template bool_op*(op: untyped, typ, base: typedesc) =
    func op*(x, y: typ): bool {.borrow.}
    func op*(x: base, y: typ): bool {.borrow.}
    func op*(x: typ, y: base): bool {.borrow.}

template numerical*(typ, base:typedesc) =
    bin_op(`+`, typ, base)
    bin_op(`-`, typ, base)
    bin_op(`*`, typ, base)
    bin_op(`div`, typ, base)
    bin_op(`mod`, typ, base)
    bool_op(`<`, typ, base)
    bool_op(`<=`, typ, base)
    bool_op(`==`, typ, base)

template def_numerical*(typ, base: untyped) =
    type
        typ* = distinct base
    numerical(typ, base)

func `[]`*[T](v: Vec[T], i: Natural): T =
    v.get(i)

func `[]`*[K, V](m: Map[K, V], k: K): V =
    m.get(k)

iterator reversed*[T](a: Vec[T]): T {.inline.} =
    var i = len(a) - 1
    while i > -1:
        yield a[i]
        dec(i)

iterator chunks*[T](a: openArray[T], chunk_size: int): tuple[first, last: int] {.inline.} =
    for first in countup(0, a.len, chunk_size):
        yield (first, min(first + chunk_size, a.len) - 1)

func get_maybe*[K, V](m: Map[K, V], k: K): Option[V] =
    if m.contains(k): some(m[k]) else: none(V)

func new_ref_seq*[T](size: Natural): ref seq[T] =
    new result
    result[].setlen(size)

func new_ref_seq_uninit*[T](size: Natural): ref seq[T] =
    new result
    result[].setlenuninit(size)

func new_ref*[T](s: T): ref T =
    new result
    result[] = s

func `or`*(a, b: enum): typeof(a) =
  when typeof(a) isnot typeof(b):
    {.warning: "Bitwise `or` is unsafe for different enum types".}
  typeof(a)(ord(a) or ord(b))

proc timer*(time: var float, str: string) =
    let next = epochTime()
    echo fmt"{str} {next - time}"
    time = next

template to_oa*(a: untyped, args: tuple[first: int, last: int]): untyped =
    a.toOpenArray(args.first, args.last)

type
    Vector*[T] = object
        x*, y*: T

    VectorI* = Vector[int]
    VectorF* = Vector[float]

    ColorGrayAlpha*[T] = object
        r*, a*: T

    ColorRgb*[T] = object
        r*, g*, b*: T

    Color*[T] = object
        r*, g*, b*, a*: T

    ColorPixel* = Color[int16]
    MaskPixel* = int16
    ColorRgbPixel* = ColorRgb[int16]
    ColorArray* = openArray[ColorPixel]
    MaskArray* = openArray[MaskPixel]
    ColorBytePixel* = Color[byte]
    ColorByteArray* = openArray[ColorBytePixel]

func `+`*[T](a, b: Vector[T]): Vector[T] =
    Vector[T](x: a.x + b.x, y: a.y + b.y)

func `-`*[T](a, b: Vector[T]): Vector[T] =
    Vector[T](x: a.x - b.x, y: a.y - b.y)

func `*`*[T](a: T, b: Vector[T]): Vector[T] =
    Vector[T](x: a * b.x, y: a * b.y)

func `*`*[T](a: Vector[T], b: T): Vector[T] =
    Vector[T](x: a.x * b, y: a.y * b)

