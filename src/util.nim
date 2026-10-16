import std/options
import parazoa

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

func get_maybe*[K, V](m: Map[K, V], k: K): Option[V] =
    if m.contains(k): some(m[k]) else: none(V)

func new_seq_ref*[T](size: Natural): ref seq[T] =
    new result
    result[].setlen(size)

func new_ref*[T](s: T): ref T =
    new result
    result[] = s
