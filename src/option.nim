type Option*[T] = object
    case
    of None: discard
    of Some: val*: T

func `==`*[T](a: Option[T], b: Option[T]): bool =
    case a
    of Some(a):
        case b
        of Some(b): a == b
        of None: false
    of None:
        case b
        of Some: false
        of None: true

func is_none*[T](a: Option[T]): bool =
    case a
    of Some: false
    of None: true

func is_some*[T](a: Option[T]): bool =
    case a
    of Some: true
    of None: false

func some*[T](val: T): Option[T] =
    Some(val: val)

