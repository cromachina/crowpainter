import std/math

import canvas_data
import util

const range_max*:int16 = 0x3fff
const range_half:int16 = range_max div 2
const range_quarter:int16 = range_max div 4
const range_2:int32 = range_max * 2
const range_4:int32 = range_max * 4
const range_12:int32 = range_max * 12
const range_16:int32 = range_max * 16
const gamma_r:int16 = int16(range_max.float32 * 0.30)
const gamma_g:int16 = int16(range_max.float32 * 0.59)
const gamma_b:int16 = int16(range_max.float32 * 0.11)

func make_sqrt_lut(): array[range_max + 1, int16] =
    for i in 0 ..< len(result):
        result[i.int16] = (sqrt(i.float / range_max.float) * range_max.float).int16

const sqrt_lut = make_sqrt_lut()

type
    ColorArray* = openArray[ColorPixel]
    MaskArray* = openArray[MaskPixel]
    Color3 = tuple[r:int16, g:int16, b:int16]

func to_int16*(data: byte): int16 =
    ((data.int16 shl 6) or (data.int16 shr 2))

func to_byte*(data: int16): byte =
    (data shr 6).byte

func to_int16*(data: openArray[byte]): ref seq[int16] =
    result = new_ref_seq_uninit[int16](data.len)
    for i in 0 ..< data.len:
        result[i] = to_int16(data[i])

func to_byte*(data: openArray[int16]): ref seq[byte] =
    result = new_ref_seq_uninit[byte](data.len)
    for i in 0 ..< data.len:
        result[i] = to_byte(data[i])

func to_ColorPixel*(data: openArray[byte]): ref seq[ColorPixel] =
    result = new_ref_seq_uninit[ColorPixel](data.len div 4)
    var di = 0
    for i in 0 ..< result[].len:
        result[i].r = to_int16(data[di])
        result[i].g = to_int16(data[di + 1])
        result[i].b = to_int16(data[di + 2])
        result[i].a = to_int16(data[di + 3])
        di += 4

func to_byte*(data: ColorArray): ref seq[byte] =
    result = new_ref_seq_uninit[byte](data.len * 4)
    var di = 0
    for i in 0 ..< data.len:
        result[di] = to_byte(data[i].r)
        result[di + 1] = to_byte(data[i].g)
        result[di + 2] = to_byte(data[i].b)
        result[di + 3] = to_byte(data[i].a)
        di += 4

func to_Color3*(C: ColorPixel): Color3 =
    return (r: C.r, g: C.g, b: C.b)

func to_ColorPixel*(x: var ColorPixel, y: Color3) =
    x.r = y.r
    x.g = y.g
    x.b = y.b

func threshold(val: int16): int16 =
    return if val == 0: 0 else: range_max

func clip(val: int32): int16 =
    return int16(max(min(val, range_max.int32), 0))

func safe_divide(a, b: int32): int32 =
    if b == 0:
        return range_max
    else:
        let rm = range_max.float32
        return (rm * (a.float32 / rm) / (b.float32 / rm)).int32

func clip_divide*(a, b: int32): int16 =
    return clip(safe_divide(a, b))

func clip_divide3[T](C: T, A: int32): Color3 =
    result.r = clip_divide(C.r, A)
    result.g = clip_divide(C.g, A)
    result.b = clip_divide(C.b, A)

func mul[T](a, b: T): T =
    return ((a.int32 * b.int32) shr 14).T

func mul*(mask: var MaskArray, opacity: int16) =
    for i in 0 ..< mask.len:
        mask[i] = mul(mask[i], opacity)

func mul3[T](C: T, A: int16): Color3 =
    result.r = mul(C.r, A)
    result.g = mul(C.g, A)
    result.b = mul(C.b, A)

func comp(C, A: int16): int16 =
    return mul(C, range_max - A)

func comp2(Cd, Cs, Ad, As: int16): int16 =
    return comp(Cd, As) + comp(Cs, Ad)

func lerp(a, b, t: int16): int16 =
    return mul(b - a, t) + a

func lerp[T](Cd, Cs: T, val: int16): Color3 =
    result.r = lerp(Cd.r, Cs.r, val)
    result.g = lerp(Cd.g, Cs.g, val)
    result.b = lerp(Cd.b, Cs.b, val)

func blend(Cs, Cd, Asrc, Adst, Aboth, B: int16): int16 =
    return mul(Cs, Asrc) + mul(Cd, Adst) + mul(Aboth, B)

func lerp*(a: var ColorArray, b: ColorArray, t: MaskArray) =
    var
        b_i = 0
        t_i = 0
    let
        a_stride = if a.len == b.len: 1 else: 0
        t_stride = if a.len == t.len: 1 else: 0
    for i {.inject.} in 0 ..< a.len:
        to_ColorPixel(a[i], lerp(a[i], b[i], t[i]))

func normal_alpha(Ad, As: int16): int16 =
    return clip(Ad + As - mul(Ad, As))

func alpha_mask_part(Cd, Cs: ColorPixel, mask: MaskPixel): ColorPixel =
    if mask < range_max:
        to_ColorPixel(result, lerp(Cd, Cs, mask))
        result.a = mul(Cs.a, mask)
    else:
        result = Cs
    result.a = normal_alpha(Cd.a, result.a)

template compwise_premul(Cd, Cs, fn: untyped) =
    result.r = fn(Cd.r, Cs.r, Cd.a, Cs.a)
    result.g = fn(Cd.g, Cs.g, Cd.a, Cs.a)
    result.b = fn(Cd.b, Cs.b, Cd.a, Cs.a)

template compwise_straight(Cd, Cs, fn: untyped) =
    result.r = fn(Cd.r, Cs.r)
    result.g = fn(Cd.g, Cs.g)
    result.b = fn(Cd.b, Cs.b)

template composite_pixel_premul(Cd, Cs, fn: untyped) =
    var Css = Cs
    to_ColorPixel(Css, mul3(Css, Css.a))
    return fn(Cd, Css)

template composite_pixel_straight(Cd, Cs, fn: untyped) =
    let
        Ad = Cd.a
        As = Cs.a
        Cdp = clip_divide3(Cd, Ad)
        Asrc = comp(As, Ad)
        Adst = comp(Ad, As)
        Aboth = mul(As, Ad)
        B = fn(Cdp, to_Color3(Cs))
    result.r = blend(Cs.r, Cdp.r, Asrc, Adst, Aboth, B.r)
    result.g = blend(Cs.g, Cdp.g, Asrc, Adst, Aboth, B.g)
    result.b = blend(Cs.b, Cdp.b, Asrc, Adst, Aboth, B.b)

template composite*(res, dst, src, mask, opacity, composite_pixel_fn: untyped) =
    var
        src_i = 0
        mask_i = 0
    let
        src_stride = if dst.len == src.len: 1 else: 0
        mask_stride = if dst.len == mask.len: 1 else: 0
    for i in 0 ..< res.len:
        var Cs = src[src_i]
        let Cd = dst[i]
        Cs.a = mul(Cs.a, opacity)
        to_ColorPixel(Cs, composite_pixel_fn(Cd, Cs))
        res[i] = alpha_mask_part(Cd, Cs, mask[mask_i])
        src_i += src_stride.int
        mask_i += mask_stride.int

# Define a compwise blend function.
template blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}, Ad {.inject.}, As {.inject.}: int16): int16 =
        body

    func `func_name compwise`(Cd {.inject.}, Cs {.inject.}: ColorPixel): Color3 =
        compwise_premul(Cd, Cs, `func_name base_func`)

    func `func_name composite_pixel`(Cd {.inject.}, Cs {.inject.}: ColorPixel): Color3 =
        composite_pixel_premul(Cd, Cs, `func_name compwise`)

    func func_name*(res {.inject.}: var ColorArray, dst {.inject.}: ColorArray, src {.inject.}: ColorArray,
                    mask {.inject.}: MaskArray, opacity {.inject.}: MaskPixel) =
        composite(res, dst, src, mask, opacity, `func_name composite_pixel`)

# Define a nonseparable blend function.
template nonsep_blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}: Color3): Color3 =
        body

    func `func_name composite_pixel`(Cd {.inject.}, Cs {.inject.}: ColorPixel): Color3 =
        composite_pixel_straight(Cd, Cs, `func_name base_func`)

    func func_name*(res {.inject.}: var ColorArray, dst {.inject.}: ColorArray, src {.inject.}: ColorArray,
                    mask {.inject.}: MaskArray, opacity {.inject.}: MaskPixel) =
        composite(res, dst, src, mask, opacity, `func_name composite_pixel`)

# Define a compwise straight blend function.
template straight_blend_func(func_name, base_func: untyped, flip_dst_src: bool = false) =
    func `func_name compwise`(Cd {.inject.}, Cs {.inject.}: Color3): Color3 =
        compwise_straight(Cd, Cs, base_func)

    when flip_dst_src:
        func `func_name composite_pixel`(Cd {.inject.}, Cs {.inject.}: ColorPixel): Color3 =
            composite_pixel_straight(Cs, Cd, `func_name compwise`)
    when not flip_dst_src:
        func `func_name composite_pixel`(Cd {.inject.}, Cs {.inject.}: ColorPixel): Color3 =
            composite_pixel_straight(Cd, Cs, `func_name compwise`)

    func func_name*(res {.inject.}: var ColorArray, dst {.inject.}: ColorArray, src {.inject.}: ColorArray,
                    mask {.inject.}: MaskArray, opacity {.inject.}: MaskPixel) =
        composite(res, dst, src, mask, opacity, `func_name composite_pixel`)

# Define a straight blend function.
template straight_func(name, body: untyped) =
    func name(Cd {.inject.}, Cs {.inject.}: int16): int16 =
        body

blend_func normal:
    return Cs + comp(Cd, As)

func normal_b(Cd, Cs, Ad, As: int16): int16 =
    return Cs + comp(Cd, As)

func composite_normal*(res: var ColorArray, dst, src: ColorArray, mask: MaskArray, opacity: MaskPixel) =
    var
        src_i = 0
        mask_i = 0
    let
        src_stride = if dst.len == src.len: 1 else: 0
        mask_stride = if dst.len == mask.len: 1 else: 0
    for i in 0 ..< res.len:
        var Cs = src[src_i]
        let Cd = dst[i]
        Cs.a = mul(Cs.a, opacity)

        to_ColorPixel(Cs, mul3(Cs, Cs.a))

        Cs.r = normal_b(Cd.r, Cs.r, Cd.a, Cs.a)
        Cs.g = normal_b(Cd.g, Cs.g, Cd.a, Cs.a)
        Cs.b = normal_b(Cd.b, Cs.b, Cd.a, Cs.a)

        res[i] = alpha_mask_part(Cd, Cs, mask[mask_i])
        src_i += src_stride.int
        mask_i += mask_stride.int

blend_func multiply:
    return mul(Cd, Cs) + comp2(Cd, Cs, Ad, As)

straight_func screen_straight:
    return Cs + Cd - mul(Cd, Cs)

blend_func screen:
    return screen_straight(Cd, Cs)

straight_func hard_light_straight:
    let Cs2 = mul(Cs.int32, range_2).int16
    if Cs > range_half:
        return screen_straight(Cd, Cs2 - range_max)
    else:
        return mul(Cd, Cs2)

straight_blend_func(overlay, hard_light_straight, true)

blend_func linear_burn:
    let
        Cdd = clip_divide(Cd, Ad)
        H = clip(Cdd + Cs - As)
    return lerp(Cs, H, Ad)

straight_func ts_linear_burn_straight:
    return clip(Cd + Cs - range_max)

straight_blend_func(ts_linear_burn, ts_linear_burn_straight)

blend_func linear_dodge:
    let
        Cdd = clip_divide(Cd, Ad)
        H = clip(Cdd + Cs)
    return lerp(Cs, H, Ad)

straight_func ts_linear_dodge_straight:
    return clip(Cd + Cs)

straight_blend_func(ts_linear_dodge, ts_linear_dodge_straight)

blend_func linear_light:
    let
        Cdd = clip_divide(Cd, Ad)
        Cs2 = mul(Cs.int32, range_2).int16
        LB = clip(Cdd + Cs2 - As)
    return lerp(Cs, LB, Ad)

straight_func ts_linear_light_straight:
    let
        Cs2 = mul(Cs.int32, range_2).int16
    if Cs > range_half:
        return ts_linear_dodge_straight(Cd, Cs2 - rangemax)
    else:
        return ts_linear_burn_straight(Cd, Cs2)

straight_blend_func(ts_linear_light, ts_linear_light_straight)

straight_func ts_color_burn_straight:
    return rang_emax - clip_divide(range_max - Cd, Cs)

straight_blend_func(ts_color_burn, ts_color_burn_straight)

blend_func color_burn:
    let
        Cdd = clip_divide(Cd, Ad)
        B = range_max - clip_divide(range_max - Cdd, range_max - As + Cs)
    return lerp(Cs, B, Ad)

straight_func ts_color_dodge_straight:
    return clip_divide(Cd, range_max - Cs)

straight_blend_func(ts_color_dodge, ts_color_dodge_straight)

blend_func color_dodge:
    let
        Cdd = clip_divide(Cd, Ad)
        H = clip_divide(Cdd, range_max - Cs)
    return lerp(Cs, H, Ad)

straight_func ts_vivid_light_straight:
    let Cs2 = mul(Cs.int32, range_2).int16
    if Cs > range_half:
        return ts_color_dodge_straight(Cd, Cs2 - range_max)
    else:
        return ts_color_burn_straight(Cd, Cs2)

straight_blend_func(ts_vivid_light, ts_vivid_light_straight)

blend_func vivid_light:
    let
        Cdd = clip_divide(Cd, Ad)
        Csd = clip_divide(Cs, As)
        Cs2 = As - mul(Cs.int32, range_2).int16
    var CB = range_max - clip_divide(range_max - Cdd, range_max - Cs2)
    if Csd > range_half:
        CB = clip_divide(Cdd, range_max + Cs2)
    if Cs == range_max:
        CB = range_max
    return lerp(Cs, CB, Ad)

straight_func soft_light_straight:
    var D, B: int16
    if Cs <= range_quarter:
        D = mul((mul((mul(range_16, Cd.int32) - range_12), Cd) + range_4), Cd.int32).int16
    else:
        D = sqrt_lut[Cd]
    if Cs <= rangehalf:
        B = Cd - mul(mul((range_max - mul(range_2, Cs.int32)), Cd.int32), (range_max - Cd).int32).int16
    else:
        B = Cd + mul((mul(range_2, Cs.int32) - range_max), (D - Cd).int32).int16
    return B

straight_blend_func(soft_light, soft_light_straight)

straight_blend_func(hard_light, hard_light_straight)

straight_func pin_light_straight:
    let Cs2 = mul(Cs.int32, range_2).int16
    if Cs > range_half:
        return max(Cs2 - range_max, Cd)
    else:
        return min(Cs2, Cd)

straight_blend_func(pin_light, pin_light_straight)

blend_func hard_mix:
    let
        Cdd = clip_divide(Cd, Ad)
        Csd = clip_divide(Cs, As)
        H = clip_divide(Cdd - As + mul(As, Csd), range_max - As)
    return lerp(Cs, H, Ad)

straight_func ts_hard_mix_straight:
    if (Cd + Cs) > range_max:
        return range_max
    else:
        return 0

straight_blend_func(ts_hard_mix, ts_hard_mix_straight)

blend_func darken:
    return min(mul(Cs, Ad), mul(Cd, As)) + comp2(Cd, Cs, Ad, As)

blend_func lighten:
    return max(mul(Cs, Ad), mul(Cd, As)) + comp2(Cd, Cs, Ad, As)

blend_func ts_difference:
    return Cs + Cd - mul(range_2, min(mul(Cs, Ad), mul(Cd, As))).int16

blend_func difference:
    let
        Cdd = clip_divide(Cd, Ad)
        D = abs(Cdd - Cs)
    return lerp(Cs, D, Ad)

blend_func exclude:
    return (mul(Cs, Ad) + mul(Cd, As) - mul(mul(range_2, Cs).int16, Cd)) + comp2(Cd, Cs, Ad, As)

blend_func subtract:
    return max(0, mul(Cd, As) - mul(Cs, Ad)) + comp2(Cd, Cs, Ad, As)

straight_func clip_divide_short:
    return clip(safe_divide(Cd, Cs))

straight_blend_func(divide, clip_divide_short)

func lum(C: Color3): int16 =
    return mul(gamma_r, C.r) + mul(gamma_g, C.g) + mul(gamma_b, C.b)

func min(C: Color3): int16 =
    return min(min(C.r, C.g), C.b)

func max(C: Color3): int16 =
    return max(max(C.r, C.g), C.b)

func clip_color(C: Color3): Color3 =
    let
        L = lum(C)
        n = min(C)
        x = max(C)
        C_rL = C.r - L
        C_gL = C.g - L
        C_bL = C.b - L
    var Co = C
    if n < 0:
        let Ln = L - n
        Co.r = L + safe_divide(mul(C_rL, L), Ln).int16
        Co.g = L + safe_divide(mul(C_gL, L), Ln).int16
        Co.b = L + safe_divide(mul(C_bL, L), Ln).int16
    if x > range_max:
        let
            L1 = range_max - L
            xL = x - L
        Co.r = L + safe_divide(mul(C_rL, L1), xL).int16
        Co.g = L + safe_divide(mul(C_gL, L1), xL).int16
        Co.b = L + safe_divide(mul(C_bL, L1), xL).int16
    return C

func set_lum(C: Color3, L: int16): Color3 =
    let d = L - lum(C)
    return clip_color((r: C.r + d, g: C.g + d, b: C.b + d))

func sat(C: Color3): int16 =
    return max(C.r, max(C.g, C.b)) - min(C.r, min(C.g, C.b))

func set_sat(C: Color3, S: int16): Color3 =
    var
        amax = max(C)
        amin = min(C)
        amid: int16
        C = C
        cmax: ptr int16
        cmid: ptr int16
        cmin: ptr int16
    if amax == C.b:
        cmax = addr C.b
        if amin == C.g:
            cmin = addr C.g
            cmid = addr C.r
        else:
            cmin = addr C.r
            cmid = addr C.g
    elif amax == C.g:
        cmax = addr C.g
        if amin == C.b:
            cmin = addr C.b
            cmid = addr C.r
        else:
            cmin = addr C.r
            cmid = addr C.b
    else:
        cmax = addr C.r
        if amin == C.b:
            cmin = addr C.b
            cmid = addr C.g
        else:
            cmin = addr C.g
            cmid = addr C.b
    amid = cmid[]
    if amax > amin:
        cmid[] = safe_divide(mul((amid - amin), S), amax - amin).int16
        cmax[] = S
    else:
        cmid[] = 0
        cmax[] = 0
    cmin[] = 0
    return C

nonsep_blend_func hue:
    return set_lum(set_sat(Cs, sat(Cd)), lum(Cd))

nonsep_blend_func saturation:
    return set_lum(set_sat(Cd, sat(Cs)), lum(Cd))

nonsep_blend_func color:
    return set_lum(Cs, lum(Cd))

nonsep_blend_func luminosity:
    return set_lum(Cd, lum(Cs))

nonsep_blend_func darken_color:
    if lum(Cs) < lum(Cd):
        return Cs
    else:
        return Cd

nonsep_blend_func lighten_color:
    if lum(Cs) > lum(Cd):
        return Cs
    else:
        return Cd

func noop[A, B](a: A, b: B) = discard

func run_blendfunc*[T](blend_mode: BlendMode, dst: ref ColorArray, src: T) =
    let fn = case blend_mode:
        of BlendMode.Pass: noop
        of BlendMode.Normal: normal
        of BlendMode.Multiply: multiply
        of BlendMode.Screen: screen
        of BlendMode.Overlay: overlay
        of BlendMode.LinearBurn: linear_burn
        of BlendMode.LinearDodge: linear_dodge
        of BlendMode.LinearLight: linear_light
        of BlendMode.ColorBurn: color_burn
        of BlendMode.ColorDodge: color_dodge
        of BlendMode.VividLight: vivid_light
        of BlendMode.SoftLight: soft_light
        of BlendMode.HardLight: hard_light
        of BlendMode.PinLight: pin_light
        of BlendMode.HardMix: hard_mix
        of BlendMode.Darken: darken
        of BlendMode.Lighten: lighten
        of BlendMode.DarkenColor: darken_color
        of BlendMode.LightenColor: lighten_color
        of BlendMode.Difference: difference
        of BlendMode.Exclude: exclude
        of BlendMode.Subtract: subtract
        of BlendMode.Divide: divide
        of BlendMode.Hue: hue
        of BlendMode.Saturation: saturation
        of BlendMode.Color: color
        of BlendMode.Luminosity: luminosity
        of BlendMode.TsLinearBurn: ts_linear_burn
        of BlendMode.TsLinearDodge: ts_linear_dodge
        of BlendMode.TsLinearLight: ts_linear_light
        of BlendMode.TsColorBurn: ts_color_burn
        of BlendMode.TsColorDodge: ts_color_dodge
        of BlendMode.TsVividLight: ts_vivid_light
        of BlendMode.TsHardMix: ts_hard_mix
        of BlendMode.TsDifference: ts_difference
    fn(dst, src)
