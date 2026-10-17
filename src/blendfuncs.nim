import std/math

import canvas_data
import util

const range_max*:int16 = 0x4000
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

template a_a_func(name, body: untyped) =
    func name*[T](dst {.inject.}: var T, src {.inject.}: T) =
        for i {.inject.} in 0 ..< TILE_DATA_SIZE:
            body

template a_s_func(name, body: untyped) =
    func name*[T](dst {.inject.}: var T, src {.inject.}: int16) =
        for i {.inject.} in 0 ..< TILE_DATA_SIZE:
            body

template strided_call_2(fn, dst, src, src_stride: untyped) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        dst[i] = fn(dst[i], src[])
        src += src_stride

template strided_call_3(fn, dst, src_a, src_a_stride, src_b, src_b_stride: untyped) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        dst[i] = fn(dst[i], src_a[], src_b[])
        src_a += src_a_stride
        src_b += src_b_stride

type
    ColorArray* = array[TILE_DATA_SIZE, ColorPixel]
    MaskArray* = array[TILE_DATA_SIZE, MaskPixel]
    Color3 = tuple[r:int16, g:int16, b:int16]

func threshold(mask: var MaskArray, val: int16): int16 =
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

func mul[T](a, b: T): T =
    return (a * b) shr 14

func mul*(mask: var MaskArray, opacity: int16) =
    for i in 0 ..< TILE_DATA_SIZE:
        mask[i] = mul(mask[i], opacity)

func comp(C, A: int16): int16 =
    return mul(C, range_max - A)

func comp2(Cd, Cs, Ad, As: int16): int16 =
    return comp(Cd, As) + comp(Cs, Ad)

func lerp(a, b, t: int16): int16 =
    return mul(b - a, t) + a

template do_comp_3(fn, field, res, a, b, c: untyped) =
    res.field = fn(a.field, b.field, c)

template compwise_3(fn: untyped, res, da, db, dc: untyped) =
    do_comp_3(fn, r, res, da, db, dc)
    do_comp_3(fn, g, res, da, db, dc)
    do_comp_3(fn, b, res, da, db, dc)

func lerp*(a: var ColorArray, b: ColorArray, t: MaskArray) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        compwise_3(lerp, a[i], a[i], b[i], t[i])

func lerp*(a: var ColorArray, b: ColorArray, t: MaskPixel) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        compwise_3(lerp, a[i], a[i], b[i], t)

func lerp*(a: var ColorArray, b: ColorPixel, t: MaskArray) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        compwise_3(lerp, a[i], a[i], b, t[i])

func lerp*(a: var ColorArray, b: ColorPixel, t: MaskPixel) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        compwise_3(lerp, a[i], a[i], b, t)

func normal_alpha(Ad, As: int16): int16 =
    return clip(Ad + As - mul(Ad, As))

func normal_alpha*(Ad: var MaskArray, As: MaskArray) =
    for i {.inject.} in 0 ..< TILE_DATA_SIZE:
        Ad[i] = normal_alpha(Ad[i], As[i])

template do_comp(fn, field, res, dst, src: untyped) =
    res.field = fn(dst.field, dst.a, src.field, src.a)

template compwise(fn, res, dst, src: untyped) =
    do_comp(fn, r, res, dst, src)
    do_comp(fn, g, res, dst, src)
    do_comp(fn, b, res, dst, src)

template vec_a_a_func(name, base_func: untyped) =
    a_a_func name:
        compwise(base_func, dst[i], src[i], dst[i])

template vec_a_s_func(name, base_func: untyped) =
    a_s_func name:
        compwise(base_func, dst[i], src, dst[i])

# Define a premultiplied blend function.
template blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}, Ad {.inject.}, As {.inject.}: int16): int16 =
        body
    vec_a_a_func(func_name, `func_name base_func`)
    vec_a_s_func(func_name, `func_name base_func`)

template nonsep_vec_a_a_func(name, base_func: untyped) =
    a_a_func name:
        dst[i] = base_func(dst[i], src[i])

template nonsep_vec_a_s_func(name, base_func: untyped) =
    a_s_func name:
        dst[i] = base_func(dst[i], src)

# Define a nonseparable premultiplied blend function.
template nonsep_blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}: ColorPixel): ColorPixel =
        body
    nonsep_vec_a_a_func(func_name, `func_name base_func`)
    nonsep_vec_a_s_func(func_name, `func_name base_func`)

func blend(Csp, Cdp, Asrc, Adst, Aboth, B: int16): int16 =
    return mul(Csp, Asrc) + mul(Cdp, Adst) + mul(Aboth, B)

func premul(Cd, Cs, Ad, As: int16, fn: proc (Cd, Cs: int16): int16 {.noSideEffect.}): int16 =
    let
        Cdp = clip_divide(Cd, Ad)
        Csp = clip_divide(Cs, As)
        B = fn(Cdp, Csp)
        Asrc = comp(As, Ad)
        Adst = comp(Ad, As)
        Aboth = mul(As, Ad)
    return blend(Csp, Cdp, Asrc, Adst, Aboth, B)

# Define a straight blend function.
template straight_func(name, body: untyped) =
    func name(Cd {.inject.}, Cs {.inject.}: int16): int16 =
        body

# Define a premultiplied blend function with a straight function.
template straight_blend_func(name, base_func: untyped) =
    blend_func name:
        return premul(Cd, Cs, Ad, As, base_func)

func premul_nonseperable(Cd, Cs: ColorPixel, fn: proc (Cd, Cs: Color3): Color3 {.noSideEffect.}): ColorPixel =
    let
        Ad = Cd.a
        As = Cs.a
        Cdp_r = clip_divide(Cd.r, Ad)
        Cdp_g = clip_divide(Cd.g, Ad)
        Cdp_b = clip_divide(Cd.b, Ad)
        Csp_r = clip_divide(Cs.r, As)
        Csp_g = clip_divide(Cs.g, As)
        Csp_b = clip_divide(Cs.b, As)
        Asrc = comp(As, Ad)
        Adst = comp(Ad, As)
        Aboth = mul(As, Ad)
        B = fn((r:Cdp_r, g:Cdp_g, b:Cdp_b), (r:Csp_r, g:Csp_g, b:Csp_b))
        B_r = blend(Csp_r, Cdp_r, Asrc, Adst, Aboth, B.r)
        B_g = blend(Csp_g, Cdp_g, Asrc, Adst, Aboth, B.g)
        B_b = blend(Csp_b, Cdp_b, Asrc, Adst, Aboth, B.b)
    return ColorPixel(r: B_r, g: B_g, b: B_b, a: As)

# Define a nonseparable straight function.
template nonsep_straight_func(name, body: untyped) =
    func name(Cd {.inject.}, Cs {.inject.}: Color3): Color3 =
        body

# Define a premultiplied nonseparable blend function with a nonseparable straight function.
template straight_nonsep_blend_func(name, base_func: untyped) =
    nonsep_blend_func name:
        return premul_nonseperable(Cd, Cs, base_func)

blend_func normal:
    return Cs + comp(Cd, As)

func composite_pixel(Cd: ColorPixel, Cs: ColorPixel, mask: MaskPixel, opacity: MaskPixel, fn: proc (Cd, Cs, Ad, As: int16): int16 {.noSideEffect.}): ColorPixel =
    var
        Cs = Cs
        Cd = Cd
    Cs.a = mul(Cs.a, opacity)
    Cs.r = mul(Cs.r, Cs.a)
    Cs.g = mul(Cs.g, Cs.a)
    Cs.b = mul(Cs.b, Cs.a)
    Cs.r = fn(Cd.r, Cs.r, Cd.a, Cs.a)
    Cs.g = fn(Cd.g, Cs.g, Cd.a, Cs.a)
    Cs.b = fn(Cd.b, Cs.b, Cd.a, Cs.a)
    if mask < range_max:
        result.r = lerp(Cd.r, Cs.r, mask)
        result.g = lerp(Cd.g, Cs.g, mask)
        result.b = lerp(Cd.b, Cs.b, mask)
        Cs.a = mul(Cs.a, mask)
    else:
        result.r = Cs.r
        result.g = Cs.g
        result.b = Cs.b
    result.a = normal_alpha(Cd.a, Cs.a)

func composite(dst: var ColorArray, Cd: ColorArray, Cs: ColorArray, Cs_stride: uint8, mask: MaskArray, mask_stride: uint8, opacity: MaskPixel, fn: proc (Cd, Cs, Ad, As: int16): int16 {.noSideEffect.}) =
    var
        Cs_i = 0
        mask_i = 0
    for i in 0 ..< TILE_DATA_SIZE:
        dst[i] = composite_pixel(Cd[i], Cs[Cs_i], mask[mask_i], opacity, fn)
        Cs_i += Cs_stride.int
        mask_i += mask_stride.int

template strided_blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}, Ad {.inject.}, As {.inject.}: int16): int16 =
        body
    func func_name*(dst {.inject.}: var ColorArray, Cd {.inject.}: ColorArray, Cs {.inject.}: ColorArray, Cs_stride {.inject.}: uint8,
                    mask {.inject.}: MaskArray, mask_stride {.inject.}: uint8, opacity {.inject.}: MaskPixel) =
        composite(dst, Cd, Cs, Cs_stride, mask, mask_stride, opacity, `func_name base_func`)

strided_blend_func normal_2:
    return Cs + comp(Cd, As)

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

# Flipped src and dst
blend_func overlay:
    return premul(Cs, Cd, As, Ad, hard_light_straight)

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

nonsep_straight_func hue_straight:
    return set_lum(set_sat(Cs, sat(Cd)), lum(Cd))

straight_nonsep_blend_func(hue, hue_straight)

nonsep_straight_func saturation_straight:
    return set_lum(set_sat(Cd, sat(Cs)), lum(Cd))

straight_nonsep_blend_func(saturation, saturation_straight)

nonsep_straight_func color_straight:
    return set_lum(Cs, lum(Cd))

straight_nonsep_blend_func(color, color_straight)

nonsep_straight_func luminosity_straight:
    return set_lum(Cd, lum(Cs))

straight_nonsep_blend_func(luminosity, luminosity_straight)

nonsep_straight_func darken_color_straight:
    if lum(Cs) < lum(Cd):
        return Cs
    else:
        return Cd

straight_nonsep_blend_func(darken_color, darken_color_straight)

nonsep_straight_func lighten_color_straight:
    if lum(Cs) > lum(Cd):
        return Cs
    else:
        return Cd

straight_nonsep_blend_func(lighten_color, lighten_color_straight)

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
