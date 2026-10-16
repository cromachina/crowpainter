import std/math

import canvas_data

const range_max:int16 = 0x4000
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
    ColorArray* = array[TILE_DATA_SIZE, ColorPixel]
    Color3 = tuple[b:int16, g:int16, r:int16]

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

func mul[T](a, b: T): T =
    return (a * b) shr 14

func comp(C, A: int16): int16 =
    return mul(C, range_max - A)

func comp2(Cd, Cs, Ad, As: int16): int16 =
    return comp(Cd, As) + comp(Cs, Ad)

func lerp(a, b, t: int16): int16 =
    return mul(b - a, t) + a

func normal_alpha(Ad, As: int16): int16 =
    return clip(Ad + As - mul(Ad, As))

template do_comp*(fn, field, dst, src, res:untyped) =
    res.field = fn(dst.field, dst.a, src.field, src.a)

template compwise*(fn, dst, src, res:untyped) =
    do_comp(fn, b, dst, src, res)
    do_comp(fn, g, dst, src, res)
    do_comp(fn, r, dst, src, res)

template array_array_func(name, body: untyped) =
    func name*(dst {.inject.}: var ColorArray, src {.inject.}: ColorArray) =
        for i  {.inject.} in 0 ..< TILE_DATA_SIZE:
            body

template array_scalar_func(name, body: untyped) =
    func name*(dst {.inject.}: var ColorArray, src {.inject.}: ColorPixel) =
        for i  {.inject.} in 0 ..< TILE_DATA_SIZE:
            body

template vectorized_array_array_func(name, base_func: untyped) =
    array_array_func name:
        compwise(basefunc, dst[i], src[i], dst[i])
        dst[i].a = normal_alpha(dst[i].a, src[i].a)

template vectorized_array_scalar_func(name, base_func: untyped) =
    array_scalar_func name:
            compwise(base_func, dst[i], src, dst[i])
            dst[i].a = normal_alpha(dst[i].a, src.a)

# Define a premultiplied blend function.
template blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}, Ad {.inject.}, As {.inject.}: int16): int16 =
        body
    vectorized_array_array_func(func_name, `func_name base_func`)
    vectorized_array_scalar_func(func_name, `func_name base_func`)

template nonsep_vectorized_array_array_func(name, base_func: untyped) =
    array_array_func name:
            dst[i] = base_func(dst[i], src[i])

template nonsep_vectorized_array_scalar_func(name, base_func: untyped) =
    array_scalar_func name:
            dst[i] = base_func(dst[i], src)

# Define a nonseparable premultiplied blend function.
template nonsep_blend_func(func_name, body: untyped) =
    func `func_name base_func`(Cd {.inject.}, Cs {.inject.}: ColorPixel): ColorPixel =
        body
    nonsep_vectorized_array_array_func(func_name, `func_name base_func`)
    nonsep_vectorized_array_scalar_func(func_name, `func_name base_func`)

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
        Cdp_b = clip_divide(Cd.b, Ad)
        Cdp_g = clip_divide(Cd.g, Ad)
        Cdp_r = clip_divide(Cd.r, Ad)
        Csp_b = clip_divide(Cs.b, As)
        Csp_g = clip_divide(Cs.g, As)
        Csp_r = clip_divide(Cs.r, As)
        Asrc = comp(As, Ad)
        Adst = comp(Ad, As)
        Aboth = mul(As, Ad)
        B = fn((b:Cdp_b, g:Cdp_g, r:Cdp_r), (b:Csp_b, g:Csp_g, r:Csp_r))
        B_b = blend(Csp_b, Cdp_b, Asrc, Adst, Aboth, B.b)
        B_g = blend(Csp_g, Cdp_g, Asrc, Adst, Aboth, B.g)
        B_r = blend(Csp_r, Cdp_r, Asrc, Adst, Aboth, B.r)
        B_a = normal_alpha(Ad, As)
    return ColorPixel(b: B_b, g: B_g, r: B_r, a: B_a)

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
    return mul(gamma_b, C.b) + mul(gamma_g, C.g) + mul(gamma_r, C.r)

func min(C: Color3): int16 =
    return min(C.b, min(C.g, C.r))

func max(C: Color3): int16 =
    return max(C.b, max(C.g, C.r))

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
    return clip_color((b: C.b + d, g: C.g + d, r: C.r + d))

func sat(C: Color3): int16 =
    return max(C.b, max(C.g, C.r)) - min(C.b, min(C.g, C.r))

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
