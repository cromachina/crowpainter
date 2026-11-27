import imguin/sdl3_opengl
import opengl
import parazoa
import sdl3_nim as sdl
import bbtree

import std/locks
import std/logging
import std/math
import std/os
import std/paths
import std/sequtils
import std/strformat
import std/strutils
import std/sugar
import std/times
import std/typedthreads
import std/tables
import std/deques
import std/sets

import blendfuncs
import canvas
# import composite
import image_io
import threadpool
import util
import option

const scroll_zoom_levels = collect(newSeq):
    for x in countup(-28, 20):
        2.0 ^ (x.float / 4)
const default_zoom_level:int = 28

const font_file = staticRead("../lib/fonts/NotoSans/NotoSans.ttf")

def_numerical(ViewportId, int)
def_numerical(ProjectId, int)

type
    Tool = ref object of RootObj
        name: string

    DynamicsProperty = object
        enabled: bool
        ## Ordered control points by X to Y
        curve: BBTree[float, float]

    DynamicsProperties = object
        value: int
        pressure: DynamicsProperty
        tilt: DynamicsProperty
        speed: DynamicsProperty
        direction: DynamicsProperty
        stroke_randomness: DynamicsProperty
        stamp_randomness: DynamicsProperty

    Texture = ref object
        size: VectorI
        data: ref seq[MaskPixel]

    BrushProperties = object
        size: DynamicsProperties
        opacity: DynamicsProperties
        flow: DynamicsProperties
        roundness: DynamicsProperties
        softness: DynamicsProperties
        spacing: DynamicsProperties
        scatter: DynamicsProperties
        angle: DynamicsProperties
        antialias: bool
        max_alpha: bool
        shape: Texture
        texture: Texture

    Brush = ref object of Tool
        current: BrushProperties
        default: BrushProperties

    ## Composited canvas image data and associated UI handles.
    CompositeData = ref object
        composite: ref seq[ColorPixelArray]
        texture: GLuint
        pbo: GLuint

proc `=destroy`(composite: type(CompositeData()[])) =
    glDeleteTextures(1, composite.texture.addr)
    glDeleteBuffers(1, composite.pbo.addr)

type
    # Because the canvas state is immutable some of it has to be duplicated in the UI.
    LayerItemData = ref object
        visible: bool
        name: string
        blend_mode: BlendMode
        opacity: float
        alpha_lock: bool
        draw_lock: bool
        move_lock: bool
        all_lock: bool
        clipping: bool
        item_rect_max: ImVec2

    LayerListData = object
        layers = newTable[LayerId, LayerItemData]()
        selection: ImGuiSelectionBasicStorage

    ## Maintains the canvas states and UI views.
    Project = ref object
        id: ProjectId
        file_path: Path
        saved_to_disk: bool = false
        states: Deque[Canvas]
        current_index: int = 0
        next_id = 0.CanvasChangeId
        saved_id: Option[CanvasChangeId]
        composites: BBtree[TimelinePos, CompositeData]
        viewports: HashSet[ViewportId]
        timeline_pos = 0.TimelinePos
        layer_list: LayerListData

    ## Describes the display of a canvas.
    Viewport = ref object
        id: ViewportId
        project_id: ProjectId
        position: ImVec2
        zoom_level: int = default_zoom_level
        rotation: float = 0
        last_brush_pos: ImVec2
        panning: bool = false
        is_open: bool = true

    NewCanvasDialog = object
        show: bool
        name: string
        size: VectorI
        background_color: array[3, cfloat]
        background_transparent: bool

    App = ref object
        running = true
        next_project_id = 0.ProjectId
        next_viewport_id = 0.ViewportId
        projects = newOrderedTable[ProjectId, Project]()
        viewports = newOrderedTable[ViewportId, Viewport]()
        recent_projects: Deque[Path]
        active_project: Project
        active_tool: Tool
        brush_fg_color: ColorRgbBytePixel
        brush_bg_color: ColorRgbBytePixel
        brush_transparency: bool
        new_canvas_dialog: NewCanvasDialog
        file_dialog: ptr ImGuiFileDialog
        last_dir = "."
        checker_texture: GLuint
        checker_sampler: GLuint
        canvas_sampler: GLuint

func `$`(v: ImVec2): string =
    &"{{{v.x}, {v.y}}}"

func vec2(x, y: float): ImVec2 =
    ImVec2(x:x, y:y)

func vec2(v: VectorI): ImVec2 =
    vec2(v.x.float, v.y.float)

func vec4(x, y, z, w: float): ImVec4 =
    ImVec4(x:x, y:y, z:z, w:w)

proc to_color(vec: ImVec4): ImU32 =
    igColorConvertFloat4ToU32(vec)

proc to_color(r, g, b, a: float): ImU32 =
    igColorConvertFloat4ToU32(vec4(r, g, b, a))

func `+`(a, b: ImVec2): ImVec2 =
    vec2(a.x + b.x, a.y + b.y)

func `*`(a: ImVec2, b: float): ImVec2 =
    vec2(a.x * b, a.y * b)

func `*`(a: ImVec2, b: ImVec2): ImVec2 =
    vec2(a.x * b.x, a.y * b.y)

proc add_hsv(color: ImVec4, h, s, v: cfloat): ImVec4 =
    var hsv: ImVec4
    igColorConvertRGBtoHSV(color.x, color.y, color.z, hsv.x.addr, hsv.y.addr, hsv.z.addr)
    hsv.x = (hsv.x + h) mod 1.0
    hsv.y = clamp(hsv.y + s, 0.cfloat .. 1.cfloat)
    hsv.z = clamp(hsv.z + v, 0.cfloat .. 1.cfloat)
    igColorConvertHSVtoRGB(hsv.x, hsv.y, hsv.z, result.x.addr, result.y.addr, result.z.addr)
    result.w = color.w

proc add_rgb(color: ImVec4, r, g, b: cfloat): ImVec4 =
    result.x = clamp(color.x + r, 0.cfloat .. 1.cfloat)
    result.y = clamp(color.y + g, 0.cfloat .. 1.cfloat)
    result.z = clamp(color.z + b, 0.cfloat .. 1.cfloat)
    result.w = color.w

const IM_COL32_WHITE = 0xffffffff.ImU32

func make_filter(filter: tuple[name:string, exts:seq[string]]): string =
    let
        dotted = filter.exts.map((e) => "." & e)
        exts_reg = join(dotted, ",")
        exts_label = join(dotted, " ")
    return fmt"{filter.name} ({exts_label}){{{exts_reg}}}"

func make_all_filter(file_filters: seq[tuple[name:string, exts:seq[string]]]): string =
    let exts = file_filters.map((f) => f.exts).concat().map((e) => "." & e).join(",")
    return fmt"Image files (*){{{exts}}}"

const file_filters = @[
    ("Crowpainter", @["crow"]),
    ("PNG", @["png"]),
    ("JPEG", @["jpg", "jpeg"]),
    ("Photoshop", @["psd", "psb"]),
    ("Targa", @["tga"]),
    ("Bitmap", @["bmp"]),
]

const save_filters = file_filters.map(make_filter).join(",")
const open_filters = fmt"{make_all_filter(file_filters)},{save_filters}"

proc check_sdl(result: bool): void =
    if not result:
        quit(&"SDL Error: {SDL_GetError()}", -1)

# TODO: Not correct for actual 2D tile layouts; Only linearly converts to tiles.
proc parellel_convert[A, B](pool: ThreadPool, res: ref seq[A], data: ref seq[B]) =
    var futures = newSeq[Future]()
    when A is ColorPixelArray and B is util.Color[byte]:
        var res2 = cast[ptr UncheckedArray[ColorPixel]](res[][0].addr)
        for (first, last) in chunks(data[], data[].len div pool.get_worker_count):
            capture first, last:
                futures.add(pool.submit(() => convert(res2.toOpenArray(first, last), data[].toOpenArray(first, last))))
    elif A is util.Color[byte] and B is ColorPixelArray:
        var data2 = cast[ptr UncheckedArray[ColorPixel]](data[][0].addr)
        for (first, last) in chunks(res[], res[].len div pool.get_worker_count):
            capture first, last:
                futures.add(pool.submit(() => convert(res[].toOpenArray(first, last), data2.toOpenArray(first, last))))
    else:
        for (first, last) in chunks(res[], res[].len div pool.get_worker_count):
            capture first, last:
                futures.add(pool.submit(() => convert(res[].toOpenArray(first, last), data[].toOpenArray(first, last))))
    wait_all(futures)

proc create_texture(): GLuint =
    var texture: GLuint
    glGenTextures(1, texture.addr)
    return texture

proc create_pixel_buffer_object(): GLuint =
    var pbo: GLuint
    glGenBuffers(1, pbo.addr)
    return pbo

proc create_checker_sampler(): GLuint =
    var sampler: GLuint
    glGenSamplers(1, sampler.addr)
    # glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glSamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, -0.5)
    return sampler

proc create_canvas_sampler(): GLuint =
    var sampler: GLuint
    glGenSamplers(1, sampler.addr)
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glSamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, -0.5)
    return sampler

const CHECKER_SIZE = 32

proc set_checker_texture(texture: GLuint, color1, color2: uint8) =
    let size = CHECKER_SIZE
    var data: seq[ColorBytePixel]
    data.setLen(size * size * 4)
    let color1 = ColorBytePixel(r: color1, g: color1, b: color1, a: 255)
    let color2 = ColorBytePixel(r: color2, g: color2, b: color2, a: 255)
    let j = size * size * 2
    let sizex = size * 2
    for y in 0 ..< size:
        for x in 0 ..< size:
            let i = y * sizex + x
            data[i] = color1
            data[i + size] = color2
            data[j + i] = color2
            data[j + i + size] = color1
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB.GLint, (size * 2).GLsizei, (size * 2).GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, data[0].addr)
    glGenerateMipmap(GL_TEXTURE_2D)

proc set_checker_texture(texture: GLuint, color1, color2: float) =
    set_checker_texture(texture, (color1 * 255).uint8, (color2 * 255).uint8)

proc checker_setup_callback(drawlist: ptr ImDrawList, drawcmd: ptr ImDrawCmd): void {.cdecl.} =
    var app = cast[ptr App](drawcmd.UserCallbackData)
    glBindSampler(0, app[].checker_sampler)

proc canvas_setup_callback(drawlist: ptr ImDrawList, drawcmd: ptr ImDrawCmd): void {.cdecl.} =
    var app = cast[ptr App](drawcmd.UserCallbackData)
    glBindSampler(0, app[].canvas_sampler)

func to_ImTextureRef(texture: GLuint): ImTextureRef =
    ImTextureRef(internal_TexData: nil, internal_TexID: cast[ImTextureID](texture))

proc draw_checker_quad(drawlist: ptr ImDrawList, texture: ImTextureRef, center, size: Imvec2, angle: float) =
    let
        checker_size = 1 / CHECKER_SIZE
        cos_a = cos(angle)
        sin_a = sin(angle)
        p0 = center + igImRotate(vec2(-size.x, -size.y) * 0.5, cos_a, sin_a)
        p1 = center + igImRotate(vec2(size.x, -size.y) * 0.5, cos_a, sin_a)
        p2 = center + igImRotate(vec2(size.x, size.y) * 0.5, cos_a, sin_a)
        p3 = center + igImRotate(vec2(-size.x, size.y) * 0.5, cos_a, sin_a)
        uv0 = vec2(0, 0)
        uv1 = vec2(1, 0) * size * checker_size
        uv2 = vec2(1, 1) * size * checker_size
        uv3 = vec2(0, 1) * size * checker_size
    ImDrawList_AddImageQuad(drawlist, texture, p0, p1, p2, p3, uv0, uv1, uv2, uv3, IM_COL32_WHITE)

# Rotated quad:
# https://github.com/ocornut/imgui/issues/1982#issuecomment-408834301
proc draw_quad(drawlist: ptr ImDrawList, texture: ImTextureRef, center, size: Imvec2, angle: float) =
    let
        cos_a = cos(angle)
        sin_a = sin(angle)
        p0 = center + igImRotate(vec2(-size.x, -size.y) * 0.5, cos_a, sin_a)
        p1 = center + igImRotate(vec2(size.x, -size.y) * 0.5, cos_a, sin_a)
        p2 = center + igImRotate(vec2(size.x, size.y) * 0.5, cos_a, sin_a)
        p3 = center + igImRotate(vec2(-size.x, size.y) * 0.5, cos_a, sin_a)
        uv0 = vec2(0, 0)
        uv1 = vec2(1, 0)
        uv2 = vec2(1, 1)
        uv3 = vec2(0, 1)
    ImDrawList_AddImageQuad(drawlist, texture, p0, p1, p2, p3, uv0, uv1, uv2, uv3, IM_COL32_WHITE)

proc update_texture(composite_data: CompositeData, pixel_size, composite_tile_size, tile_offset, offset_size: VectorI) =
    glBindTexture(GL_TEXTURE_2D, composite_data.texture)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, composite_data.pbo)
    var data = cast[ptr UncheckedArray[ColorBytePixel]](glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY))
    if data != nil:
        for tile_y in countup(tile_offset.y, tile_offset.y + offset_size.y - 1):
            for tile_x in countup(tile_offset.x, tile_offset.x + offset_size.x - 1):
                let tile_i = tile_y * composite_tile_size.x + tile_x
                let tile = composite_data.composite[tile_i]
                for y in 0 ..< TILE_SIZE:
                    let data_y = tile_y * TILE_SIZE + y
                    if data_y >= pixel_size.y:
                        continue
                    for x in 0 ..< TILE_SIZE:
                        let data_x = tile_x * TILE_SIZE + x
                        if data_x >= pixel_size.x:
                            break
                        let data_i = data_y * pixel_size.x + data_x
                        let i = y * TILE_SIZE + x
                        convert(data[data_i], tile[i])
        discard glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
    else:
        echo "update_texture: NO DATA!"
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pixel_size.x.GLsizei, pixel_size.y.GLsizei, GL_RGBA, GL_UNSIGNED_BYTE, nil)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    glGenerateMipmap(GL_TEXTURE_2D)

proc resize_texture(composite_data: CompositeData, pixel_size: VectorI) =
    glBindTexture(GL_TEXTURE_2D, composite_data.texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA.GLint, pixel_size.x.GLsizei, pixel_size.y.GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, nil)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, composite_data.pbo)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pixel_size.x * pixel_size.y * 4, nil, GL_STREAM_DRAW)

func get_current_canvas(project: Project): Canvas =
    project.states[project.current_index]

func is_saved(project: Project): bool =
    project.saved_id == some(project.get_current_canvas().change_id)

func set_saved(project: Project) =
    project.saved_id = some(project.get_current_canvas().change_id)

func push_canvas(project: Project, canvas: Canvas, stack_limit: int = 100) =
    var canvas = canvas
    canvas.change_id = project.next_id
    project.next_id += 1
    if project.states.len > 0:
        let last_index = project.states.len - 1
        if project.current_index < last_index:
            project.states.shrink(fromLast = last_index - project.current_index)
    project.states.addLast(canvas)
    if project.states.len > stack_limit:
        project.states.shrink(fromFirst = project.states.len() - stack_limit)
    project.current_index = project.states.len() - 1

func replace_current_canvas(project: Project, canvas: Canvas) =
    project.states[project.current_index] = canvas

func rebuild_layer_list(project: Project) =
    project.layer_list.layers.clear()
    let canvas = project.get_current_canvas()
    for layer in canvas.layer_lookup.values:
        project.layer_list.layers[layer.id] = LayerItemData(
            name: layer.name,
            visible: layer.visible,
            blend_mode: layer.blend_mode,
            opacity: opacity_to_view(layer.opacity),
        )

func get_next_viewport_id(app: App): ViewportId =
    result = app.next_viewport_id
    app.next_viewport_id += 1

func get_next_project_id(app: App): ProjectId =
    result = app.next_project_id
    app.next_project_id += 1

proc new_project(app: App, name: string, pixel_size: VectorI) =
    let canvas = new_blank_canvas(pixel_size)
    var project = new Project
    var viewport = new Viewport
    var composite = new CompositeData
    composite.composite = new_tile_array[ColorPixelArray](pixel_size)
    composite.texture = create_texture()
    composite.pbo = create_pixel_buffer_object()
    resize_texture(composite, pixel_size)
    let tile_size = get_tiles_size(pixel_size)
    update_texture(composite, pixel_size, tile_size, VectorI(x:0, y:0), tile_size - VectorI(x:1, y:1))
    viewport.id = app.get_next_viewport_id()
    project.id = app.get_next_project_id()
    viewport.project_id = project.id
    project.file_path = Path(name)
    project.push_canvas(canvas)
    project.viewports.incl(viewport.id)
    project.composites = project.composites.add(0.TimelinePos, composite)
    app.projects[project.id] = project
    app.viewports[viewport.id] = viewport
    app.active_project = project
    project.rebuild_layer_list()

func add_new_layer(app: App) =
    var project = app.active_project
    if project.isNil:
        return
    project.push_canvas(new_layer(project.get_current_canvas()))
    project.rebuild_layer_list()

func add_new_group(app: App) =
    var project = app.active_project
    if project.isNil:
        return
    project.push_canvas(new_group(project.get_current_canvas()))
    project.rebuild_layer_list()

proc draw_open_dialog(app: App) =
    if IGFD_DisplayDialog(app.file_dialog, "open-dialog", ImGui_WindowFlags_NoCollapse.ImGuiWindowFlags, vec2(500, 400), vec2(10000, 10000)):
        defer: IGFD_CloseDialog(app.file_dialog)
        if IGFD_IsOk(app.file_dialog):
            var sel = IGFD_GetSelection(app.file_dialog, IGFD_ResultMode_AddIfNoFileExt.IGFD_ResultMode)
            defer: IGFD_Selection_DestroyContent(sel.addr)
            let table = cast[ptr UncheckedArray[IGFD_Selection_Pair]](sel.table)
            for i in 0 ..< sel.count:
                echo fmt"{table[i].fileName}; {table[i].filePathName}"

proc draw_save_dialog(app: App) =
    if IGFD_DisplayDialog(app.file_dialog, "save-dialog", ImGui_WindowFlags_NoCollapse.ImGuiWindowFlags, vec2(500, 400), vec2(10000, 10000)):
        defer: IGFD_CloseDialog(app.file_dialog)
        if IGFD_IsOk(app.file_dialog):
            var file_name: cstring
            file_name = IGFD_GetFilePathName(app.file_dialog, IGFD_ResultMode_AddIfNoFileExt.IGFD_ResultMode)
            echo file_name

proc draw_new_canvas_dialog(app: App) =
    let id = "New Canvas###new-canvas-dialog"
    if app.new_canvas_dialog.show:
        igOpenPopup_Str(id.cstring, ImGui_PopupFlags_None.ImGuiPopupFlags)
        app.new_canvas_dialog.show = false
    var center = igGetMainViewport().ImGuiViewport_GetCenter()
    igSetNextWindowPos(center, ImGui_Cond_Appearing.ImGuiCond, vec2(0.5, 0.5))
    if igBeginPopupModal(id.cstring, nil, (ImGui_WindowFlags_AlwaysAutoResize).ImGuiWindowFlags):
        igInputText("File Name", app.new_canvas_dialog.name.cstring, app.new_canvas_dialog.name.len().csize_t, ImGui_InputTextFlags_None.ImGuiInputTextFlags, nil, nil)
        igInputInt("Width", cast[ptr cint](app.new_canvas_dialog.size.x.addr), 10, 100, ImGui_InputTextFlags_CharsDecimal.ImGuiInputTextFlags)
        igInputInt("Height", cast[ptr cint](app.new_canvas_dialog.size.y.addr), 10, 100, ImGui_InputTextFlags_CharsDecimal.ImGuiInputTextFlags)
        igColorEdit3("Background", app.new_canvas_dialog.background_color, ImGui_ColorEditFlags_NoInputs.ImGuiColorEditFlags)
        igCheckbox("Transparent Background", app.new_canvas_dialog.background_transparent.addr)
        igBeginGroup()
        if igButton("OK", vec2(0, 0)):
            igCloseCurrentPopup()
            app.new_project(app.new_canvas_dialog.name, app.new_canvas_dialog.size)
        igSameLine(0, 10)
        if igButton("Cancel", vec2(0, 0)):
            igCloseCurrentPopup()
        igEndGroup()
        igEndPopup()

func get_zoom_factor(level: int): float =
    scroll_zoom_levels[clamp(level, 0, scroll_zoom_levels.len - 1)]

proc draw_viewport(app: App, viewport: Viewport) =
    let project = app.projects[viewport.project_id]
    if igBegin(fmt"{project.file_path}###{viewport.id.int}".cstring, viewport.is_open.addr, (ImGui_WindowFlags_None).ImGuiWindowFlags):
        let
            canvas = project.get_current_canvas()
            win_center = igGetWindowPos() + igGetWindowSize() * 0.5
            scale = get_zoom_factor(viewport.zoom_level)
            angle = viewport.rotation
            offset = igImTrunc_Vec2(win_center + viewport.position)
            canvas_size = igImTrunc_Vec2(vec2(canvas.size) * scale)
            composite = project.composites.get(project.timeline_pos)
        case composite
        of Some(comp):
            let drawlist = igGetWindowDrawList()
            # if transparent background:
            ImDrawList_AddCallback(drawlist, checker_setup_callback, app.addr, sizeof(app.addr).csize_t)
            draw_checker_quad(drawlist, to_ImTextureRef(app.checker_texture), offset, canvas_size, angle)
            ImDrawList_AddCallback(drawlist, canvas_setup_callback, app.addr, sizeof(app.addr).csize_t)
            draw_quad(drawlist, to_ImTextureRef(comp.texture), offset, canvas_size, angle)
            ImDrawList_AddCallback(drawlist, igGetPlatformIO_Nil().DrawCallback_ResetRenderState, nil, 0)
        of None: discard
    igEnd()

proc close_viewport(app: App, viewport: Viewport) =
    var project = app.projects[viewport.project_id]
    if project.viewports.len == 1 and not project.is_saved():
        # show save query
        discard
    # else:
    # Delete viewport
    project.viewports.excl(viewport.id)
    app.viewports.del(viewport.id)
    # Switch to next viewport. Probably need to get window from imgui
    if app.viewports.len > 0:
        for viewport in app.viewports.values:
            app.active_project = app.projects[viewport.project_id]
    else:
        app.active_project = nil

proc draw_viewports(app: App) =
    var to_close: seq[Viewport]
    for viewport in app.viewports.mvalues:
        app.draw_viewport(viewport)
        if not viewport.is_open:
            to_close.add(viewport)
    for viewport in to_close:
        app.close_viewport(viewport)

proc draw_layer_item(app: App, project: Project, layer_id: LayerId) =
    var canvas = project.get_current_canvas()
    var layer = canvas.get(layer_id)
    var layer_widget = project.layer_list.layers[layer_id]
    let id = layer.id.int
    let button_flags = ImGui_ButtonFlags_PressedOnClick
    let indent = 20.0
    let style = igGetStyle()
    let rounding = style[].FrameRounding
    let border_size = style[].FrameBorderSize
    let spacing = style[].ItemInnerSpacing
    var start = igGetCursorScreenPos()
    let drawlist = igGetWindowDrawList()
    drawlist.ImDrawList_ChannelsSplit(2)
    drawlist.ImDrawList_ChannelsSetCurrent(1)

    drawlist.ImDrawList_PushClipRect(start, layer_widget.item_rect_max, true)
    if igCheckbox(fmt"###layer-visible-{id}".cstring, layer_widget.visible.addr):
        layer.visible = layer_widget.visible
        canvas.update(layer)
        project.replace_current_canvas(canvas)
    igSameLine(0, spacing.x)

    case layer
    of Group(group):
        let arrow_dir = if group.open: ImGui_Dir_Down else: ImGuiDir_Right
        if igArrowButton(fmt"###group-open-{id}".cstring, arrow_dir.ImGuiDir):
            group.open = not group.open
            canvas.update(layer)
            project.replace_current_canvas(canvas)
    else:
        igDummy(vec2(20, -1))
    igSameLine(0, spacing.x)

    igBeginGroup()
    igPushStyleVar_Y(ImGui_StyleVar_ItemSpacing.ImGuiStyleVar, 0.0)
    igText(layer.name.cstring)
    let font_size = igGetFontSize() * 0.75
    igPushFont(nil, font_size)
    igText(get_blend_mode_name(layer.blend_mode).cstring)
    igText(fmt"{opacity_to_view(layer.opacity).int}%%".cstring)
    igPopFont()
    igPopStyleVar(1)
    igEndGroup()
    drawlist.ImDrawList_PopClipRect()

    drawlist.ImDrawList_ChannelsSetCurrent(0)
    let cur_end = igGetCursorScreenPos()
    var height = cur_end.y - start.y

    igSetCursorScreenPos(start)
    igSetNextItemAllowOverlap()
    igInvisibleButton(fmt"###layer-node-{id}".cstring, vec2(-1, height), button_flags.ImGuiButtonFlags)
    var factor = 0.0
    if layer_id in canvas.multi_selected_layers:
        factor += 0.4
    if igIsItemHovered(ImGui_HoveredFlags_None.ImguiHoveredFlags):
        factor += 0.25
        if igIsMouseDoubleClicked_Nil(ImGui_MouseButton_Left.ImGuiMouseButton):
            debugEcho &"double clicked {layer.name}"
            # TODO do rename layer here
        elif igIsMouseClicked_Bool(ImGui_MouseButton_Left.ImGuiMouseButton, false):
            canvas.selected_layer = some(layer_id)
            canvas.multi_selected_layers = parazoa.toSet([layer_id])
            project.replace_current_canvas(canvas)
        if igIsMouseDown_Nil(ImGui_MouseButton_Left.ImGuiMouseButton):
            factor += 0.25
    let rect_min = igGetItemRectMin()
    let rect_max = igGetItemRectMax()
    var color = igImLerp_Vec4(igGetStyleColorVec4(ImGui_Col_FrameBg.ImGuiCol)[], igGetStyleColorVec4(ImGui_Col_FrameBgActive.ImGuiCol)[], factor)
    if layer.is_group():
        color = add_hsv(color, 0, -0.2, -0.2)
    drawlist.ImDrawList_AddRectFilled(rect_min, rect_max, to_color(color), rounding, ImDrawFlags_None.ImDrawFlags)
    if layer.is_group():
        let border_color = igGetStyleColorVec4(ImGui_Col_Border.ImGuiCol)[]
        drawlist.ImDrawList_AddRect(rect_min, rect_max, to_color(border_color), rounding, border_size, ImDrawFlags_None.ImDrawFlags)

    drawlist.ImDrawList_ChannelsMerge()

    # Hacky 1 frame delay size recomputation stuff.
    layer_widget.item_rect_max = rect_max

    case layer
    of Group(group):
        if group.open:
            igIndent(indent)
            for layer in reverse_group_layers_iter(canvas, group):
                draw_layer_item(app, project, layer.id)
            igUnIndent(indent)
    else: discard

proc draw_layers_list(app: App) =
    if igBegin("Layers", nil, (ImGui_WindowFlags_None).ImGuiWindowFlags):
        let project = app.active_project
        if not project.isNil:
            let canvas = project.get_current_canvas()
            for layer in reverse_group_layers_iter(canvas):
                draw_layer_item(app, project, layer.id)
    igEnd()

proc draw_menu_bar(app: App) =
    if igBeginMainMenuBar():
        if igBeginMenu("File", true):
            var want_save_as = false
            if igMenuItem_Bool("New ...", nil, false, true):
                app.new_canvas_dialog.show = true
            if igMenuItem_Bool("Open ...", nil, false, true):
                var config = IGFD_FileDialog_Config_Get()
                config.path = app.last_dir.cstring
                config.countSelectionMax = 0
                config.flags = (ImGui_FileDialogFlags_Modal or
                    ImGui_FileDialogFlags_ShowDevicesButton or
                    ImGui_FileDialogFlags_CaseInsensitiveExtentionFiltering).ImGuiFileDialogFlags
                IGFD_OpenDialog(app.file_dialog,
                    "open_dialog",
                    "Open",
                    open_filters,
                    config)
            # TODO Check if there is something in the clipboard to enable or disable this:
            if igMenuItem_Bool("New from Clipboard ...", nil, false, true):
                discard
            if igMenuItem_Bool("Save ...", nil, false, true):
                # TODO Check if current viewport.canvas has a save path
                # Save if it does else run save dialog.
                want_save_as = true
            if igMenuItem_Bool("Save As...", nil, false, true) or want_save_as:
                var config = IGFD_FileDialog_Config_Get()
                config.path = app.last_dir.cstring
                config.countSelectionMax = 1
                config.flags = (ImGui_FileDialogFlags_Modal or
                    ImGui_FileDialogFlags_ShowDevicesButton or
                    ImGui_FileDialogFlags_CaseInsensitiveExtentionFiltering).ImGuiFileDialogFlags
                IGFD_OpenDialog(app.file_dialog,
                    "save_dialog",
                    "Save",
                    save_filters,
                    config)
            if igMenuItem_Bool("Close ...", nil, false, true):
                discard
            if igMenuItem_Bool("Exit ...", nil, false, true):
                app.running = false
            igEndMenu()
        if igBeginMenu("Edit", true):
            igEndMenu()
        if igBeginMenu("Canvas", true):
            igEndMenu()
        if igBeginMenu("Layer", true):
            igEndMenu()
        if igBeginMenu("Selection", true):
            igEndMenu()
        if igBeginMenu("View", true):
            igEndMenu()
        if igBeginMenu("Window", true):
            igEndMenu()
        igEndMainMenuBar()

proc `=destroy`(app: var typeof(App()[])) =
    IGFD_Destroy(app.file_dialog)
    glDeleteTextures(1, app.checker_texture.addr)
    glDeleteSamplers(1, app.checker_sampler.addr)
    glDeleteSamplers(1, app.canvas_sampler.addr)

proc new_app(): App =
    result.new
    result.file_dialog = IGFD_Create()
    result.checker_texture = create_texture()
    set_checker_texture(result.checker_texture, 0.4, 0.6)
    result.checker_sampler = create_checker_sampler()
    result.canvas_sampler = create_canvas_sampler()
    result.new_canvas_dialog.name = newString(256)
    copy_into(result.new_canvas_dialog.name, "New Canvas")
    result.new_canvas_dialog.size = VectorI(x:2000, y:2000)
    result.new_canvas_dialog.background_color = [1.0, 1.0, 1.0]
    result.new_canvas_dialog.background_transparent = false

proc main(): void =
    # Initialize SDL
    var logger = newConsoleLogger()
    check_sdl(SDL_Init(SDL_INIT_VIDEO))
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)
    var window = SDL_CreateWindow("crowpainter", 800, 600, SDL_WINDOW_RESIZABLE or SDL_WINDOW_OPENGL)
    check_sdl(window != nil)
    defer: SDL_DestroyWindow(window)
    var gl_context = SDL_GL_CreateContext(window)
    check_sdl(gl_context != nil)
    loadExtensions()
    defer: SDL_GL_DestroyContext(gl_context)
    check_sdl(SDL_GL_MakeCurrent(window, gl_context))
    SDL_GL_SetSwapInterval(1)
    SDL_ShowWindow(window)

    # Initialize Imgui
    let ctx = igCreateContext(nil)
    defer: igDestroyContext(ctx)
    let io = igGetIO_ContextPtr(ctx)
    io.ConfigFlags = io.ConfigFlags or ImGui_ConfigFlags_DockingEnable.ImGuiConfigFlags
    io.ConfigWindowsMoveFromTitleBarOnly = true
    var font_cfg = ImFontConfig_ImFontConfig()
    defer: ImFontConfig_destroy(font_cfg)
    font_cfg[].FontDataOwnedByAtlas = false
    var data_temp = font_file
    let font = io.Fonts.ImFontAtlas_AddFontFromMemoryTTF(data_temp[0].addr, font_file.len().cint, 18, font_cfg, nil)
    data_temp = ""
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context)
    defer: ImGui_ImplSdl3_Shutdown()
    ImGui_ImplOpenGL3_Init("#version 430 core")
    defer: ImGui_ImplOpenGL3_Shutdown()

    # Init state
    var pool = new_thread_pool(2)
    var app = new_app()

    app.new_project(app.new_canvas_dialog.name, app.new_canvas_dialog.size)
    var style = igGetStyle()
    style[].FrameRounding = 4
    style[].WindowRounding = 4

    # TESTING
    app.add_new_layer()
    app.add_new_layer()
    app.add_new_layer()
    app.add_new_group()
    app.add_new_layer()
    app.add_new_layer()

    while app.running:
        var e: SDL_Event
        while SDL_PollEvent(addr e):
            if e.type_field.SDL_EventType == SDL_EVENT_QUIT:
                app.running = false
                break
            ImGui_ImplSDL3_ProcessEvent(addr e)
            # if io.WantCaptureKeyboard or io.WantCaptureMouse:
            let etype = e.type_field.SDL_EventType
            case etype:
                of SDL_EVENT_PEN_MOTION:
                    logger.log(lvlInfo, &"Pen Motion: {e.pmotion}")
                of SDL_EVENT_PEN_AXIS:
                    logger.log(lvlInfo, &"Pen Axis: {e.paxis}")
                of SDL_EVENT_PEN_DOWN:
                    logger.log(lvlInfo, &"Pen Down: {e.ptouch}")
                of SDL_EVENT_PEN_UP:
                    logger.log(lvlInfo, &"Pen Up: {e.ptouch}")
                else:
                    discard

        ImGui_Impl_OpenGL3_NewFrame()
        ImGui_ImplSDL3_NewFrame()
        igNewFrame()

        igDockSpaceOverViewport(igGetID_Str("Main Dockspace"), igGetMainViewport(), (ImGui_DockNodeFlags_PassthruCentralNode).ImGuiDockNodeFlags, ImGuiWindowClass_ImGuiWindowClass())

        app.draw_menu_bar()
        app.draw_viewports()
        app.draw_layers_list()
        app.draw_new_canvas_dialog()
        app.draw_open_dialog()
        app.draw_save_dialog()

        # let style = igGetStyle()
        # igShowStyleEditor(style)

        igRender()
        glViewport(0, 0, io.DisplaySize.x.GLsizei, io.DisplaySize.y.GLsizei)
        let bg_color = style[].Colors[ImGui_Col_WindowBg.int]
        glClearColor(bg_color.x, bg_color.y, bg_color.z, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData())
        SDL_GL_SwapWindow(window)

when isMainModule:
    main()
