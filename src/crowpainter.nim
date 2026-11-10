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

import blendfuncs
import canvas_data
# import composite
import image_io
import threadpool
import util
import option

const scroll_zoom_levels = collect(newSeq):
    for x in countup(-28, 20):
        2.0 ^ (x.float / 4)
const default_zoom_level:int = 28

type
    Tool = ref object of RootObj
        name: string

    DynamicsProperty = object
        enabled: bool
        curve: seq[VectorF]

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

    CompositeData = object
        composite: ref seq[ColorPixelArray]
        texture: GLuint
        pbo: GLuint

    Viewport = ref object
        id: int
        position: ImVec2
        zoom_level: int = default_zoom_level
        rotation: float = 0
        last_mouse_pos: ImVec2
        panning: bool = false

    Project = ref object
        id: int
        file_path: Path
        saved_to_disk: bool = false
        states: Vec[Canvas]
        current_id:int = 0
        next_id: int = 1
        saved_id: int = -1
        composites: BBtree[TimelinePos, CompositeData]
        viewports: OrderedTable[int, Viewport]
        timeline_pos = 0.TimelinePos

    App = ref object
        next_project_id: int = 0
        next_viewport_id: int = 0
        projects: OrderedTable[int, Project]
        recent_projects: Deque[Path]
        active_project: Project
        active_tool: Tool
        brush_fg_color: ColorRgbBytePixel
        brush_bg_color: ColorRgbBytePixel
        brush_transparency: bool

    NewCanvasDialog = object
        show: bool
        name: string
        size: VectorI
        background: array[4, cfloat]

func vec2(x, y: float): ImVec2 =
    ImVec2(x:x, y:y)

func vec2(v: VectorI): ImVec2 =
    vec2(v.x.float, v.y.float)

func vec4(x, y, z, w: float): Imvec4 =
    ImVec4(x:x, y:y, z:z, w:w)

func `+`(a, b: ImVec2): ImVec2 =
    vec2(a.x + b.x, a.y + b.y)

func `*`(a: ImVec2, b: float): ImVec2 =
    vec2(a.x * b, a.y * b)

const IM_COL32_WHITE = 0xffffffff.ImU32

proc check_sdl(result: bool): void =
    if not result:
        quit(&"SDL Error: {SDL_GetError()}", -1)

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

proc create_sampler(): GLuint =
    var sampler: GLuint
    glGenSamplers(1, sampler.addr)
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glSamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, -0.5)
    return sampler

var canvas_sampler: GLuint
proc canvas_setup_callback(a0: ptr ImDrawList, a1: ptr ImDrawCmd): void {.cdecl.} =
    glBindSampler(0, canvas_sampler)

# Rotated quad:
# https://github.com/ocornut/imgui/issues/1982#issuecomment-408834301
proc image_rotated(tex_id: GLuint, center, size: Imvec2, angle: float) =
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
        drawlist = igGetWindowDrawList()
        tex_ref = ImTextureRef(internal_TexData: nil, internal_TexID: cast[ImTextureID](tex_id))
        pio = igGetPlatformIO_Nil()
    ImDrawList_AddCallback(drawlist, canvas_setup_callback, nil, 0)
    ImDrawList_AddImageQuad(drawlist, tex_ref, p0, p1, p2, p3, uv0, uv1, uv2, uv3, IM_COL32_WHITE)
    ImDrawList_AddCallback(drawlist, pio.DrawCallback_ResetRenderState, nil, 0)

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
                    for x in 0 ..< TILE_SIZE:
                        let data_x = tile_x * TILE_SIZE + x
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

func new_layer(canvas: Canvas): Canvas =
    discard

func new_blank_canvas(size: VectorI): Canvas =
    let frame = ColorFrame(
        id: result.next_frame_id,
        data: new_tile_array[ColorPixelArray](size),
    )
    var layer = Raster()
    {.cast(uncheckedAccess).}:
        layer.id = result.next_layer_id
        layer.name = fmt"Layer {layer.id.int}"
        layer.frames = layer.frames.add(0.TimelinePos, frame.id)
    result.size = size
    result.next_frame_id += 1.FrameId
    result.next_layer_id += 1.LayerId
    result.frame_lookup = result.frame_lookup.add(frame.id, frame)
    result.layer_lookup = result.layer_lookup.add(layer.id, layer)
    result.selected_frame = Some(val: frame.id)
    return insert_new_to_selected_layer(result, layer)

proc new_project(app: App, name: string, pixel_size: VectorI) =
    let canvas = new_blank_canvas(pixel_size)
    var project = new Project
    var viewport = new Viewport
    var composite: CompositeData
    composite.composite = new_tile_array[ColorPixelArray](pixel_size)
    composite.texture = create_texture()
    composite.pbo = create_pixel_buffer_object()
    resize_texture(composite, pixel_size)
    let tile_size = get_tiles_size(pixel_size)
    update_texture(composite, pixel_size, tile_size, VectorI(x:0, y:0), tile_size - VectorI(x:1, y:1))
    viewport.id = app.next_viewport_id
    project.id = app.next_project_id
    app.next_viewport_id += 1
    app.next_project_id += 1
    project.file_path = Path(name)
    project.states = project.states.add(canvas)
    project.viewports[viewport.id] = viewport
    project.composites = project.composites.add(0.TimelinePos, composite)
    app.projects[project.id] = project

proc handle_new_canvas_dialog(app: App, dialog: var NewCanvasDialog) =
    let id = "New Canvas"
    if dialog.show:
        igOpenPopup_Str(id.cstring, ImGui_PopupFlags_None.ImGuiPopupFlags)
        dialog.show = false
    var center = igGetMainViewport().ImGuiViewport_GetCenter()
    igSetNextWindowPos(center, ImGui_Cond_Appearing.ImGuiCond, vec2(0.5, 0.5))
    if igBeginPopupModal(id.cstring, nil, ImGui_WindowFlags_AlwaysAutoResize.ImGuiWindowFlags):
        igInputText("File Name", dialog.name.cstring, dialog.name.len().csize_t, ImGui_InputTextFlags_None.ImGuiInputTextFlags, nil, nil)
        igInputInt("Width", cast[ptr cint](dialog.size.x.addr), 10, 100, ImGui_InputTextFlags_CharsDecimal.ImGuiInputTextFlags)
        igInputInt("Height", cast[ptr cint](dialog.size.y.addr), 10, 100, ImGui_InputTextFlags_CharsDecimal.ImGuiInputTextFlags)
        igColorEdit4("Background", dialog.background, ImGui_ColorEditFlags_NoInputs.ImGuiColorEditFlags)
        igBeginGroup()
        if igButton("OK", vec2(0, 0)):
            igCloseCurrentPopup()
            new_project(app, dialog.name, dialog.size)
        igSameLine(0, 10)
        if igButton("Cancel", vec2(0, 0)):
            igCloseCurrentPopup()
        igEndGroup()
        igEndPopup()

proc show_viewport(viewport: Viewport) =
    discard

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
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context)
    defer: ImGui_ImplSdl3_Shutdown()
    ImGui_ImplOpenGL3_Init("#version 430 core")
    defer: ImGui_ImplOpenGL3_Shutdown()

    # Init state
    var pool = new_thread_pool(2)

    var app = new App
    var new_canvas_dialog: NewCanvasDialog
    new_canvas_dialog.name = newString(256)
    copy_into(new_canvas_dialog.name, "New Canvas")
    new_canvas_dialog.size = VectorI(x:2048, y:2048)
    new_canvas_dialog.background = [1.0, 1.0, 1.0, 1.0]
    var file_dialog = IGFD_Create()
    defer: IGFD_Destroy(file_dialog)
    var last_dir = "."

    canvas_sampler = create_sampler()

    new_project(app, new_canvas_dialog.name, new_canvas_dialog.size)

    # var image = image_io.load[ImageRgba]("fg.png")
    # var canvas_texture = create_texture()
    # glBindTexture(GL_TEXTURE_2D, canvas_texture)
    # glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA.GLint, image.size.x.GLsizei, image.size.y.GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data[0].addr)
    # glGenerateMipmap(GL_TEXTURE_2D)

    var running = true
    while running:
        var e: SDL_Event
        while SDL_PollEvent(addr e):
            if e.type_field.SDL_EventType == SDL_EVENT_QUIT:
                running = false
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

        if igBeginMainMenuBar():
            if igBeginMenu("File", true):
                var want_save_as = false
                if igMenuItem_Bool("New ...", nil, false, true):
                    new_canvas_dialog.show = true
                if igMenuItem_Bool("Open ...", nil, false, true):
                    var config = IGFD_FileDialog_Config_Get()
                    config.path = last_dir.cstring
                    config.countSelectionMax = 0
                    config.flags = (ImGui_FileDialogFlags_Modal or
                        ImGui_FileDialogFlags_ShowDevicesButton or
                        ImGui_FileDialogFlags_CaseInsensitiveExtentionFiltering).ImGuiFileDialogFlags
                    IGFD_OpenDialog(file_dialog,
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
                    config.path = last_dir.cstring
                    config.countSelectionMax = 1
                    config.flags = (ImGui_FileDialogFlags_Modal or
                        ImGui_FileDialogFlags_ShowDevicesButton or
                        ImGui_FileDialogFlags_CaseInsensitiveExtentionFiltering).ImGuiFileDialogFlags
                    IGFD_OpenDialog(file_dialog,
                        "save_dialog",
                        "Save",
                        save_filters,
                        config)
                if igMenuItem_Bool("Close ...", nil, false, true):
                    discard
                if igMenuItem_Bool("Exit ...", nil, false, true):
                    running = false
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

        igDockSpaceOverViewport(igGetID_Str("Main Dockspace"), igGetMainViewport(), (ImGui_DockNodeFlags_PassthruCentralNode).ImGuiDockNodeFlags, ImGuiWindowClass_ImGuiWindowClass())

        # Viewports
        for project in app.projects.mvalues:
            for viewport in project.viewports.mvalues:
                if igBegin((fmt"{project.file_path}###viewport-{viewport.id}").cstring, nil, (ImGui_WindowFlags_None).ImGuiWindowFlags):
                    let
                        canvas = project.states[project.current_id]
                        win_center = igGetWindowPos() + igGetWindowSize() * 0.5
                        scale = 1.0
                        angle = viewport.rotation
                        offset = win_center + viewport.position
                        canvas_size = vec2(canvas.size) * scale
                        composite = project.composites.get(project.timeline_pos)
                    # echo fmt"{angle} {offset} {canvas_size}"
                    case composite
                    of Some(comp):
                        image_rotated(comp.texture, offset, canvas_size, angle)
                    of None: discard
                igEnd()

        # Dialogs
        if IGFD_DisplayDialog(file_dialog, "open_dialog", ImGui_WindowFlags_NoCollapse.ImGuiWindowFlags, vec2(500, 400), vec2(10000, 10000)):
            defer: IGFD_CloseDialog(file_dialog)
            if IGFD_IsOk(file_dialog):
                var sel = IGFD_GetSelection(file_dialog, IGFD_ResultMode_AddIfNoFileExt.IGFD_ResultMode)
                defer: IGFD_Selection_DestroyContent(sel.addr)
                let table = cast[ptr UncheckedArray[IGFD_Selection_Pair]](sel.table)
                for i in 0 ..< sel.count:
                    echo fmt"{table[i].fileName}; {table[i].filePathName}"

        if IGFD_DisplayDialog(file_dialog, "save_dialog", ImGui_WindowFlags_NoCollapse.ImGuiWindowFlags, vec2(500, 400), vec2(10000, 10000)):
            defer: IGFD_CloseDialog(file_dialog)
            if IGFD_IsOk(file_dialog):
                var file_name: cstring
                file_name = IGFD_GetFilePathName(file_dialog, IGFD_ResultMode_AddIfNoFileExt.IGFD_ResultMode)
                echo file_name

        handle_new_canvas_dialog(app, new_canvas_dialog)

        igRender()
        glViewport(0, 0, io.DisplaySize.x.GLsizei, io.DisplaySize.y.GLsizei)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData())
        SDL_GL_SwapWindow(window)

when isMainModule:
    main()
