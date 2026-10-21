import imguin/sdl3_opengl
import opengl
import parazoa
import sdl3_nim as sdl

import std/locks
import std/logging
import std/math
import std/os
import std/strformat
import std/sugar
import std/times
import std/typedthreads

import blendfuncs
import canvas_data
import composite
import image_io
import threadpool
import util

proc check_sdl(result: bool): void =
    if not result:
        quit(&"SDL Error: {SDL_GetError()}", -1)

proc parellel_convert[A, B](pool: ThreadPool, res: ref seq[A], data: ref seq[B]) =
    var futures = newSeq[Future]()
    for (first, last) in chunks(res[], res[].len div pool.get_worker_count):
        capture first, last:
            futures.add(pool.submit(() => blendfuncs.convert(res[].toOpenArray(first, last), data[].toOpenArray(first, last))))
    wait_all(futures)

# Maybe make a shader to render a tiled texture for canvases and render that
# into a viewport in imgui's callbacks, or all canvases into framebuffers to reduce context switching.

const canvas_vertex_shader = """
    #version 430 core
    layout (location = 0) in vec2 Position;
    layout (location = 1) in vec2 UV;
    layout (location = 2) in vec4 Color;
    uniform mat4 ProjMtx;
    out vec2 Frag_UV;
    out vec4 Frag_Color;
    void main()
    {
        Frag_UV = UV;
        Frag_Color = Color;
        gl_Position = ProjMtx * vec4(Position.xy,0,1);
    }
"""

const canvas_fragment_shader = """
    #version 430 core
    in vec2 Frag_UV;
    in vec4 Frag_Color;
    uniform sampler2D Texture;
    layout (location = 0 ) out vec4 Out_Color;
    void main() {
        Out_Color = Frag_Color * texture(Texture, Frag_UV.st, -0.5);
    }
"""

proc check_shader(handle: GLuint, desc: string) =
    var status: GLint
    var log_length: GLint
    glGetShaderiv(handle, GL_COMPILE_STATUS, status.addr)
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, log_length.addr)
    if status.GLBoolean == GL_FALSE:
        echo fmt"Failed to compile {desc}"
    if log_length > 1:
        var buffer = cast[cstring](alloc(log_length + 1))
        defer: dealloc(buffer)
        glGetShaderInfoLog(handle, log_length, nil, buffer)
        echo buffer

proc check_program(handle: GLuint, desc: string) =
    var status: GLint
    var log_length: GLint
    glGetProgramiv(handle, GL_LINK_STATUS, status.addr)
    glGetProgramiv(handle, GL_INFO_LOG_LENGTH, log_length.addr)
    if status.GLBoolean == GL_FALSE:
        echo fmt"Failed to link {desc}"
    if log_length > 1:
        var buffer = cast[cstring](alloc(log_length + 1))
        defer: dealloc(buffer)
        glGetProgramInfoLog(handle, log_length, nil, buffer)
        echo buffer

proc create_canvas_shader_program(): GLuint =
    var vert_handle = glCreateShader(GL_VERTEX_SHADER)
    var arr = allocCStringArray([canvas_vertex_shader])
    defer: deallocCStringArray(arr)
    glShaderSource(vert_handle, 1, arr, nil)
    glCompileShader(vert_handle)
    check_shader(vert_handle, "vertex shader")
    var frag_handle = glCreateShader(GL_FRAGMENT_SHADER)
    var arr2 = allocCStringArray([canvas_fragment_shader])
    defer: deallocCStringArray(arr2)
    glShaderSource(frag_handle, 1, arr2, nil)
    glCompileShader(frag_handle)
    check_shader(frag_handle, "fragment shader")
    var program = glCreateProgram()
    glAttachShader(program, vert_handle)
    glAttachShader(program, frag_handle)
    glLinkProgram(program)
    check_program(program, "program")
    glDetachShader(program, vert_handle)
    glDetachShader(program, frag_handle)
    glDeleteShader(vert_handle)
    glDeleteShader(frag_handle)
    return program

proc create_canvas_texture(): GLuint =
    var texture: GLuint
    glGenTextures(1, texture.addr)
    glBindTexture(GL_TEXTURE_2D, texture)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.5)
    return texture

proc create_canvas_framebuffer() =
    var fbo: GLuint
    glGenFrameBuffers(1, fbo.addr)
    var fbo_tex: GLuint
    glGenTextures(1, fbo_tex.addr)
    glBindTexture(GL_TEXTURE_2D, fbo_tex)

proc create_sampler(): GLuint =
    var sampler: GLuint
    glGenSamplers(1, sampler.addr)
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glSamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, -0.5)
    return sampler

var canvas_program: GLuint
var canvas_sampler: GLuint

proc canvas_setup_callback(a0: ptr ImDrawList, a1: ptr ImDrawCmd): void {.cdecl.} =
    # glUseProgram(canvas_program)
    glBindSampler(0, canvas_sampler)

func vec2(x, y: float): ImVec2 =
    ImVec2(x:x, y:y)

func `+`(a, b: ImVec2): ImVec2 =
    vec2(a.x + b.x, a.y + b.y)

func `*`(a: ImVec2, b: float): ImVec2 =
    vec2(a.x * b, a.y * b)

const IM_COL32_WHITE = 0xffffffff.ImU32

# Rotated quad:
# https://github.com/ocornut/imgui/issues/1982#issuecomment-408834301
proc image_rotated(tex_id: GLuint, center, size: Imvec2, angle: float) =
    let
        cos_a = cos(angle)
        sin_a = sin(angle)
        p0 = center + igImRotate(vec2(-size.x * 0.5, -size.y * 0.5), cos_a, sin_a)
        p1 = center + igImRotate(vec2(size.x * 0.5, -size.y * 0.5), cos_a, sin_a)
        p2 = center + igImRotate(vec2(size.x * 0.5, size.y * 0.5), cos_a, sin_a)
        p3 = center + igImRotate(vec2(-size.x * 0.5, size.y * 0.5), cos_a, sin_a)
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
    var pool = new_thread_pool(true, 2)

    # canvas_program = create_canvas_shader_program()
    canvas_sampler = create_sampler()

    var image = image_io.load("fg.png")
    var canvas_texture = create_canvas_texture()
    glBindTexture(GL_TEXTURE_2D, canvas_texture)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA.GLint, image.width.GLsizei, image.height.GLsizei, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data[0].addr)
    glGenerateMipmap(GL_TEXTURE_2D)

    var running = true
    while running:
        var e: SDL_Event
        while SDL_PollEvent(addr e):
            if e.type_field.enum_SDL_EventType == SDL_EVENT_QUIT:
                running = false
                break
            ImGui_ImplSDL3_ProcessEvent(addr e)
            if io.WantCaptureKeyboard or io.WantCaptureMouse:
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
                if igMenuItem_Bool("New ...", nil, false, true):
                    discard
                if igMenuItem_Bool("Open ...", nil, false, true):
                    discard
                if igMenuItem_Bool("New from Clipboard ...", nil, false, true):
                    discard
                if igMenuItem_Bool("Save ...", nil, false, true):
                    discard
                if igMenuItem_Bool("Save As...", nil, false, true):
                    discard
                if igMenuItem_Bool("Close ...", nil, false, true):
                    discard
                if igMenuItem_Bool("Exit ...", nil, false, true):
                    discard
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

        igDockSpaceOverViewport(0, igGetMainViewport(), ImGui_DockNodeFlags_PassthruCentralNode.ImGuiDockNodeFlags, ImGuiWindowClass_ImGuiWindowClass())
        igBegin("canvas", nil, (ImGui_WindowFlags_None).ImGuiWindowFlags)

        let
            win_center = igGetWindowPos() + igGetWindowSize() * 0.5
            scale = 0.1
            offset = vec2(0, 0) + win_center
            image_size = vec2(image.width.float, image.height.float) * scale
        image_rotated(canvas_texture, win_center, image_size, 45)

        igEnd()

        igRender()
        glViewport(0, 0, io.DisplaySize.x.GLsizei, io.DisplaySize.y.GLsizei)
        glClearColor(0.5, 0.5, 0.6, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData())
        SDL_GL_SwapWindow(window)

proc test() =
    var total = epochTime()
    var time = epochTime()
    var pool = new_thread_pool(true, 2)

    var fg = image_io.load("fg.png")
    timer(time, "load fg")

    var bg = image_io.load("bg.png")
    timer(time, "load bg")

    var fg2 = new_ref_seq_uninit[ColorPixel](fg.data[].len)
    parellel_convert(pool, fg2, fg.data)
    timer(time, "convert fg")

    var bg2 = new_ref_seq_uninit[ColorPixel](bg.data[].len)
    parellel_convert(pool, bg2, bg.data)
    timer(time, "convert bg")

    var res = new_ref_seq_uninit[ColorPixel](fg2[].len)
    var mask = new_ref_seq_uninit[int16](1)
    mask[][0] = range_max
    var futures = newSeq[Future]()
    for i in chunks(res[], TILE_DATA_SIZE * 3):
        capture i:
            futures.add(pool.submit(() => blendfuncs.normal(res[].to_oa(i), bg2[].to_oa(i), fg2[].to_oa(i), mask[], blendfuncs.range_max)))
    wait_all(futures)
    timer(time, "composite")

    var out_image = image_io.new_image(fg.width, fg.height)
    parellel_convert(pool, out_image.data, res)
    timer(time, "convert res")

    if not out_image.write_jpg("test.jpg"):
        echo "failed to write image"
    timer(time, "write")
    timer(total, "total")

when isMainModule:
    main()
    # test()
