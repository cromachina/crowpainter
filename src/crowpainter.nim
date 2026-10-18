when not compileOption("mm", "atomicArc"):
    {.fatal: "This program must be compiled with --mm:atomicArc".}

when not compileOption("threads"):
    {.fatal: "This program must be compiled with --threads:on".}

import imguin/sdl3_renderer
import sdl3_nim as sdl
import parazoa

import std/strformat
import std/logging
import std/typedthreads
import std/locks
import std/os

import canvas_data
import composite
import util
import blendfuncs

proc check_sdl(result: bool): void =
    if not result:
        quit(&"SDL Error: {SDL_GetError()}", -1)

# var
#     comp_lock: Lock

# initLock(comp_lock)

# proc composite_thread(data: tuple[canvas: ref Canvas]) {.thread.} =
#     comp_lock.acquire()
#     defer: comp_lock.release()
#     data.canvas.size = Vector(x:200, y:200)
#     data.canvas.top_level = data.canvas.top_level.add(Layer())
#     sleep(1000)
#     echo data.canvas[]

# proc test_thread_temp_data() =
#     var canvas = new Canvas
#     block:
#         var worker: Thread[tuple[canvas: ref Canvas]]
#         createThread(worker, composite_thread, (canvas,))
#         echo canvas[]
#         sleep(500)
#         comp_lock.acquire()
#         echo canvas[]
#         comp_lock.release()
#         sleep(1000)
#         joinThread(worker)

# test_thread_temp_data()

# deinitLock(comp_lock)

proc main(): void =
    var logger = newConsoleLogger()
    check_sdl(SDL_Init(SDL_INIT_VIDEO))
    var window: ptr SDL_Window
    var renderer: ptr SDL_Renderer
    check_sdl(SDL_CreateWindowAndRenderer("video-mask-painter", 800, 600,
        SDL_WINDOW_RESIZABLE or SDL_WINDOW_OPENGL, addr window, addr renderer))
    defer:
        SDL_DestroyWindow(window)
        SDL_DestroyRenderer(renderer)
    check_sdl(SDL_SetRenderVSync(renderer, 1))

    let ctx = igCreateContext(nil)
    defer: igDestroyContext(ctx)
    let io = igGetIO_ContextPtr(ctx)
    io.ConfigFlags = io.ConfigFlags or ImGui_ConfigFlags_DockingEnable.ImGuiConfigFlags
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer)
    defer: ImGui_ImplSDL3_Shutdown()
    ImGui_ImplSDLRenderer3_Init(renderer)
    defer: ImGui_ImplSDLRenderer3_Shutdown()

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

        ImGui_ImplSDLRenderer3_NewFrame()
        ImGui_ImplSDL3_NewFrame()
        igNewFrame()

        igShowDemoWindow(nil)
        igBegin("hello", nil, (ImGui_WindowFlags_None or ImGui_WindowFlags_NoCollapse).ImGuiWindowFlags)
        igEnd()

        igRender()
        SDL_SetRenderScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y)
        SDL_SetRenderDrawColorFloat(renderer, 0.5, 0.5, 0.6, 1.0)
        SDL_RenderClear(renderer)
        ImGui_ImplSDLRenderer3_RenderDrawData(cast[ptr impl_sdlrenderer3.ImDrawData](igGetDrawData()), renderer)
        SDL_RenderPresent(renderer)

import times
import image_io

proc timer(time: var float, str: string) =
    let next = cpuTime()
    echo fmt"{str} {next - time}"
    time = next

proc test() =
    # var canvas = Canvas()
    # echo canvas.size
    # var backdrop = composite_tile(canvas, canvas.top_level, 0.FrameId, Vector(x:0, y:0), ColorTile.new)
    var total = cpuTime()
    var time = cpuTime()
    var fg = image_io.load("fg.png", image_io.ChannelType.Rgba)
    timer(time, "load fg")

    var bg = image_io.load("bg.png", image_io.ChannelType.Rgba)
    timer(time, "load bg")

    var fg2 = blendfuncs.to_ColorPixel(fg.data)
    timer(time, "convert fg")
    var bg2 = blendfuncs.to_ColorPixel(bg.data)
    timer(time, "convert bg")

    var res = newSeqUninit[ColorPixel](fg2.len)
    var mask = newSeqUninit[int16](1)
    mask[0] = range_max
    timer(time, "alloc result")

    var first = (fg.channels * fg.width * 100).int
    var last = (fg.channels * fg.width * 300).int
    var slice = first .. last

    # blendfuncs.normal(res.toOpenArray(first, last), bg2[slice], fg2[slice], mask, blendfuncs.range_max)
    blendfuncs.normal(res, bg2, fg2, mask, blendfuncs.range_max)

    timer(time, "composite")

    var res2 = Image(width: fg.width, height:fg.height, channels:fg.channels)
    res2.data = blendfuncs.to_byte(res)
    timer(time, "convert res")

    if not res2.write_jpg("test.jpg"):
        echo "failed to write image"

    timer(time, "write")

    timer(total, "total")
    # var dst: ColorArray
    # var src: ColorArray
    # var mask: MaskArray
    # var mask_stride:uint8 = 0
    # var src_stride:uint8 = 1
    # blendfuncs.normal_2(dst, dst, src, src_stride, mask, mask_stride, range_max)

when isMainModule:
    # main()
    test()
