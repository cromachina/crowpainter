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
import std/sugar

import canvas_data
import composite
import util
import blendfuncs

proc check_sdl(result: bool): void =
    if not result:
        quit(&"SDL Error: {SDL_GetError()}", -1)

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

import threadpool
import image_io
import times

proc timer(time: var float, str: string) =
    let next = epochTime()
    echo fmt"{str} {next - time}"
    time = next

proc parellel_convert[A, B](pool: ThreadPool, res: ref seq[A], data: ref seq[B]) =
    var futures = newSeq[Future]()
    let chunk_size = res[].len div pool.get_worker_count()
    for first in countup(0, res[].len, chunk_size):
        let last = min(first + chunk_size, res[].len) - 1
        capture first, last:
            futures.add(pool.submit(() => blendfuncs.convert(res[].toOpenArray(first, last), data[].toOpenArray(first, last))))
    wait_all(futures)

proc test() =
    var total = epochTime()
    var time = epochTime()

    var pool = new_thread_pool(true, 2)
    timer(time, "load threadpool")

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
    timer(time, "alloc result")

    var futures = newSeq[Future]()
    let chunk_size = TILE_DATA_SIZE * 3
    for first in countup(0, res[].len, chunk_size):
        let last = min(first + chunk_size, res[].len) - 1
        capture first, last:
            futures.add(pool.submit(() => blendfuncs.normal(res[].toOpenArray(first, last), bg2[].toOpenArray(first, last), fg2[].toOpenArray(first, last), mask[], blendfuncs.range_max)))
    wait_all(futures)

    timer(time, "composite")

    var out_image = image_io.new_image(fg.width, fg.height)
    parellel_convert(pool, out_image.data, res)
    timer(time, "convert res")

    if not out_image.write_jpg("test.jpg"):
        echo "failed to write image"
    timer(time, "write")

    pool.shutdown()
    timer(time, "shutdown threadpool")

    timer(total, "total")

when isMainModule:
    # main()
    test()
