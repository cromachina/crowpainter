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
    let next = cpuTime()
    echo fmt"{str} {next - time}"
    time = next

type TestTaskData = ref object of RootObj
    blendfunc: typeof(blendfuncs.normal)
    first, last: int
    res, dst, src: ref seq[ColorPixel]
    mask: ref seq[MaskPixel]
    opacity: int16

type ConvArgs[A, B] = ref object of RootObj
    first, last: int
    res: ref seq[A]
    data: ref seq[B]

proc do_parallel_convert[A, B](data: ref RootObj) {.thread.} =
    let args = ConvArgs[A, B](data)
    blendfuncs.convert(args.res[].toOpenArray(args.first, args.last), args.data[].toOpenArray(args.first, args.last))

proc parellel_convert[A, B](pool: ThreadPool, res: ref seq[A], data: ref seq[B]) =
    var futures = newSeq[Future]()
    var ix = 0
    var offset = 0
    var chunk_size = res[].len div pool.get_worker_count()
    while true:
        var args = ConvArgs[A, B](first: offset, last: min(offset + chunk_size, res[].len) - 1, res: res, data:data)
        futures.add(pool.submit(ix, do_parallel_convert[A, B], args))
        ix += 1
        offset += chunk_size
        if offset >= res[].len:
            break
    wait_all(futures)

proc do_blend_task_test(data: ref RootObj) {.thread.} =
    let
        args = TestTaskData(data)
        first = args[].first
        last = args[].last
    args[].blendfunc(args.res[].toOpenArray(first, last), args.dst[].toOpenArray(first, last), args.src[].toOpenArray(first, last), args.mask[], blendfuncs.range_max)

proc test() =
    var total = cpuTime()
    var time = cpuTime()

    var pool = new_thread_pool(2)
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
    let chunk_size = res[].len div pool.get_worker_count()
    var ix = 0
    var offset = 0
    while true:
        var data = TestTaskData(blendfunc: blendfuncs.normal, first:offset, last:min(offset + chunk_size, res[].len) - 1, res: res, dst: bg2, src:fg2, mask:mask, opacity: blendfuncs.range_max)
        futures.add(pool.submit(ix, do_blend_task_test, data))
        ix += 1
        offset += chunk_size
        if offset >= res[].len:
            break
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
