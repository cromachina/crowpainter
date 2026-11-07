when not (compileOption("mm", "atomicArc") or compileOption("mm", "boehm")):
    {.fatal: "This program must be compiled with --mm:atomicArc or --mm:boehm".}

when not compileOption("threads"):
    {.fatal: "This program must be compiled with --threads:on".}

# A threadpool that assigns tasks to CPU bound threads based on an
# index so that the same index (mod worker count) with potentially the same
# data goes to the same CPU each time.

# If a shared channel is used for all threads, then the index doesn't matter.

import std/cpuinfo
import std/deques
import std/locks
import std/sysatomics
import std/typedthreads

type
    TaskProc* = typeof(proc() {.noSideEffect.})

    LCond* = ref object
        lock: Lock
        cond: Cond

    ChannelType[T] = object
        lcond: LCond
        queue: Deque[T]
    Channel*[T] = ref ChannelType[T]

    # Doesn't return anything, for now.
    Future* = ref object
        lcond: LCond
        finished: bool

    TaskData = tuple[future: Future, task: TaskProc]

    WorkerThread* = ref object
        channel: Channel[TaskData]
        running: bool
        thread: Thread[tuple[wt: WorkerThread]]

    ThreadPool* = ref object
        shutdown = false
        workers: seq[WorkerThread]

proc new_lcond*(): LCond =
    when defined boehmGC:
        result.new(proc (x: auto) =
            x.lock.deinitLock
            x.cond.deinitCond
        )
    else:
        result.new(proc (x: typeof(LCond()[])) =
            x.lock.deinitLock
            x.cond.deinitCond
        )
    result.lock.initLock
    result.cond.initCond

proc wait*(lcond: LCond) =
    lcond.cond.wait(lcond.lock)

proc signal*(lcond: LCond) =
    lcond.cond.signal()

proc new_channel*[T](): Channel[T] =
    result.new
    result.lcond = new_lcond()

proc put*[T](channel: Channel[T], value: T) =
    withLock channel.lcond.lock:
        channel.queue.addLast(value)
        channel.lcond.signal()

proc get*[T](channel: Channel[T]): T =
    withLock channel.lcond.lock:
        while channel.queue.len == 0:
            channel.lcond.wait()
        return channel.queue.popFirst()

proc new_future*(): Future =
    result.new
    result.lcond = new_lcond()

proc has_result*(future: Future): bool =
    atomicLoadN(future.finished.addr, ATOMIC_SEQ_CST)

proc wait_result*(future: Future) =
    withLock future.lcond.lock:
        if not future.has_result():
            future.lcond.wait()

proc set_result*(future: Future) =
    atomicStoreN(future.finished.addr, true, ATOMIC_SEQ_CST)
    future.lcond.signal()

proc wait_all*(futures: openArray[Future]) =
    for future in futures:
        future.wait_result()

proc worker_thread_runner(args: tuple[wt: WorkerThread]) {.thread.} =
    while atomicLoadN(args.wt.running.addr, ATOMIC_SEQ_CST):
        let (future, task) = args.wt.channel.get()
        withLock future.lcond.lock:
            task()
            future.set_result()

proc new_worker_thread*(channel: Channel[TaskData] = nil, affinity: int = -1): WorkerThread =
    result.new
    result.channel = if channel == nil: new_channel[TaskData]() else: channel
    atomicStoreN(result.running.addr, true, ATOMIC_SEQ_CST)
    result.thread.createThread(worker_thread_runner, (result,))
    if affinity >= 0:
        result.thread.pinToCpu(affinity)

func get_worker_count*(pool: ThreadPool): int =
    pool.workers.len

proc submit*(thread: WorkerThread, task: TaskProc): Future =
    var future = new_future()
    thread.channel.put((future, task))
    return future

# Check shutdown flag here, maybe, if using threadpool from other threads.
proc submit*(pool: ThreadPool, task: TaskProc, index:int = 0): Future =
    let worker = pool.workers[index mod pool.workers.len]
    return worker.submit(task)

# Maybe finish all pending tasks first.
# Matters more if threadpools are made from other threads.
proc `=destroy`(pool: var typeof(ThreadPool()[])) =
    if pool.shutdown:
        return
    pool.shutdown = true
    var future = new_future()
    for worker in pool.workers:
        atomicStoreN(worker.running.addr, false, ATOMIC_SEQ_CST)
    for worker in pool.workers:
        discard worker.submit(func () = discard)
    for worker in pool.workers:
        worker.thread.joinThread()
    pool.workers.setLen(0)

proc new_thread_pool*(logical_per_phsycial_cores:int = 1, worker_count: int = 0, shared_channels: bool = true): ThreadPool =
    let
        proc_count = max(1, cpuinfo.countProcessors())
        count = if worker_count == 0: max(1, proc_count div logical_per_phsycial_cores) else: worker_count
        channel = if shared_channels: new_channel[TaskData]() else: nil
    result.new
    for i in 0 ..< count:
        result.workers.add(new_worker_thread(channel))
