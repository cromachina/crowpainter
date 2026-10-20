when not compileOption("mm", "atomicArc"):
    {.fatal: "This program must be compiled with --mm:atomicArc".}

when not compileOption("threads"):
    {.fatal: "This program must be compiled with --threads:on".}

# A threadpool that assigns tasks to CPU bound threads based on an
# index so that the same index (mod worker count) with potentially the same
# data goes to the same CPU each time.

# If a shared channel is used for all threads, then the index doesn't matter.

import std/atomics
import std/cpuinfo
import std/deques
import std/locks
import std/sysatomics
import std/typedthreads

type
    TaskProc* = typeof(proc() {.noSideEffect.})

    ChannelType[T] = object
        lock: Lock
        cond: Cond
        queue: Deque[T]
    Channel[T] = ref ChannelType[T]

    # Doesn't return anything, for now.
    Future* = ref object
        lock: Lock
        cond: Cond
        finished: bool

    TaskData = tuple[future: Future, task: TaskProc]

    WorkerThread* = ref object
        channel: Channel[TaskData]
        running: bool
        thread: Thread[tuple[wt: WorkerThread]]

    ThreadPool* = ref object
        shutdown = false
        workers: seq[WorkerThread]

proc del_channel[T](obj: ChannelType[T]) =
    obj.lock.deinitLock
    obj.cond.deinitCond

proc new_channel*[T](): Channel[T] =
    result.new(del_channel[T])
    result.lock.initLock
    result.cond.initCond

proc put*[T](channel: Channel[T], value: T) =
    withLock channel.lock:
        channel.queue.addLast(value)
        channel.cond.signal()

proc get*[T](channel: Channel[T]): T =
    withLock channel.lock:
        while channel.queue.len == 0:
            channel.cond.wait(channel.lock)
        return channel.queue.popFirst()

proc del_future(obj: typeof(Future()[])) =
    obj.lock.deinitLock
    obj.cond.deinitCond

proc new_future*(): Future =
    result.new(del_future)
    result.lock.initLock
    result.cond.initCond

proc has_result*(future: Future): bool =
    atomicLoadN(future.finished.addr, ATOMIC_SEQ_CST)

proc wait_result*(future: Future) =
    withLock future.lock:
        if not future.has_result():
            future.cond.wait(future.lock)

proc set_result(future: Future) =
    atomicStoreN(future.finished.addr, true, ATOMIC_SEQ_CST)
    future.cond.signal()

proc wait_all*(futures: openArray[Future]) =
    for future in futures:
        future.wait_result()

proc worker_thread_runner(args: tuple[wt: WorkerThread]) {.thread.} =
    while atomicLoadN(args.wt.running.addr, ATOMIC_SEQ_CST):
        let (future, task) = args.wt.channel.get()
        withLock future.lock:
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

func noop() = discard

# Maybe add a shutdown flag and finish all pending tasks first.
# Matters more if threadpools are made from other threads.
proc shutdown*(pool: ThreadPool) =
    if pool.shutdown:
        return
    pool.shutdown = true
    var future = new_future()
    for worker in pool.workers:
        atomicStoreN(worker.running.addr, false, ATOMIC_SEQ_CST)
    for worker in pool.workers:
        discard worker.submit(noop)
    for worker in pool.workers:
        worker.thread.joinThread()
    pool.workers.setLen(0)

proc new_thread_pool*(shared_channels: bool = false, logical_per_phsycial_cores:int = 1, worker_count: int = 0): ThreadPool =
    let
        proc_count = max(1, cpuinfo.countProcessors())
        count = if worker_count == 0: max(1, proc_count div logical_per_phsycial_cores) else: worker_count
        channel = if shared_channels: new_channel[TaskData]() else: nil
    result.new(shutdown)
    for i in 0 ..< count:
        result.workers.add(new_worker_thread(channel))
