from concurrent.futures import Future, Executor
import threading
import weakref

from PySide6.QtCore import *

class _WorkItem(QRunnable):
    # Like thread._WorkItem, except a QRunnable
    def __init__(self, future, fn, args, kwargs):
        super().__init__(self)
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            self = None
        else:
            self.future.set_result(result)

class QThreadPoolExecutor(Executor):
    def __init__(self, max_workers=None) -> None:
        self._thread_pool = QThreadPool()
        self._workers = weakref.WeakSet()
        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        if max_workers:
            self._thread_pool.setMaxThreadCount(max_workers)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            future = Future()
            worker = _WorkItem(future, fn, args, kwargs)
            self._workers.add(worker)
            self._thread_pool.start(worker)
            return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            self._shutdown = True
            if cancel_futures:
                for work_item in self._workers:
                    work_item.future.cancel()
                self._workers.clear()
            if wait:
                self._thread_pool.waitForDone(-1)

    def get_thread_pool(self):
        return self._thread_pool
