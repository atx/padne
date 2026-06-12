"""
Process/thread fan-out for embarrassingly parallel stages.

The Parallel class bundles one parallelism budget (Config.jobs) with the
lazily-created worker pool that enforces it, and provides two
order-preserving map primitives:

    thread_map(fn, items)   for work that releases the GIL (C++ kernels).
                            Shares memory with the caller, nothing is pickled.
    process_map(fn, items)  for GIL-bound pure-Python work. fn must be
                            importable top-level; items and results must be
                            cheap to pickle.

Both degrade to a plain serial loop in the caller's thread when jobs == 1,
so `--jobs 1` gives clean tracebacks, a working pdb and full typeguard
coverage.

Application code uses the module-level functions, which delegate to a
process-wide singleton instance (the same pattern as random.Random vs the
random module functions). Tests can instantiate their own Parallel, which
also works as a context manager that shuts its pool down on exit.

Worker processes are started with the "spawn" method; fork is not safe here
since the parent typically has pcbnew (and in GUI mode Qt) loaded. The spawn
pool is created lazily on first use and kept warm until shutdown().

Stage timings recorded via context.stage_timer inside process workers are
shipped back and merged into the parent's active timing session, so
parallelized stages stay visible in the timing summary. Their perf times sum
across workers and may exceed wall clock time for the enclosing stage.
"""

import atexit
import concurrent.futures
import functools
import logging
import multiprocessing
import os
import threading

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, TypeVar

from . import context

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _default_jobs() -> int:
    if hasattr(os, "sched_getaffinity"):
        # Unlike cpu_count(), this respects taskset/cgroup restrictions
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass(frozen=True)
class Config:
    """Parallelism configuration. The default budget is the number of
    available CPU cores."""
    jobs: int = field(default_factory=_default_jobs)

    def __post_init__(self):
        if self.jobs < 1:
            raise ValueError(f"jobs must be >= 1, got {self.jobs}")


def _init_worker(log_level: int) -> None:
    # Spawned workers start with unconfigured logging; mirror the parent's
    # root level so worker-side log output is not lost.
    logging.basicConfig(level=log_level)


def _timed_call(fn: Callable[[T], R], item: T
                ) -> tuple[R, dict[str, context.Session.Duration]]:
    """Worker-side wrapper: run fn under a fresh timing session and return
    its durations alongside the result so the parent can merge them."""
    with context.timing_session() as session:
        result = fn(item)
    return result, session.snapshot()


class Parallel:
    """One parallelism budget plus the worker pool that enforces it.

    The configuration is fixed at construction; to change the budget,
    create a new instance (the module-level configure() does exactly that
    for the shared singleton). Application code should normally go through
    the module-level functions; separate instances exist for tests and
    other scoped uses. Usable as a context manager: the pool is shut down
    on exit.
    """

    def __init__(self, config: Optional[Config] = None):
        self._lock = threading.Lock()
        self._config = config if config is not None else Config()
        self._process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def pool_is_live(self) -> bool:
        with self._lock:
            return self._process_pool is not None

    def shutdown(self) -> None:
        """Tear down the warm process pool. The next process_map call
        recreates it with the current configuration."""
        with self._lock:
            if self._process_pool is None:
                return
            self._process_pool.shutdown(wait=True, cancel_futures=True)
            self._process_pool = None
            atexit.unregister(self.shutdown)

    def __enter__(self) -> "Parallel":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def _process_pool_instance(self) -> concurrent.futures.ProcessPoolExecutor:
        with self._lock:
            if self._process_pool is None:
                self._process_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._config.jobs,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(logging.getLogger().getEffectiveLevel(),),
                )
                # Registered per instance and only while a pool is live, so
                # idle instances are not kept alive by the atexit registry
                atexit.register(self.shutdown)
            return self._process_pool

    def thread_map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply fn to every item in threads, returning results in input
        order. Only worthwhile when fn spends its time outside the GIL."""
        if self.config.jobs == 1:
            return [fn(item) for item in items]

        # ThreadPoolExecutor spawns threads lazily, one per submitted task
        # up to max_workers, so a large budget with few items is harmless.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.jobs
        ) as executor:
            return list(executor.map(fn, items))

    def process_map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply fn to every item in worker processes, returning results in
        input order.

        fn must be defined at the top level of an importable module; items
        and results must be picklable. The first worker exception propagates
        to the caller unchanged.
        """
        if self.config.jobs == 1:
            return [fn(item) for item in items]

        pool = self._process_pool_instance()
        results = []
        for result, durations in pool.map(functools.partial(_timed_call, fn), items):
            context.merge_durations(durations)
            results.append(result)
        return results


# The module-default instance used by application code.
_default = Parallel()


def config() -> Config:
    return _default.config


def configure(jobs: int) -> None:
    """Replace the module-default instance with one using the given budget.

    Raises RuntimeError if the current default has a live pool; parallelism
    must be configured before any parallel work runs (or after shutdown()).
    """
    global _default
    if _default.pool_is_live:
        raise RuntimeError(
            "Cannot reconfigure parallelism while the process pool is "
            "running; call parallel.shutdown() first"
        )
    _default = Parallel(Config(jobs=jobs))


def thread_map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    return _default.thread_map(fn, items)


def process_map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    return _default.process_map(fn, items)


def shutdown() -> None:
    _default.shutdown()
