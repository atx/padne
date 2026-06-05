"""
Process-wide context for cross-cutting concerns. Currently hosts the
stage-timing API used to measure how long named stages take across the
codebase.

Usage:

    with context.timing_session() as session:
        with context.stage_timer("connectivity"):
            ...                       # -> "padne.solver.connectivity"

        @context.stage_timer
        def assemble_system(...): ... # -> "padne.solver.assemble_system"

        @context.stage_timer("custom_name")
        def f(...): ...               # -> "<f's module>.custom_name"

    with session.durations as d:
        ...  # dict[str, Session.Duration]

The module-level stage_timer is a no-op when called outside an active
timing_session.

`Session` exposes the same polymorphic `stage_timer` API directly, so
once you have a session reference you can also do:

    @session.stage_timer            # key = "<func module>.<qualname>"
    def f(): ...

    with session.stage_timer("padne.solver.something"):
        ...

When invoked on a Session directly the string is the *full* key with no
caller-module prefix. The module-level wrapper is what adds the prefix.
"""

import contextlib
import inspect
import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


class LockedObject(Generic[T]):
    """Wraps a value so it cannot be accessed without holding the lock.

        owned = LockedObject([])
        with owned as items:
            items.append(...)
    """

    def __init__(self, obj: T) -> None:
        self._obj = obj
        self._lock = threading.Lock()

    def __enter__(self) -> T:
        self._lock.acquire()
        return self._obj

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


class Session:
    """A single bag of stage timings shared across all threads that run
    inside one timing_session() block."""

    @dataclass
    class Duration:
        perf_time: float = 0.0
        process_time: float = 0.0
        call_count: int = 0

    def __init__(self):
        self.durations = LockedObject({})

    def stage_timer(self, arg):
        """Polymorphic:
            with session.stage_timer("full.key"): ...    # CM
            @session.stage_timer                          # bare decorator
            @session.stage_timer("full.key")              # parameterized decorator
        """
        if callable(arg):
            func = arg
            key = f"{func.__module__}.{func.__qualname__}"
            return self._cm(key)(func)
        return self._cm(arg)

    @contextlib.contextmanager
    def _cm(self, key: str):
        start_perf = time.perf_counter()
        start_process = time.process_time()
        try:
            yield
        finally:
            # Record even on exception so failing stages still show up.
            perf = time.perf_counter() - start_perf
            proc = time.process_time() - start_process
            with self.durations as durations:
                existing = durations.get(key)
                if existing is None:
                    durations[key] = Session.Duration(perf, proc, 1)
                else:
                    existing.perf_time += perf
                    existing.process_time += proc
                    existing.call_count += 1


class _NullSession:
    """Stand-in returned when no timing_session is active. Mirrors the
    polymorphic Session.stage_timer API but does no work."""

    def stage_timer(self, arg):
        if callable(arg):
            return arg
        return self._cm()

    @contextlib.contextmanager
    def _cm(self):
        yield


_null_session = _NullSession()
_active: Session | None = None


@contextlib.contextmanager
def timing_session():
    """Open a timing session. While active, module-level stage_timer
    entries record into it.

    TODO: support multiple concurrent independent sessions. Right now we
    only allow one active session per process and raise if a second is
    opened. The check-and-set below races for that case; it's only safe
    today because the single-session restriction means nobody opens two
    in parallel.
    """
    global _active
    if _active is not None:
        raise RuntimeError(
            "A timing session is already active; "
            "concurrent sessions are not yet supported"
        )
    _active = Session()
    try:
        yield _active
    finally:
        _active = None


def _caller_module_name() -> str:
    # Call chain: <user> -> stage_timer() -> _caller_module_name()
    frame = inspect.currentframe()
    caller = frame.f_back.f_back
    return caller.f_globals.get("__name__", "<unknown>")


def stage_timer(arg):
    """Module-level wrapper that adds the caller-module prefix and defers
    session lookup to call/enter time.

        with stage_timer("name"):     # key = "<caller module>.name"
            ...

        @stage_timer                  # key = "<func module>.<func qualname>"
        def f(): ...

        @stage_timer("name")          # key = "<caller module>.name"
        def g(): ...
    """

    @contextlib.contextmanager
    def _deferred(key: str):
        global _active
        # Session lookup deferred to enter-time so decorators bound at import
        # time still resolve to whatever session is active at *call* time.
        session = _active if _active is not None else _null_session
        with session.stage_timer(key):
            yield

    if callable(arg):
        func = arg
        key = f"{func.__module__}.{func.__qualname__}"
        return _deferred(key)(func)

    name = arg
    module = _caller_module_name()
    return _deferred(f"{module}.{name}")
