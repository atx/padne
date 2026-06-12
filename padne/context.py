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
from dataclasses import dataclass, replace
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

    def snapshot(self) -> dict[str, "Session.Duration"]:
        """Return an independent copy of the recorded durations. Used to
        ship timings across process boundaries (see padne.parallel)."""
        with self.durations as d:
            return {k: replace(v) for k, v in d.items()}

    def merge(self, snapshot: dict[str, "Session.Duration"]) -> None:
        """Accumulate durations from another session's snapshot into this
        one. Times sum and call counts add, same as repeated stage entries."""
        with self.durations as d:
            for key, duration in snapshot.items():
                existing = d.get(key)
                if existing is None:
                    d[key] = replace(duration)
                else:
                    existing.perf_time += duration.perf_time
                    existing.process_time += duration.process_time
                    existing.call_count += duration.call_count

    def format_summary(self) -> str:
        """Return a tree-shaped table of recorded stages.

        Keys are split on '.' to form a hierarchy; siblings are sorted by
        descending perf time (a branch's sort key is the sum of its
        descendants' perf time, with leaf keys contributing their own
        recorded perf time)."""
        with self.durations as d:
            snapshot = {
                k: (v.call_count, v.perf_time, v.process_time)
                for k, v in d.items()
            }
        if not snapshot:
            return "(no stages recorded)"

        tree: dict = {}
        for key in snapshot:
            node = tree
            for part in key.split("."):
                node = node.setdefault(part, {})

        def subtree_perf(parts: list[str]) -> float:
            full = ".".join(parts)
            if full in snapshot:
                return snapshot[full][1]
            prefix = full + "."
            return sum(
                perf for k, (_, perf, _) in snapshot.items()
                if k.startswith(prefix)
            )

        rows: list[tuple[str, tuple[int, float, float] | None]] = []

        def walk(subtree: dict, parts: list[str], indent: str) -> None:
            items = sorted(
                subtree.items(),
                key=lambda kv: subtree_perf(parts + [kv[0]]),
                reverse=True,
            )
            for i, (name, children) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                full_key = ".".join(parts + [name])
                rows.append((indent + connector + name, snapshot.get(full_key)))
                if children:
                    walk(
                        children, parts + [name],
                        indent + ("    " if is_last else "│   "),
                    )

        roots = sorted(
            tree.items(),
            key=lambda kv: subtree_perf([kv[0]]),
            reverse=True,
        )
        for name, children in roots:
            rows.append((name, snapshot.get(name)))
            if children:
                walk(children, [name], "")

        label_width = max(len("stage"), max(len(label) for label, _ in rows))
        lines = [
            f"{'stage':<{label_width}}  {'perf':>11}  {'cpu':>11}  {'calls':>6}",
            f"{'-' * label_width}  {'-' * 11}  {'-' * 11}  {'-' * 6}",
        ]
        for label, data in rows:
            if data is None:
                lines.append(f"{label:<{label_width}}")
            else:
                calls, perf, cpu = data
                lines.append(
                    f"{label:<{label_width}}  "
                    f"{perf:>10.4f}s  {cpu:>10.4f}s  {calls:>6d}"
                )
        return "\n".join(lines)


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


def merge_durations(snapshot: dict[str, Session.Duration]) -> None:
    """Merge a durations snapshot into the active session. No-op when no
    session is active, mirroring the stage_timer behavior."""
    if _active is not None:
        _active.merge(snapshot)


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
