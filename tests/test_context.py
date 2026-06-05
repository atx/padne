import threading
import time
import pytest

from padne import context


class TestTimingSession:

    def test_durations_empty_when_nothing_recorded(self):
        with context.timing_session() as s:
            pass
        with s.durations as d:
            assert d == {}

    def test_concurrent_open_raises(self):
        # Two overlapping timing_session blocks are not supported (yet).
        # The inner attempt must raise; the outer must still close cleanly,
        # and a new session must be openable afterwards.
        with context.timing_session():
            with pytest.raises(RuntimeError):
                with context.timing_session():
                    pass
        with context.timing_session():
            pass

    def test_sessions_are_isolated(self):
        with context.timing_session() as s1:
            with context.stage_timer("only_in_first"):
                pass
        with context.timing_session() as s2:
            with context.stage_timer("only_in_second"):
                pass
        with s1.durations as d1:
            assert set(d1) == {"test_context.only_in_first"}
        with s2.durations as d2:
            assert set(d2) == {"test_context.only_in_second"}


class TestStageTimerWithForm:

    def test_records_under_caller_module_dot_name(self):
        with context.timing_session() as s:
            with context.stage_timer("phase_a"):
                pass
        # Frame inspection should pick up this test module's __name__.
        with s.durations as d:
            assert "test_context.phase_a" in d
            assert d["test_context.phase_a"].call_count == 1

    def test_no_op_outside_session(self):
        # Must not raise and must not leave dangling state behind.
        with context.stage_timer("outside"):
            pass
        # Opening a fresh session afterwards should be clean.
        with context.timing_session() as s:
            pass
        with s.durations as d:
            assert d == {}

    def test_perf_time_reflects_wall_clock(self):
        sleep_duration = 0.05
        with context.timing_session() as s:
            with context.stage_timer("sleepy"):
                time.sleep(sleep_duration)
        with s.durations as d:
            entry = d["test_context.sleepy"]
        assert entry.perf_time >= sleep_duration
        assert entry.process_time < entry.perf_time  # sleep -> wall > cpu

    def test_records_on_exception(self):
        # Even if the timed block raises, the partial duration must land
        # in the durations dict so failing stages are still visible.
        with context.timing_session() as s:
            with pytest.raises(ValueError):
                with context.stage_timer("boom"):
                    raise ValueError("nope")
        with s.durations as d:
            assert "test_context.boom" in d
            assert d["test_context.boom"].call_count == 1


class TestStageTimerDecorator:

    def test_bare_decorator_uses_qualname(self):
        @context.stage_timer
        def my_decorated_func():
            return 42

        with context.timing_session() as s:
            assert my_decorated_func() == 42
        # qualname for a function defined inside a method is nested
        expected = (
            f"test_context."
            f"{TestStageTimerDecorator.test_bare_decorator_uses_qualname.__qualname__}"
            f".<locals>.my_decorated_func"
        )
        with s.durations as d:
            assert expected in d

    def test_parameterized_decorator_uses_explicit_name(self):
        @context.stage_timer("explicit")
        def f():
            pass

        with context.timing_session() as s:
            f()
        with s.durations as d:
            assert "test_context.explicit" in d

    def test_aggregation_across_many_calls(self):
        # Hot-path scenario: decorator on a function called many times in
        # one session. Total/count must aggregate, not overwrite.
        @context.stage_timer("hot")
        def hot():
            pass

        n = 50
        with context.timing_session() as s:
            for _ in range(n):
                hot()
        with s.durations as d:
            entry = d["test_context.hot"]
        assert entry.call_count == n
        assert entry.perf_time >= 0.0

    def test_decorator_session_lookup_is_deferred(self):
        # Decorator bound *before* a session opens must still record into
        # the session that's active when the function is *called*.
        @context.stage_timer
        def bound_outside_session():
            pass

        with context.timing_session() as s:
            bound_outside_session()

        expected = (
            f"test_context."
            f"{TestStageTimerDecorator.test_decorator_session_lookup_is_deferred.__qualname__}"
            f".<locals>.bound_outside_session"
        )
        with s.durations as d:
            assert expected in d

    def test_decorator_no_op_outside_session(self):
        @context.stage_timer
        def f():
            return "value"

        # Should still call through and return the value, just not record.
        assert f() == "value"


class TestDirectSessionAPI:
    # Session.stage_timer is the canonical entry point; the module-level
    # function is a thin prefix-adding wrapper. These tests pin down that
    # session-level usage works on its own and that the string passed to a
    # Session is treated as the *full* key (no caller-module prefix).

    def test_with_form_uses_string_as_full_key(self):
        with context.timing_session() as s:
            with s.stage_timer("padne.solver.bespoke"):
                pass
        with s.durations as d:
            assert "padne.solver.bespoke" in d
            assert "test_context.padne.solver.bespoke" not in d

    def test_bare_decorator_on_session(self):
        with context.timing_session() as s:
            @s.stage_timer
            def my_fn():
                return 7
            assert my_fn() == 7
        expected = (
            f"test_context."
            f"{TestDirectSessionAPI.test_bare_decorator_on_session.__qualname__}"
            f".<locals>.my_fn"
        )
        with s.durations as d:
            assert expected in d

    def test_parameterized_decorator_on_session(self):
        with context.timing_session() as s:
            @s.stage_timer("explicit.key")
            def f():
                pass
            f()
        with s.durations as d:
            assert "explicit.key" in d


class TestSessionDurationNested:
    # Duration moved to Session.Duration. Verify the stored values are
    # instances of that nested class so external consumers can reference
    # the type via Session.Duration without surprises.

    def test_recorded_value_is_session_duration(self):
        with context.timing_session() as s:
            with context.stage_timer("typed"):
                pass
        with s.durations as d:
            entry = d["test_context.typed"]
        assert isinstance(entry, context.Session.Duration)


class TestConcurrentMerging:

    def test_threads_recording_same_key_aggregate_safely(self):
        # N threads each call the same decorated function once concurrently.
        # Without the lock on the durations dict, the read-modify-write on
        # Duration could lose updates and final call_count would be < N.
        @context.stage_timer("contended")
        def work():
            # Burn a tiny bit of time to widen the race window.
            time.sleep(0.001)

        n_threads = 32
        barrier = threading.Barrier(n_threads)

        def runner():
            barrier.wait()  # release all threads as close to simultaneously as possible
            work()

        with context.timing_session() as s:
            threads = [threading.Thread(target=runner) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        with s.durations as d:
            entry = d["test_context.contended"]
        assert entry.call_count == n_threads

    def test_workers_see_active_session(self):
        # A worker thread spawned inside a timing_session must observe the
        # session as active (since the session is process-wide). Verify a
        # plain threading.Thread can record into the parent's session with
        # no explicit context propagation.
        recorded_in_thread = threading.Event()

        def worker():
            with context.stage_timer("from_worker"):
                pass
            recorded_in_thread.set()

        with context.timing_session() as s:
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        assert recorded_in_thread.is_set()
        with s.durations as d:
            assert "test_context.from_worker" in d


class TestLockedObject:

    def test_yields_underlying_object(self):
        inner = {"a": 1}
        wrapped = context.LockedObject(inner)
        with wrapped as obj:
            assert obj is inner

    def test_lock_is_actually_held(self):
        # Acquiring the LockedObject from a second thread must block until
        # the first one releases. This is the whole point of the abstraction.
        wrapped = context.LockedObject([])
        first_inside = threading.Event()
        second_attempted = threading.Event()
        second_acquired = threading.Event()

        def first():
            with wrapped:
                first_inside.set()
                # Wait until the second thread has tried (and is blocked)
                second_attempted.wait()
                # Brief pause to give second thread a chance to race in
                time.sleep(0.05)
                assert not second_acquired.is_set()

        def second():
            first_inside.wait()
            second_attempted.set()
            with wrapped:
                second_acquired.set()

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert second_acquired.is_set()  # eventually got it after t1 released

    def test_released_on_exception(self):
        wrapped = context.LockedObject([])
        with pytest.raises(RuntimeError):
            with wrapped:
                raise RuntimeError("inside")
        # If the lock leaked, this re-acquire would deadlock; the test
        # will time out under pytest's default timeout config.
        with wrapped as obj:
            assert obj == []
