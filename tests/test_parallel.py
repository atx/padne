import os
import threading

import pytest

from padne import context, parallel


# Functions used by process_map must be defined at module top level so the
# spawn pickle protocol can import them in the worker processes.

def square(x):
    return x * x


def get_pid(_):
    return os.getpid()


class BoomError(Exception):
    pass


def raise_on_two(x):
    if x == 2:
        raise BoomError("x was two")
    return x


def record_stage(x):
    with context.stage_timer("worker_stage"):
        return x + 1


@pytest.fixture(autouse=True)
def fresh_default_instance(monkeypatch):
    """Point the module-level singleton at a pristine instance for each
    test so tests cannot leak configuration or warm pools into each other."""
    monkeypatch.setattr(parallel, "_default", parallel.Parallel())
    yield
    # configure() may have swapped the default, so shut down whatever the
    # current one is before monkeypatch restores the real default.
    parallel.shutdown()


class TestConfig:

    def test_default_jobs_is_available_cpu_count(self):
        assert parallel.Config().jobs == len(os.sched_getaffinity(0))

    def test_module_default_uses_default_config(self):
        assert parallel.config() == parallel.Config()

    def test_configure_sets_jobs(self):
        parallel.configure(jobs=3)
        assert parallel.config().jobs == 3

    def test_jobs_must_be_positive(self):
        with pytest.raises(ValueError):
            parallel.Config(jobs=0)

    def test_configure_while_pool_running_raises(self):
        parallel.configure(jobs=2)
        # Force pool creation
        parallel.process_map(square, [1, 2, 3])
        with pytest.raises(RuntimeError):
            parallel.configure(jobs=4)
        # After an explicit shutdown, reconfiguration must work again
        parallel.shutdown()
        parallel.configure(jobs=4)
        assert parallel.config().jobs == 4


class TestParallelInstance:

    def test_instances_are_independent(self):
        with parallel.Parallel(parallel.Config(jobs=2)) as p1:
            with parallel.Parallel(parallel.Config(jobs=3)) as p2:
                assert p1.config.jobs == 2
                assert p2.config.jobs == 3

    def test_context_manager_shuts_down_pool(self):
        with parallel.Parallel(parallel.Config(jobs=2)) as p:
            p.process_map(square, [1, 2])
            assert p.pool_is_live
        assert not p.pool_is_live


class TestThreadMap:

    def test_preserves_input_order(self):
        parallel.configure(jobs=4)
        items = list(range(20))
        assert parallel.thread_map(square, items) == [x * x for x in items]

    def test_serial_when_jobs_is_one(self):
        parallel.configure(jobs=1)
        idents = parallel.thread_map(lambda _: threading.get_ident(), [1, 2, 3])
        assert set(idents) == {threading.get_ident()}

    def test_actually_runs_concurrently(self):
        # All n workers must be inside fn simultaneously to pass the
        # barrier; serial execution would break it via the timeout.
        n = 4
        parallel.configure(jobs=n)
        barrier = threading.Barrier(n, timeout=10)

        def wait_at_barrier(x):
            barrier.wait()
            return x

        assert parallel.thread_map(wait_at_barrier, list(range(n))) == list(range(n))

    def test_propagates_exception(self):
        parallel.configure(jobs=4)
        with pytest.raises(BoomError):
            parallel.thread_map(raise_on_two, [1, 2, 3])

    def test_empty_items(self):
        assert parallel.thread_map(square, []) == []

    def test_stage_timers_record_into_active_session(self):
        # Threads share the process-wide session, no merging machinery needed
        parallel.configure(jobs=4)
        with context.timing_session() as s:
            parallel.thread_map(record_stage, list(range(8)))
        with s.durations as d:
            assert d["test_parallel.worker_stage"].call_count == 8


class TestProcessMap:

    def test_preserves_input_order(self):
        parallel.configure(jobs=4)
        items = list(range(10))
        assert parallel.process_map(square, items) == [x * x for x in items]

    def test_runs_in_worker_processes(self):
        parallel.configure(jobs=2)
        pids = parallel.process_map(get_pid, [1, 2, 3, 4])
        assert all(pid != os.getpid() for pid in pids)

    def test_serial_when_jobs_is_one(self):
        parallel.configure(jobs=1)
        pids = parallel.process_map(get_pid, [1, 2, 3])
        assert set(pids) == {os.getpid()}

    def test_propagates_exception(self):
        parallel.configure(jobs=2)
        with pytest.raises(BoomError):
            parallel.process_map(raise_on_two, [1, 2, 3])

    def test_pool_survives_a_failed_task(self):
        parallel.configure(jobs=2)
        with pytest.raises(BoomError):
            parallel.process_map(raise_on_two, [1, 2, 3])
        assert parallel.process_map(square, [2, 3, 4]) == [4, 9, 16]

    def test_empty_items(self):
        assert parallel.process_map(square, []) == []

    def test_worker_timings_merge_into_parent_session(self):
        parallel.configure(jobs=2)
        n = 6
        with context.timing_session() as s:
            results = parallel.process_map(record_stage, list(range(n)))
        assert results == [x + 1 for x in range(n)]
        with s.durations as d:
            assert d["test_parallel.worker_stage"].call_count == n

    def test_works_without_active_session(self):
        # Worker timings are silently dropped when the parent has no
        # session open; the map itself must be unaffected.
        parallel.configure(jobs=2)
        assert parallel.process_map(record_stage, [1, 2]) == [2, 3]
