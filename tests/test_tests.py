"""Tests to validate typeguard integration."""

import sys

import pytest
from typeguard import TypeCheckError
from typeguard._importhook import TypeguardLoader

import padne.kicad  # noqa: F401
from padne import parallel
import padne.tests as t


class TestTypeguard:
    """Test that typeguard correctly catches type violations."""

    def test_incorrect_argument(self):
        # Check that correct usage passes. This also increases coverage :)
        assert t.add_numbers(1, 2) == 3
        with pytest.raises(TypeCheckError):
            t.add_numbers("x", "y")

    def test_incorrect_return(self):
        with pytest.raises(TypeCheckError):
            t.wrong_return_type()

    def test_process_map_is_checked(self):
        # Spawned workers do not inherit the parent's typeguard import hook
        with parallel.Parallel(parallel.Config(jobs=2)) as p:
            with pytest.raises(TypeCheckError):
                p.process_map(t.wrong_return_type_of_arg, [1])

    def test_all_padne_modules_instrumented(self):
        # Modules imported before typeguard installs its import hook (e.g.
        # by conftest.py) silently escape instrumentation.
        uninstrumented = sorted(
            name
            for name, module in sys.modules.items()
            if (name == "padne" or name.startswith("padne."))
            and module.__spec__.origin.endswith(".py")
            and not isinstance(module.__spec__.loader, TypeguardLoader)
        )
        assert uninstrumented == []
