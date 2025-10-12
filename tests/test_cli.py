"""Simple, lightweight tests for CLI utility functions."""

import pytest
import warnings
import unittest.mock
from padne import cli


class TestCollectWarnings:

    # This is just to suppress pytest being annoying
    @unittest.mock.patch("warnings.showwarning")
    def test_collects_warnings(self, _):
        with cli.collect_warnings() as warns:
            assert len(warns) == 0
            warnings.warn("First warning", UserWarning)
            assert len(warns) == 1
            warnings.warn("Second warning", DeprecationWarning)
            assert len(warns) == 2
            warnings.warn("Third warning", RuntimeWarning)

        assert len(warns) == 3
        assert warns[0].category == UserWarning
        assert warns[1].category == DeprecationWarning
        assert warns[2].category == RuntimeWarning

    @unittest.mock.patch("warnings.showwarning")
    def test_shows_warnings(self, mock_showwarning):
        with cli.collect_warnings():
            assert mock_showwarning.call_count == 0
            warnings.warn("First warning", UserWarning)
            assert mock_showwarning.call_count == 1
            args = mock_showwarning.call_args[0]
            assert str(args[0]) == "First warning"
            assert args[1] == UserWarning
            warnings.warn("Second warning", DeprecationWarning)
            assert mock_showwarning.call_count == 2
            args = mock_showwarning.call_args[0]
            assert str(args[0]) == "Second warning"
            assert args[1] == DeprecationWarning

    @unittest.mock.patch("warnings.showwarning")
    def test_context_manager_restores_state(self, _):
        with cli.collect_warnings() as warns:
            warnings.warn("A warning", UserWarning)

        assert len(warns) == 1
        warnings.warn("Another warning", UserWarning)
        assert len(warns) == 1  # Should not have changed
