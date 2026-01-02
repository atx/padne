"""Tests to validate typeguard integration."""

import pytest
from typeguard import TypeCheckError

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
