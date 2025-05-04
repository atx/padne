

import pytest

from padne import units


class TestParsing:
    @pytest.mark.parametrize(
        "input_str, expected_value, expected_unit",
        [
            # Basic integers and floats
            ("100", 100.0, ""),
            ("0.5", 0.5, ""),
            ("-25", -25.0, ""),
            ("-3.14", -3.14, ""),
            # Scientific notation
            ("1e3", 1000.0, ""),
            ("1.5e-2", 0.015, ""),
            ("-2E6", -2000000.0, ""),
            # Units without prefixes
            ("10A", 10.0, "A"),
            ("2.5V", 2.5, "V"),
            ("1R", 1.0, "R"),
            # Units with prefixes
            ("100mA", 0.1, "A"),
            ("50uV", 0.00005, "V"),
            ("2kR", 2000.0, "R"),
            ("1MV", 1000000.0, "V"),
            ("10nA", 1e-8, "A"),
            ("3pA", 3e-12, "A"),
            ("5GA", 5e9, "A"),
            ("2TA", 2e12, "A"),
            # Spaces
            ("100 mA", 0.1, "A"),
            (" 50 uV ", 0.00005, "V"),
            ("1e3 A", 1000.0, "A"),
            # Edge cases
            (".5", 0.5, ""),
            ("-.5", -0.5, ""),
            ("1.", 1.0, ""),
            ("-2.", -2.0, ""),
        ]
    )
    def test_valid_parsing(self, input_str, expected_value, expected_unit):
        """Test parsing of various valid value strings."""
        parsed = units.Value.parse(input_str)
        assert parsed.value == pytest.approx(expected_value)
        assert parsed.unit == expected_unit

    @pytest.mark.parametrize(
        "invalid_str",
        [
            "",           # Empty string
            "abc",        # Non-numeric
            "10x",        # Invalid unit/suffix
            "10 kkA",     # Double prefix
            "10Am",       # Unit before prefix
            "10e",        # Incomplete scientific notation
            "10e+",       # Incomplete scientific notation
            "10e-",       # Incomplete scientific notation
            "--10",       # Double negative sign
            "++10",       # Double positive sign
            "10..5",      # Double decimal point
            "10 mA V",    # Multiple units
            "10 ZV",      # Invalid prefix
        ]
    )
    def test_invalid_parsing(self, invalid_str):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError):
            units.Value.parse(invalid_str)
