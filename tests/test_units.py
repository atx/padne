

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


class TestSIPrettyFormat:

    def test_zero_value(self):
        """Test that zero is formatted correctly."""
        assert units.Value(0, "V").pretty_format() == "0 V"
        assert units.Value(0.0, "A").pretty_format() == "0 A"

    def test_very_small_values(self):
        """Test that very small values are treated as zero."""
        assert units.Value(1e-15, "W").pretty_format() == "0 W"
        assert units.Value(-1e-15, "Ω").pretty_format() == "0 Ω"

    def test_small_values(self):
        """Test values requiring smaller SI prefixes."""
        assert units.Value(0.000000001, "A").pretty_format() == "1 nA"
        assert units.Value(0.000001, "V").pretty_format() == "1 μV"
        assert units.Value(0.001, "W").pretty_format() == "1 mW"
        assert units.Value(0.0012, "A").pretty_format() == "1.2 mA"
        assert units.Value(0.0000000567, "V").pretty_format() == "56.7 nV"

    def test_regular_values(self):
        """Test values without SI prefixes."""
        assert units.Value(1, "m").pretty_format() == "1 m"
        assert units.Value(2.5, "s").pretty_format() == "2.5 s"
        assert units.Value(9.99, "Hz").pretty_format() == "9.99 Hz"
        assert units.Value(10, "kg").pretty_format() == "10 kg"
        assert units.Value(99.9, "°C").pretty_format() == "99.9 °C"
        assert units.Value(100, "Pa").pretty_format() == "100 Pa"
        assert units.Value(999, "m").pretty_format() == "999 m"

    def test_large_values(self):
        """Test values requiring larger SI prefixes."""
        assert units.Value(1000, "Hz").pretty_format() == "1 kHz"
        assert units.Value(1234, "V").pretty_format() == "1.234 kV"
        assert units.Value(1000000, "W").pretty_format() == "1 MW"
        assert units.Value(1200000, "A").pretty_format() == "1.2 MA"
        assert units.Value(1000000000, "Pa").pretty_format() == "1 GPa"
        assert units.Value(1200000000000, "Hz").pretty_format() == "1.2 THz"

    def test_negative_values(self):
        """Test negative values."""
        assert units.Value(-1000, "V").pretty_format() == "-1 kV"
        assert units.Value(-0.001, "A").pretty_format() == "-1 mA"
        assert units.Value(-10, "°C").pretty_format() == "-10 °C"

    def test_different_units(self):
        """Test with various unit symbols."""
        assert units.Value(1500, "V").pretty_format() == "1.5 kV"
        assert units.Value(1500, "W").pretty_format() == "1.5 kW"
        assert units.Value(1500, "Ω").pretty_format() == "1.5 kΩ"
        assert units.Value(1500, "").pretty_format() == "1.5 k"  # No unit

    def test_precision_handling(self):
        """Test precision rules for different magnitudes."""
        # >100: 1 decimal place
        assert units.Value(123.456, "V").pretty_format() == "123.5 V"
        # 10-100: 2 decimal places
        assert units.Value(12.345, "V").pretty_format() == "12.35 V"

    def test_trailing_zeros_removal(self):
        """Test that trailing zeros are removed."""
        assert units.Value(1.000, "V").pretty_format() == "1 V"
        assert units.Value(1.200, "V").pretty_format() == "1.2 V"
        assert units.Value(1.230, "V").pretty_format() == "1.23 V"

    def test_boundary_cases(self):
        """Test values at the boundaries between SI prefixes."""
        assert units.Value(999.9, "V").pretty_format() == "999.9 V"
        assert units.Value(1000, "V").pretty_format() == "1 kV"
        assert units.Value(0.001, "V").pretty_format() == "1 mV"
        assert units.Value(0.0009999, "V").pretty_format() == "999.9 μV"


class TestPrettyFormatDecimalPlaces:
    """Test the decimal_places parameter in pretty_format."""

    def test_specified_decimal_places(self):
        """Test formatting with specified decimal places."""
        value = units.Value(23.97654, "V")
        
        # Test various decimal place counts
        assert value.pretty_format(0) == "24 V"
        assert value.pretty_format(1) == "24.0 V"
        assert value.pretty_format(2) == "23.98 V"
        assert value.pretty_format(3) == "23.977 V"
        assert value.pretty_format(5) == "23.97654 V"

    def test_decimal_places_with_si_prefixes(self):
        """Test that decimal places work correctly with SI prefixes."""
        # Small value that gets micro prefix
        small_value = units.Value(0.00002397, "V")  # 23.97 μV
        assert small_value.pretty_format(1) == "24.0 μV"
        assert small_value.pretty_format(2) == "23.97 μV"
        assert small_value.pretty_format(4) == "23.9700 μV"
        
        # Large value that gets kilo prefix
        large_value = units.Value(2397.654, "V")  # 2.397654 kV
        assert large_value.pretty_format(1) == "2.4 kV"
        assert large_value.pretty_format(3) == "2.398 kV"
        assert large_value.pretty_format(6) == "2.397654 kV"

    def test_decimal_places_with_negative_values(self):
        """Test decimal places with negative values."""
        value = units.Value(-23.97654, "V")
        assert value.pretty_format(2) == "-23.98 V"
        assert value.pretty_format(4) == "-23.9765 V"

    def test_decimal_places_preserves_trailing_zeros(self):
        """Test that specified decimal places preserves trailing zeros."""
        value = units.Value(24.0, "V")
        
        # With smart formatting, trailing zeros are removed
        assert value.pretty_format() == "24 V"
        
        # With specified decimal places, trailing zeros are preserved
        assert value.pretty_format(0) == "24 V"
        assert value.pretty_format(1) == "24.0 V"
        assert value.pretty_format(3) == "24.000 V"

    def test_decimal_places_zero_value(self):
        """Test that zero is handled correctly with decimal places."""
        value = units.Value(0.0, "V")
        
        # Zero should always be formatted as "0 unit" regardless of decimal places
        assert value.pretty_format(0) == "0 V"
        assert value.pretty_format(3) == "0 V"
        assert value.pretty_format(5) == "0 V"

    def test_decimal_places_very_small_values(self):
        """Test decimal places with very small values treated as zero."""
        value = units.Value(1e-15, "V")
        
        # Very small values should be treated as zero regardless of decimal places
        assert value.pretty_format(0) == "0 V"
        assert value.pretty_format(5) == "0 V"

    def test_smart_vs_specified_precision(self):
        """Test that smart precision and specified precision give different results."""
        # Value that would normally be formatted with 2 decimals in smart mode
        value = units.Value(12.345678, "V")
        
        # Smart precision (should give 2 decimals for values 10-100)
        assert value.pretty_format() == "12.35 V"
        
        # Specified precision
        assert value.pretty_format(1) == "12.3 V"
        assert value.pretty_format(4) == "12.3457 V"
        assert value.pretty_format(6) == "12.345678 V"

    def test_decimal_places_boundary_values(self):
        """Test decimal places at SI prefix boundaries."""
        # Value right at 1000 boundary
        boundary_value = units.Value(999.999, "V")
        assert boundary_value.pretty_format(0) == "1000 V"  # Rounds up to 1000
        assert boundary_value.pretty_format(1) == "1000.0 V"
        assert boundary_value.pretty_format(3) == "999.999 V"
        
        # Value just over 1000 (gets kilo prefix)
        kilo_value = units.Value(1000.001, "V")
        assert kilo_value.pretty_format(0) == "1 kV"
        assert kilo_value.pretty_format(3) == "1.000 kV"
        assert kilo_value.pretty_format(6) == "1.000001 kV"
