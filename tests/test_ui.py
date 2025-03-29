import pytest
from padne.ui import pretty_format_si_number

class TestSIPrettyFormat:
    
    def test_zero_value(self):
        """Test that zero is formatted correctly."""
        assert pretty_format_si_number(0, "V") == "0 V"
        assert pretty_format_si_number(0.0, "A") == "0 A"
        
    def test_very_small_values(self):
        """Test that very small values are treated as zero."""
        assert pretty_format_si_number(1e-15, "W") == "0 W"
        assert pretty_format_si_number(-1e-15, "Ω") == "0 Ω"
        
    def test_small_values(self):
        """Test values requiring smaller SI prefixes."""
        assert pretty_format_si_number(0.000000001, "A") == "1 nA"
        assert pretty_format_si_number(0.000001, "V") == "1 μV"
        assert pretty_format_si_number(0.001, "W") == "1 mW"
        assert pretty_format_si_number(0.0012, "A") == "1.2 mA"
        assert pretty_format_si_number(0.0000000567, "V") == "56.7 nV"
        
    def test_regular_values(self):
        """Test values without SI prefixes."""
        assert pretty_format_si_number(1, "m") == "1 m"
        assert pretty_format_si_number(2.5, "s") == "2.5 s"
        assert pretty_format_si_number(9.99, "Hz") == "9.99 Hz"
        assert pretty_format_si_number(10, "kg") == "10 kg"
        assert pretty_format_si_number(99.9, "°C") == "99.9 °C"
        assert pretty_format_si_number(100, "Pa") == "100 Pa"
        assert pretty_format_si_number(999, "m") == "999 m"
        
    def test_large_values(self):
        """Test values requiring larger SI prefixes."""
        assert pretty_format_si_number(1000, "Hz") == "1 kHz"
        assert pretty_format_si_number(1234, "V") == "1.234 kV"
        assert pretty_format_si_number(1000000, "W") == "1 MW"
        assert pretty_format_si_number(1200000, "A") == "1.2 MA"
        assert pretty_format_si_number(1000000000, "Pa") == "1 GPa"
        assert pretty_format_si_number(1200000000000, "Hz") == "1.2 THz"
        
    def test_negative_values(self):
        """Test negative values."""
        assert pretty_format_si_number(-1000, "V") == "-1 kV"
        assert pretty_format_si_number(-0.001, "A") == "-1 mA"
        assert pretty_format_si_number(-10, "°C") == "-10 °C"
        
    def test_different_units(self):
        """Test with various unit symbols."""
        assert pretty_format_si_number(1500, "V") == "1.5 kV"
        assert pretty_format_si_number(1500, "W") == "1.5 kW"
        assert pretty_format_si_number(1500, "Ω") == "1.5 kΩ"
        assert pretty_format_si_number(1500, "") == "1.5 k"  # No unit
        
    def test_precision_handling(self):
        """Test precision rules for different magnitudes."""
        # >100: 1 decimal place
        assert pretty_format_si_number(123.456, "V") == "123.5 V"
        # 10-100: 2 decimal places
        assert pretty_format_si_number(12.345, "V") == "12.35 V"
        
    def test_trailing_zeros_removal(self):
        """Test that trailing zeros are removed."""
        assert pretty_format_si_number(1.000, "V") == "1 V"
        assert pretty_format_si_number(1.200, "V") == "1.2 V"
        assert pretty_format_si_number(1.230, "V") == "1.23 V"
        
    def test_boundary_cases(self):
        """Test values at the boundaries between SI prefixes."""
        assert pretty_format_si_number(999.9, "V") == "999.9 V"
        assert pretty_format_si_number(1000, "V") == "1 kV"
        assert pretty_format_si_number(0.001, "V") == "1 mV"
        assert pretty_format_si_number(0.0009999, "V") == "999.9 μV"
