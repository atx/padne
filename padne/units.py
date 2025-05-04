import re
from dataclasses import dataclass


# SI Prefixes and their multipliers
_SI_PREFIXES = {
    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3,
    'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12,
}

_KNOWN_UNITS = {
    "A", "V", "R"
}


@dataclass(frozen=True)
class Value:
    value: float
    unit: str

    @classmethod
    def parse(cls, s: str) -> "Value":
        """
        Parse a string containing a value with optional SI prefix and unit.
        
        Examples:
            "100mA" -> Value(value=0.1, unit="A")
            "0.1A" -> Value(value=0.1, unit="A")
            "1e4A" -> Value(value=10000.0, unit="A")
            "100 mA" -> Value(value=0.1, unit="A")
            "50uV" -> Value(value=0.00005, unit="V")
            "10" -> Value(value=10.0, unit="")
        
        Args:
            s: String to parse
            
        Returns:
            Value object with parsed value and unit
            
        Raises:
            ValueError: If the string cannot be parsed
        """
        if not s or not s.strip():
            raise ValueError(f"Empty value string: '{s}'")

        # First, drop all spaces
        s = s.replace(" ", "")

        # Next, attempt to parse the unit, if it is present
        last_character = s[-1]
        unit = ""
        if last_character in _KNOWN_UNITS:
            s = s[:-1]
            unit = last_character

        # Check for SI prefix
        last_character = s[-1]
        multiplier = 1.0
        if last_character in _SI_PREFIXES:
            s = s[:-1]
            multiplier = _SI_PREFIXES[last_character]

        # Great, the rest is just a float
        value = float(s) * multiplier

        return cls(value=value, unit=unit)
