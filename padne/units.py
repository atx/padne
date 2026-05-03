import re
from dataclasses import dataclass


# Single source of truth for SI prefixes: (display symbol, decimal exponent).
# The empty-prefix entry is only used for formatting (the base unit case).
_SI_PREFIXES: tuple[tuple[str, int], ...] = (
    ("T", 12),
    ("G", 9),
    ("M", 6),
    ("k", 3),
    ("", 0),
    ("m", -3),
    ("μ", -6),
    ("n", -9),
    ("p", -12),
)

# Alternative spellings accepted on input. "u" is the ASCII fallback for "μ".
_PARSE_PREFIX_ALIASES = {"u": "μ"}

_PREFIX_TO_MULTIPLIER: dict[str, float] = {
    prefix: 10.0 ** exponent for prefix, exponent in _SI_PREFIXES if prefix
}
_PREFIX_TO_MULTIPLIER.update({
    alias: _PREFIX_TO_MULTIPLIER[canonical]
    for alias, canonical in _PARSE_PREFIX_ALIASES.items()
})

_EXPONENT_TO_PREFIX: dict[int, str] = {
    exponent: prefix for prefix, exponent in _SI_PREFIXES
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
        if last_character in _PREFIX_TO_MULTIPLIER:
            s = s[:-1]
            multiplier = _PREFIX_TO_MULTIPLIER[last_character]

        # Great, the rest is just a float
        value = float(s) * multiplier

        return cls(value=value, unit=unit)

    def pretty_format(self, decimal_places: int | None = None) -> str:
        """Pretty format the stored value with SI prefix and unit.

        Uses self.value and self.unit.

        Args:
            decimal_places: Number of decimal places to show. If None, uses smart precision
                          based on magnitude (1-3 decimal places).

        Returns:
            A formatted string with the value, appropriate SI prefix, and unit

        Examples:
            >>> Value(0.000001, "A").pretty_format()
            '1.000 μA'
            >>> Value(1500, "V").pretty_format()
            '1.500 kV'
            >>> Value(23.97, "V").pretty_format(3)
            '23.970 V'
            >>> Value(23.97, "V").pretty_format(5)
            '23.97000 V'
        """
        if self.value == 0:
            return f"0 {self.unit}"

        # Determine the appropriate prefix for the value
        abs_value = abs(self.value)
        exponent = 0

        if abs_value < 1e-10:
            return f"0 {self.unit}"  # Treat very small values as zero

        if abs_value >= 1:
            while abs_value >= 1000 and exponent < 12:
                abs_value /= 1000
                exponent += 3
        else:
            while abs_value < 1 and exponent > -12:
                abs_value *= 1000
                exponent -= 3

        # Format the value with the appropriate precision
        if decimal_places is not None:
            # Use specified decimal places
            formatted_value = f"{abs_value:.{decimal_places}f}"
        else:
            # Use smart precision based on magnitude
            if abs_value >= 100:
                formatted_value = f"{abs_value:.1f}"
            elif abs_value >= 10:
                formatted_value = f"{abs_value:.2f}"
            else:
                formatted_value = f"{abs_value:.3f}"

        # Remove trailing zeros after decimal point (only for smart precision)
        if decimal_places is None and "." in formatted_value:
            formatted_value = formatted_value.rstrip("0").rstrip(".")

        # Apply the sign from the original value
        if self.value < 0:
            formatted_value = "-" + formatted_value

        # Return the formatted string with prefix and unit
        return f"{formatted_value} {_EXPONENT_TO_PREFIX[exponent]}{self.unit}"
