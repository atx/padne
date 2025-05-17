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

    def pretty_format(self) -> str:
        """Pretty format the stored value with SI prefix and unit.

        Uses self.value and self.unit.

        Returns:
            A formatted string with the value, appropriate SI prefix, and unit

        Examples:
            >>> Value(0.000001, "A").pretty_format()
            '1.000 μA'
            >>> Value(1500, "V").pretty_format()
            '1.500 kV'
        """
        if self.value == 0:
            return f"0 {self.unit}"

        # Define SI prefixes and their corresponding powers of 10
        prefixes = {
            -12: "p",  # pico
            -9: "n",   # nano
            -6: "μ",   # micro
            -3: "m",   # milli
            0: "",     # base unit
            3: "k",    # kilo
            6: "M",    # mega
            9: "G",    # giga
            12: "T"    # tera
        }

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
        # Use fewer decimal places for larger numbers
        if abs_value >= 100:
            formatted_value = f"{abs_value:.1f}"
        elif abs_value >= 10:
            formatted_value = f"{abs_value:.2f}"
        else:
            formatted_value = f"{abs_value:.3f}"

        # Remove trailing zeros after decimal point
        if "." in formatted_value:
            formatted_value = formatted_value.rstrip("0").rstrip(".")

        # Apply the sign from the original value
        if self.value < 0:
            formatted_value = "-" + formatted_value

        # Return the formatted string with prefix and unit
        return f"{formatted_value} {prefixes[exponent]}{self.unit}"
