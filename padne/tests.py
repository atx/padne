
# Test functions for validating typeguard actually works

def add_numbers(a: int, b: int) -> int:
    return a + b


def wrong_return_type() -> str:
    return 42
