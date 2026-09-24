
# Test functions for validating typeguard actually works

def add_numbers(a: int, b: int) -> int:
    return a + b


def wrong_return_type() -> str:
    return 42


def wrong_return_type_of_arg(x: int) -> str:
    return x
