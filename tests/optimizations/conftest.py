"""This file contains fixtures for use in pytest unit tests for subgradient descent methods."""

import pytest


# Sample functions for testing:
def f_square(x: float) -> float:
    """Sample squared function.

    Args:
        x (float): the number to square.

    Returns:
        float: The sqaure of x.
    """
    return x**2


def f_abs(x: float) -> float:
    """Sample abs function.

    Args:
        x (float): the number to take abs.

    Returns:
        float: The abs of x.
    """
    return abs(x)


def f_cube(x: float) -> float:
    """Sample cubed function.

    Args:
        x: the number to cube.

    Returns:
        float: The cube of x.
    """
    return x**3


def neg_f_square(x: float) -> float:
    """Sample negative squared function.

    Args:
        x (float): the number to negative square.

    Returns:
        float: The negative sqaure of x.
    """
    return -(x**2)


@pytest.fixture
def funcs():
    """Sample fixture of functions for subgradient descent tests."""
    return [f_square, f_abs, f_cube]
