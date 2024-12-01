"""This file contains tests for the subgradient descent functions."""

import numpy
import pytest

from kinetic_project.optimizations.subgradient_methods import (
    augmented_lagrangian,
    compute_subgradient,
    constraint_violation,
    numerical_subgradient,
    objective_function,
    run_multiple_initializations_parallel,
    run_single_initialization,
    subgradient_descent,
)

from .conftest import f_abs, f_cube, f_square


def test_objective_function_minimization(funcs):
    """Tests the objective_function function for minimization."""
    x = numpy.array([0.5, 0.25, 0.25])
    result = objective_function(x, funcs)
    expected = 0.5**2 + abs(0.25) + (0.25) ** 3
    assert numpy.isclose(
        result, expected
    ), f"Expected {expected}, got {result}."


def test_objective_function_maximization(funcs):
    """Tests the objective_function function with argmax=True."""
    x = numpy.array([0.5, 0.25, 0.25])
    result = objective_function(x, funcs, argmax=True)
    expected = -(0.5**2 + abs(0.25) + (0.25) ** 3)
    assert numpy.isclose(
        result, expected
    ), f"Expected {expected}, got {result}."


def test_constraint_violation_satisfied():
    """Tests the constraint_violation function for correct satisfaction."""
    x = numpy.array([0.5, 0.25, 0.25])
    assert numpy.isclose(
        constraint_violation(x), 0
    ), "Constraint should be satisfied."


def test_constraint_violation_unsatisfied():
    """Tests the constraint_violation function for incorrect satisfaction."""
    x = numpy.array([0.6, 0.4, 0.2])
    assert numpy.isclose(
        constraint_violation(x), 0.2
    ), "Constraint violation incorrect."


def test_augmented_lagrangian(funcs):
    """Tests for the augmented_lagrangian function."""
    x = numpy.array([0.5, 0.25, 0.25])
    lambd = 1.0
    rho = 10.0
    result = augmented_lagrangian(x, funcs, lambd, rho)
    c = constraint_violation(x)
    expected = objective_function(x, funcs) + lambd * c + 0.5 * rho * c**2
    assert numpy.isclose(
        result, expected
    ), f"Expected {expected}, got {result}."


def test_numerical_subgradient():
    """Tests for the numerical_subgradient function."""
    x = 0.5
    epsilon = 1e-6
    result = numerical_subgradient(f_square, x, epsilon)
    expected = (f_square(x + epsilon) - f_square(x)) / epsilon
    assert numpy.isclose(
        result, expected
    ), f"Expected {expected}, got {result}."


def test_compute_subgradient(funcs):
    """Tests for the compute_subgradient function."""
    x = numpy.array([0.5, -0.5, 0.25])
    result = compute_subgradient(x, funcs)
    expected = numpy.array(
        [
            numerical_subgradient(f_square, 0.5),
            numerical_subgradient(f_abs, -0.5),
            numerical_subgradient(f_cube, 0.25),
        ]
    )
    assert numpy.allclose(
        result, expected
    ), f"Expected {expected}, got {result}."


def test_subgradient_descent_convergence(funcs):
    """Tests for the subgradient_descent function."""
    x_0 = numpy.array([0.5, 0.25, 0.25])
    max_iter = 500
    tol = 1e-6
    result, _, _, history = subgradient_descent(
        funcs, x_0, max_iter=max_iter, tol=tol
    )
    assert abs(constraint_violation(result)) < tol, "Constraint not satisfied."
    assert len(history) <= max_iter, "Exceeded max iterations."


def test_run_single_initialization(funcs):
    """Tests for the run_single_initialization function."""
    x_0 = numpy.array([0.5, 0.25, 0.25])
    args = (funcs, x_0, {"max_iter": 500, "tol": 1e-6})
    result, _, _, _ = run_single_initialization(args)
    assert (
        abs(constraint_violation(result)) < args[2]["tol"]
    ), "Constraint not satisfied."


def test_run_multiple_initializations_parallel(funcs):
    """Tests for the run_multiple_initializations_parallel function."""
    num_starts = 10
    tol = 1e-6
    best_solution, all_solutions = run_multiple_initializations_parallel(
        funcs, num_starts=num_starts, max_workers=4, max_iter=500, tol=tol
    )
    best_x, _, _, _ = best_solution
    assert (
        abs(constraint_violation(best_x)) < tol
    ), "Best solution does not satisfy constraint."
    assert (
        len(all_solutions) == num_starts
    ), "Not all initializations were run."


@pytest.mark.slow
def test_parallel_scalability(funcs):
    """Tests the scalability of the parallelization implementation.
    Can be skipped with `-m "not slow"` flag.
    """
    num_starts = 50
    _, all_solutions = run_multiple_initializations_parallel(
        funcs, num_starts=num_starts, max_workers=8, max_iter=500, tol=1e-6
    )
    assert (
        len(all_solutions) == num_starts
    ), "Not all initializations were run."
