"""This file contains functions used for the
Subgradient Descent method using the Augmented
Lagrangian."""

from collections.abc import Callable
from multiprocessing import Pool
from typing import Any

import numpy


def objective_function(
    x: numpy.ndarray,
    funcs: list[Callable],
    argmax: bool = False,
) -> float:
    """Compute the sum of all objective functions.

    Args:
        x (numpy.ndarray): The vector of coordinates.
        funcs (list[callable]): The list of functions
            in the objective.
        argmax (bool): Whether to invert the objective
            to maximize the objective and find the
            argmax (True), instead of minimizing the
            objective to find the argmin (False).
            Defaults to False.

    Returns:
        numpy.float64: The objective function.
    """
    if argmax:
        return -sum(f(x_i) for x_i, f in zip(x, funcs))
    return sum(f(x_i) for x_i, f in zip(x, funcs))


def constraint_violation(x: numpy.ndarray) -> float:
    """Compute the constraint violation c(x).

    Args:
        x (numpy.ndarray): The vector of coordinates.

    Returns:
        numpy.float64: The amount of constraint violation.
    """
    return sum(x) - 1


def augmented_lagrangian(
    x: numpy.ndarray,
    funcs: list[Callable],
    lambd: float,
    rho: float,
    argmax: bool = False,
) -> float:
    """Compute the Augmented Lagrangian.

    Args:
        x (numpy.ndarray): The vector of coordinates.
        funcs (list[callable]): The list of functions
            in the objective.
        lambd (float): Lagrangian multiplier.
        rho (float): Penalty for constraint violations.
        argmax (bool): Whether to invert the objective
            to maximize the objective and find the
            argmax (True), instead of minimizing the
            objective to find the argmin (False).
            Defaults to False.

    Returns:
        numpy.ndarray: The Augmented Lagrangian.
    """
    c = constraint_violation(x)
    return objective_function(x, funcs, argmax) + lambd * c + 0.5 * rho * c**2


def numerical_subgradient(
    f: Callable,
    x: float,
    epsilon: float = 1e-6,
    argmax: bool = False,
) -> float:
    """Estimate the gradient/subgradient numerically,
    using a finite difference method.

    Args:
        f (callable): The function whose subgradient
            should be estimated.
        x (float): The point at which to estimate the
            subgradient.
        epsilon (float): The offset from the point
            for the finite difference method.
        argmax (bool): Whether to invert the objective
            to maximize the objective and find the
            argmax (True), instead of minimizing the
            objective to find the argmin (False).
            Defaults to False.

    Returns:
        float: The calculated subgradient of f at x.
    """
    if argmax:
        return -(f(x + epsilon) - f(x)) / epsilon
    return (f(x + epsilon) - f(x)) / epsilon


def compute_subgradient(
    x: numpy.ndarray,
    funcs: list[Callable],
    argmax: bool = False,
) -> numpy.ndarray:
    """Compute the subgradient of all functions at point x.

    x (numpy.ndarray): The vector of coordinates.
        funcs (list[callable]): The list of functions
            in the objective.
        argmax (bool): Whether to invert the objective
            to maximize the objective and find the
            argmax (True), instead of minimizing the
            objective to find the argmin (False).
            Defaults to False.

    Returns:
        numpy.ndarray: The subgradient of the
            objective function.
    """
    subgrads = []
    for x_i, f in zip(x, funcs):
        # Replace this with a specific gradient or subgradient computation for f
        # if f == f_abs:  # Example: Check for non-differentiable |x|
        #     subgrads.append(subgradient_abs(x_i))
        # elif f == f_square:  # Example: f(x) = x^2
        #     subgrads.append(2 * x_i)
        # else:
        #     raise ValueError(
        #         "Gradient/subgradient computation not implemented for this function."
        #     )
        subgrads.append(numerical_subgradient(f, x_i, argmax=argmax))
    return numpy.array(subgrads)


def subgradient_descent(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    funcs: list[Callable],
    x_0: numpy.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta_0: float = 0.01,
    rho_0: float = 1.0,
    beta: float = 1.5,
    gamma: float = 0.9,
    argmax: bool = False,
) -> tuple[numpy.ndarray, float, float, list[dict]]:
    """Perform subgradient descent on an Augmented Lagrangian
    with a dynamically updated penalty parameter and step
    size.

    Parameters:
        funcs (list[callable]): The list of functions
            in the objective.
        x_0 (numpy.ndarray): Initial coordinate vector.
            If None, uses an equal distribution.
            Defaults to None.
        max_iter (int): Maximum number of iterations.
            Defaults to 1000.
        tol (float): Tolerance for convergence.
            Defaults to 1e-6.
        eta_0 (float): Initial step size.
            Defaults to 0.01.
        rho_0 (float): Initial penalty parameter.
            Defaults to 1.0.
        beta (float): Factor to increase rho.
            Defaults to 1.5.
        gamma (float): Reduction factor for step size.
            Defaults to 0.9.
        argmax (bool): Whether to invert the objective
            to maximize the objective and find the
            argmax (True), instead of minimizing the
            objective to find the argmin (False).
            Defaults to False.

    Returns:
        tuple: (
            Optimal arguments x,
            Objective function value at optimal arguments x,
            Lagrange multiplier lambda,
            history
        )
    """
    # Initialization:
    n = len(funcs)
    x = (
        x_0 if x_0 is not None else numpy.ones(n) / n
    )  # Start with equal distribution, by default.
    lambd: float = 0.0  # Initial Lagrange multiplier.
    rho = rho_0
    eta = eta_0
    history: list[dict[str, Any]] = []  # To store convergence history.
    c = constraint_violation(x)

    for iteration in range(max_iter):

        # Compute subgradient:
        subgrad = compute_subgradient(x, funcs, argmax) + lambd + rho * c

        # Update variables:
        x -= eta * subgrad
        # Ensure x_i in [0, 1]:
        x = numpy.clip(x, 0, 1)

        # Find new constraint violation:
        c = constraint_violation(x)

        # Update Lagrange multiplier:
        lambd += rho * c

        # Check convergence
        history.append(
            {
                "iteration": iteration,
                "x": x.copy(),
                "c": c,
                "objective": objective_function(x, funcs, argmax),
            }
        )
        if abs(c) < tol:
            break

        # Update penalty parameter rho if constraint violation does not improve:
        if iteration > 0 and abs(history[-2]["c"]) - abs(c) < tol / 10:
            rho *= beta

        # Update step size (decay):
        eta *= gamma

    return x, objective_function(x, funcs, argmax), lambd, history


def run_single_initialization(
    args: Any,
) -> tuple[numpy.ndarray, float, float, list[dict]]:
    """Run a single initialization of the subgradient_descent function.

    Args:
        args (Any): All arguments to the subgradient_descent function.

    Returns:
        tuple: (
            Optimal arguments x,
            Objective function value at optimal arguments x,
            Lagrange multiplier lambda,
            history
        )
    """
    funcs, x_0, kwargs = args
    return subgradient_descent(funcs, x_0=x_0, **kwargs)


def run_multiple_initializations_parallel(
    funcs: list[Callable],
    num_starts: int = 10,
    max_workers: int | None = None,
    **kwargs: dict[str, Any],
) -> tuple[
    tuple[numpy.ndarray, float, float, list[dict]],
    list[tuple[numpy.ndarray, float, float, list[dict]]],
]:
    """Run multiple initializations of the
    subgradient_descent function, in parallel.

    Args:
        funcs (list[callable]): The list of functions
            in the objective.
        num_starts (int): Number of initialization
            points to use. Defaults to 10.
        max_workers (int | None): Maximum number of
            parallel processes to start at a time.
            If None, uses the number of available
            CPUs on the system. Defaults to None.
        **kwargs (dict[str, Any]): Additional keyword
            arguments to be passed to subgradient_descent
            methods.

    Returns:
        tuple[tuple, list]: The first element in
            this tuple looks as follows, and is for
            the initialization which produced the
            best results:
            (
                Optimal arguments x,
                Objective function value at optimal arguments x,
                Lagrange multiplier lambda,
                history
            )
            The second element in the tuple is a list of tuples
            with structure identical to that of the first
            element in the tuple, where each element in the
            list is from a different initialization.
    """
    n = len(funcs)
    # Generate multiple initial random points:
    initializations = [numpy.random.rand(n) for _ in range(num_starts)]
    # Normalize to satisfy the constraint:
    initializations = [x / numpy.sum(x) for x in initializations]

    # Prepare arguments for parallel processing:
    args = [(funcs, x_0, kwargs) for x_0 in initializations]

    # Use multiprocessing.Pool to parallelize:
    with Pool(processes=max_workers) as pool:
        results = pool.map(run_single_initialization, args)

    # Select the best result based on objective value:
    # Minimize objective value:
    best_solution = min(results, key=lambda r: r[1])
    return best_solution, results
