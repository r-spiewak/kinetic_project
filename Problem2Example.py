"""This file runs the Problem 2 Example and
writes the output to a separate text file."""

import time

from kinetic_project.optimizations.subgradient_methods import (
    run_multiple_initializations_parallel,
)


def f_1(x: float) -> float:
    return 2 * x ** (1 / 3)


def f_2(x: float) -> float:
    return 5 * x**2


def f_3(x: float) -> float:
    return x + 1 if x > 0.5 else 0


def main():
    """The main function to be run."""
    start_time = time.time()
    # Define your functions as a list
    funcs = [
        # Functions must be defined explicitly,
        # instead of as lambda functions, due to
        # pickling requirements of multiprocessing.
        f_1,
        f_2,
        f_3,
    ]
    argmax = True

    # solution, lagrange_multiplier, history = subgradient_descent(funcs, argmax=argmax)
    (best_x, best_value, best_lambda, best_history), all_results = (
        run_multiple_initializations_parallel(
            funcs, num_starts=20, argmax=argmax
        )
    )

    print("Optimal arguments:", best_x)
    # print("Objective function value: ", -history[-1]["objective"] if argmax else history[-1]["objective"])
    print("Objective function value: ", -best_value if argmax else best_value)
    # print("Lagrange multiplier:", lagrange_multiplier)

    # print(f"Subgraph iteration time: {(time.time()-start_time)/60} min")
    # outfilename = Path(__file__).name + ".txt"
    # with open(outfilename, "w", encoding="utf8") as file:
    #     file.write(str(subgraphs))
    # print(f"Total script time: {(time.time()-start_time)/60} min")
    print(f"Total script time: {(time.time()-start_time)} s")
    # pdb.set_trace()


if __name__ == "__main__":
    main()
