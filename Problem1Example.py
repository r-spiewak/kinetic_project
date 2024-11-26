"""This file runs the Problem 1 Example and
writes the output to a separate text file."""

from pathlib import Path

from kinetic_project.graphs.generate_random_graph import (
    generate_random_directed_graph,
)
from kinetic_project.graphs.graph_ops import iterate_subgraphs


def main():
    """The main function to be run."""
    G = generate_random_directed_graph(v=10, e=20, seed=2)
    subgraphs = iterate_subgraphs(G, k=10, v=2)
    outfilename = Path(__file__).name
    with open(outfilename, "r", encoding="utf8") as file:
        file.write(subgraphs)


if __name__ == "__main__":
    main()
