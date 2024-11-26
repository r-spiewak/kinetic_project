"""This file runs the Problem 1 Example and
writes the output to a separate text file."""

import time
from pathlib import Path

from kinetic_project.graphs.generate_random_graph import (
    generate_random_directed_graph,
)
from kinetic_project.graphs.graph_ops import iterate_subgraphs


def main():
    """The main function to be run."""
    G = generate_random_directed_graph(v=10, e=20, seed=2)
    start_time = time.time()
    subgraphs = iterate_subgraphs(G, k=10, v=2)
    print(f"Subgraph iteration time: {(time.time()-start_time)/60} min")
    outfilename = Path(__file__).name + ".txt"
    with open(outfilename, "w", encoding="utf8") as file:
        file.write(str(subgraphs))
    print(f"Total script time: {(time.time()-start_time)/60} min")


if __name__ == "__main__":
    main()
