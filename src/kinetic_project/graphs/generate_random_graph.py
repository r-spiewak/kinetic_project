"""This file contains code for the generation of
a random graph."""

import networkx


def generate_random_directed_graph(
    v: int,
    e: int,
    seed: int | None = None,
) -> networkx.Graph:
    """Generates a random Erdos-Renyi directed graph
    with v vertices and e edges.

    Args:
        v (int): Number of vertices in the graph.
        e (int): Expected number of edges in the graph.
        seed (int | None): random seed for the graph.
            Defaults to None.

    Returns:
        networkx.Graph: The generated graph.

    Raises:
        ValueError: If too many e for v.
    """
    # e = pv(v-1)/2  # Is this only for undirected?
    # p = 2*e/(v*(v-1))
    # e = pv(v-1)  # Maybe this for directed? Yes, but still probability based.
    # p = e/(v*(v-1))
    # return  networkx.fast_gnp_random_graph(v, p, seed=seed, directed=True)
    G = networkx.gnm_random_graph(v, e, seed=seed, directed=True)
    if len(G.edges) < e:
        raise ValueError(f"{e} edges is too many for {v} vertices.")
    return G
