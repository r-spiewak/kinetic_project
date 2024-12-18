"""This file contains tests for the functions in the generate_random_graph.py module."""

import networkx
import pytest

from kinetic_project.graphs.generate_random_graph import (
    generate_random_directed_graph,
)


def test_generate_random_directed_graph_structure():
    """This function tests the structure generated by the function generate_random_directed_graph."""
    v, e = 5, 10
    graph = generate_random_directed_graph(v, e)
    assert (
        len(graph.nodes) == v
    ), "Graph should have correct number of vertices."
    assert len(graph.edges) == e, "Graph should have correct number of edges."


def test_generate_random_directed_graph_empty_graph():
    """This function tests the structure generated by the function generate_random_directed_graph for an empty graph."""
    v, e = 0, 0
    graph = generate_random_directed_graph(v, e)
    assert len(graph.nodes) == 0, "Empty graph should have no vertices."
    assert len(graph.edges) == 0, "Empty graph should have no edges."


def test_generate_random_directed_graph_seed():
    """This function tests the structure generated by the function generate_random_directed_graph for the seed argument."""
    v, e, seed = 5, 7, 42
    graph1 = generate_random_directed_graph(v, e, seed)
    graph2 = generate_random_directed_graph(v, e, seed)
    assert networkx.is_isomorphic(
        graph1, graph2
    ), "Graphs generated with the same seed should be identical."


def test_generate_random_directed_graph_invalid_edges():
    """This function tests the structure generated by the function generate_random_directed_graph for an invalid number of edges."""
    v, e = 5, 30  # Too many edges for the given number of vertices.
    with pytest.raises(ValueError):
        generate_random_directed_graph(v, e)
