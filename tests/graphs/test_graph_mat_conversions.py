"""This file contains tests for the functions in the graph_mat_conversions.py module."""

import networkx
import numpy
import pytest

from kinetic_project.graphs.graph_mat_conversions import (
    graph_to_mat,
    mat_to_graph,
)


def test_graph_to_mat_valid_conversion():
    """This function tests that the convertsion from a graph to a matrix produces the expected result."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    adjacency_matrix = graph_to_mat(G)
    expected_matrix = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    assert numpy.array_equal(
        adjacency_matrix, expected_matrix
    ), "Adjacency matrix should match expected output"


def test_mat_to_graph_valid_conversion():
    """This function tests that the convertsion from a matrix to a graph produces the expected result."""
    adjacency_matrix = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    graph = mat_to_graph(adjacency_matrix)
    edges = list(graph.edges)
    expected_edges = [(0, 1), (1, 2), (2, 0)]
    assert sorted(edges) == sorted(
        expected_edges
    ), "Graph edges should match expected output."


def test_graph_to_mat_and_back():
    """This function tests that the convertsion from a graph to a matrix and back to a graph produces the expected result."""
    original_graph = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    mat = graph_to_mat(original_graph)
    reconstructed_graph = mat_to_graph(mat)
    assert networkx.is_isomorphic(
        original_graph, reconstructed_graph
    ), "Original and reconstructed graph should be identical."


def test_mat_to_graph_with_labels():
    """This function tests that the convertsion from a matrix to a graph includes the expected labels."""
    adjacency_matrix = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    labels = ["A", "B"]
    graph = mat_to_graph(adjacency_matrix, labels)
    assert (
        list(graph.nodes) == labels
    ), "Graph nodes should match the provided labels."


# Can't actually make the invalid matrix in here. Numpy won't allow it.
# def test_mat_to_graph_invalid_matrix():
#     """This function tests that the convertsion from a matrix to a graph with an invalid matrix produces the expected error."""
#     invalid_matrix = numpy.array([
#         [0, 1, 2],
#         [1, 0],
#     ])
#     with pytest.raises(ValueError):
#         mat_to_graph(invalid_matrix)
