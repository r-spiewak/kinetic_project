"""This file contains tests for the functions in the graph_dimen_type_val.py module."""

import networkx
import numpy
import pytest

from kinetic_project.graphs.graph_dimen_type_val import (
    dimen_type_val,
    graph_dimen_type_val,
)


# Tests for `graph_dimen_type_val`:
def test_graph_dimen_type_val_with_networkx_graph():
    """Tests the function graph_dimen_type_val with a networkx.Graph."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    adj_matrix, dim = graph_dimen_type_val(G)
    expected_matrix = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    assert numpy.array_equal(
        adj_matrix, expected_matrix
    ), "Adjacency matrix should match expected output."
    assert dim == 3, "Dimension should match the number of nodes in the graph."


def test_graph_dimen_type_val_with_numpy_array():
    """Tests the function graph_dimen_type_val with a numpy.ndarray."""
    adj_matrix = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    result_matrix, dim = graph_dimen_type_val(adj_matrix)
    assert numpy.array_equal(
        result_matrix, adj_matrix
    ), "Adjacency matrix should match the input."
    assert dim == 2, "Dimension should match the matrix size."


def test_graph_dimen_type_val_with_non_square_matrix():
    """Tests the function graph_dimen_type_val with a non-square matrix."""
    adj_matrix = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    with pytest.raises(ValueError, match="do not match."):
        graph_dimen_type_val(adj_matrix)


# Tests for `dimen_type_val`:
def test_dimen_type_val_with_default_verts():
    """Tests the function dimen_type_val with default verts."""
    adj_matrix = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    result_matrix, verts = dimen_type_val(adj_matrix)
    expected_verts = numpy.array([[1], [1]])
    assert numpy.array_equal(
        result_matrix, adj_matrix
    ), "Adjacency matrix should match the input."
    assert numpy.array_equal(
        verts, expected_verts
    ), "Default verts vector should be ones."


def test_dimen_type_val_with_custom_verts():
    """Tests the function dimen_type_val with custom verts."""
    adj_matrix = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    verts = numpy.array([[1], [0]])
    result_matrix, result_verts = dimen_type_val(adj_matrix, verts)
    assert numpy.array_equal(
        result_matrix, adj_matrix
    ), "Adjacency matrix should match the input."
    assert numpy.array_equal(
        result_verts, verts
    ), "Verts vector should match the input."


def test_dimen_type_val_with_invalid_verts_shape():
    """Tests the function dimen_type_val with invalid verts shape."""
    adj_matrix = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    invalid_verts = numpy.array([1, 0])  # Incorrect shape.
    with pytest.raises(ValueError, match="is not"):
        dimen_type_val(adj_matrix, invalid_verts)


def test_dimen_type_val_with_networkx_graph_and_custom_verts():
    """Tests the function dimen_type_val with networkx.Graph and custom verts."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    verts = numpy.array([[1], [1], [0]])
    adj_matrix, result_verts = dimen_type_val(G, verts)
    expected_matrix = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    assert numpy.array_equal(
        adj_matrix, expected_matrix
    ), "Adjacency matrix should match graph conversion."
    assert numpy.array_equal(
        result_verts, verts
    ), "Verts vector should match the input."


def test_dimen_type_val_with_networkx_graph_default_verts():
    """Tests the function dimen_type_val with networkx.Graph and default verts."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    adj_matrix, verts = dimen_type_val(G)
    expected_matrix = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    expected_verts = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        adj_matrix, expected_matrix
    ), "Adjacency matrix should match graph conversion."
    assert numpy.array_equal(
        verts, expected_verts
    ), "Default verts vector should be ones."


def test_dimen_type_val_with_empty_graph():
    """Tests the function dimen_type_val with empty graph."""
    adj_matrix = numpy.array([[]])
    with pytest.raises(ValueError, match="do not match."):
        dimen_type_val(adj_matrix)
