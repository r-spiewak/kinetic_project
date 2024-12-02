"""This file contains tests for the functions in the graph_ops.py module."""

import numpy

from kinetic_project.graphs.graph_ops import (
    iterate_subgraphs,
    prune_graph,
    validate_graph_sink_source_condition,
    validate_vertices,
    zero_rows_and_cols,
)


def test_iterate_subgraphs_basic():
    """Tests the function iterate_subgraphs in a simple case."""
    A = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    subgraphs = iterate_subgraphs(A)
    assert len(subgraphs) > 0, "Should return at least one subgraph."


def test_iterate_subgraphs_with_constraints():
    """Tests the function iterate_subgraphs with a required vertex."""
    A = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    required_vertices = [0]
    subgraphs = iterate_subgraphs(A, v=required_vertices)
    # assert all(validate_vertices(verts, required_vertices) for _, verts in subgraphs), \
    #     "All subgraphs should include required vertices."
    assert all(
        vert in verts for _, verts in subgraphs for vert in required_vertices
    ), "All subgraphs should include required vertices."


def test_prune_graph():
    """Tests the function prune_graph."""
    A = numpy.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )
    verts = numpy.array([[1], [1], [0]])
    pruned_A, pruned_verts = prune_graph(A, verts)
    assert pruned_A.shape == (
        2,
        2,
    ), "Pruned matrix should only include active vertices."
    assert numpy.array_equal(
        pruned_verts, [0, 1]
    ), "Pruned vertices should match active ones."


def test_validate_graph_sink_source_condition():
    """Tests the function validate_graph_sink_source_condition."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    verts = numpy.array([[1], [1], [1]])
    validated_A, validated_verts = validate_graph_sink_source_condition(
        A, verts
    )
    assert validated_A.shape == (3, 3), "Matrix shape should not change."
    assert (
        validated_verts[2] == 0
    ), "Sink/source vertices should be marked as inactive."


def test_validate_vertices():
    """Tests the function validate_vertices."""
    active_verts = numpy.array([[1], [1], [0]])
    required_vertices = [0, 1]
    assert validate_vertices(
        active_verts, required_vertices
    ), "Should return True for active required vertices."
    assert not validate_vertices(
        active_verts, [2]
    ), "Should return False for inactive required vertices."


def test_zero_rows_and_cols():
    """Tests the function zero_rows_and_cols."""
    mat = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    vec = numpy.array([[1], [0], [1]])
    zeroed_mat = zero_rows_and_cols(mat, vec)
    expected_mat = numpy.array(
        [
            [1, 0, 3],
            [0, 0, 0],
            [7, 0, 9],
        ]
    )
    assert numpy.array_equal(
        zeroed_mat, expected_mat
    ), "Rows and columns should be zeroed correctly."


def test_zero_rows_and_cols_in_place():
    """Tests that the argument in_place=True to the function zero_rows_and_cols makes the changes in place."""
    mat = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    vec = numpy.array([[1], [0], [1]])
    zeroed_mat = zero_rows_and_cols(mat, vec, in_place=True)
    assert zeroed_mat is mat, "Operation should be performed in place."
