"""This file contains tests for the functions in the deg_ops.py module."""

import sys

import networkx
import numpy
import pytest

from kinetic_project.graphs.deg_ops import (
    AllowedDegs,
    in_deg,
    in_or_out_deg,
    out_deg,
)


def test_in_or_out_deg_in_degree():
    """Tests the function in_or_out_deg for AllowedDegs.IN."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    verts = numpy.array([[1], [1], [1]])
    result = in_or_out_deg(A, AllowedDegs.IN, verts)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "In-degree calculation is incorrect."


def test_in_or_out_deg_out_degree():
    """Tests the function in_or_out_deg for AllowedDegs.OUT."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    verts = numpy.array([[1], [1], [1]])
    result = in_or_out_deg(A, AllowedDegs.OUT, verts)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "Out-degree calculation is incorrect."


def test_in_or_out_deg_invalid_enum():
    """Tests the function in_or_out_deg for an invalid AllowedDeg enum."""
    A = numpy.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    verts = numpy.array([[1], [1]])
    if sys.version_info[0] == 3 and sys.version_info[1] < 12:
        with pytest.raises(TypeError):
            in_or_out_deg(A, "invalid_enum", verts)
    else:
        with pytest.raises(ValueError):
            in_or_out_deg(A, "invalid_enum", verts)


def test_in_deg():
    """Tests the function in_deg."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    result = in_deg(A)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "In-degree function is incorrect."


def test_out_deg():
    """Tests the function out_deg."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    result = out_deg(A)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "Out-degree function is incorrect."


def test_in_or_out_deg_with_partial_vertices():
    """Tests the function in_or_out_deg when only some vertices are active."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    verts = numpy.array([[1], [0], [1]])  # Only vertices 0 and 2 are active.
    in_result = in_or_out_deg(A, AllowedDegs.IN, verts)
    out_result = in_or_out_deg(A, AllowedDegs.OUT, verts)
    # Note: the function `in_or_out_deg doesn't remove vertices or edges.
    # That is part of the job of the function `validate_graph_sink_source_condition`.
    # Therefore, though vertex 1 is not active, and the values
    # expected_in = numpy.array([[1], [0], [0]])
    # expected_out = numpy.array([[0], [0], [1]])
    # may be the final output of this stage, in_or_out_deg only
    # calculates the degrees from the given matrix, so the expected
    # outputs are actually as follows:
    expected_in = numpy.array([[1], [0], [1]])
    expected_out = numpy.array([[1], [0], [1]])
    assert numpy.array_equal(
        in_result, expected_in
    ), "In-degree with partial vertices is incorrect."
    assert numpy.array_equal(
        out_result, expected_out
    ), "Out-degree with partial vertices is incorrect."


def test_in_deg_with_graph_object():
    """Tests the function in_or_out_deg with a networkx.Graph object input."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    result = in_deg(G)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "In-degree for networkx.Graph is incorrect."


def test_out_deg_with_graph_object():
    """Tests the function out_deg with a networkx.Graph object input."""
    G = networkx.DiGraph([(0, 1), (1, 2), (2, 0)])
    result = out_deg(G)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "Out-degree for networkx.Graph is incorrect."


# This is handled directly by graph_dimen_type_val:
# def test_in_or_out_deg_with_empty_graph():
#     """Tests the function in_deg with a networkx.Graph object input."""
#     A = numpy.array([[]])
#     verts = numpy.array([[]])
#     result = in_or_out_deg(A, AllowedDegs.IN, verts)
#     assert result.size == 0, "In-degree for empty graph should return an empty array."


def test_in_or_out_deg_without_active_vertices():
    """Tests the function in_or_out_deg without any input active verts."""
    A = numpy.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    verts = None
    result = in_or_out_deg(A, AllowedDegs.IN, verts)
    expected = numpy.array([[1], [1], [1]])
    assert numpy.array_equal(
        result, expected
    ), "In-degree should handle None vertices correctly."
