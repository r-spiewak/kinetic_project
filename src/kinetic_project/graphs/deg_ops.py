"""This file contains functions for graph degree
calculations."""

from enum import Enum

import networkx
import numpy

from kinetic_project.graphs.graph_dimen_type_val import dimen_type_val


class AllowedDegs(Enum):
    """Class enumerating the allowed options for
    degree calculations."""

    IN = 0
    OUT = 1


def in_or_out_deg(
    G: networkx.Graph | numpy.ndarray,
    in_or_out: AllowedDegs,
    verts: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Calculates the in- or out-degree for each vertex.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        in_or_out (str): Whether to compute the in- or
            out- degrees. Allowed values are "in" and "out".
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.

    Returns:
        numpy.ndarray: A vector of in- or out-degrees.

    Raises:
        ValueError: If in_or_out is not "in" or "out".
    """
    A, verts = dimen_type_val(G, verts)
    if in_or_out not in AllowedDegs:
        raise ValueError(f"{in_or_out} is not in {AllowedDegs}.")
    if in_or_out == AllowedDegs.IN:
        return numpy.matmul(A, verts)
    return numpy.matmul(verts.transpose(), A).transpose()


def in_deg(
    G: networkx.Graph | numpy.ndarray,
    verts: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Calculates the in-degree for each vertex.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.

    Returns:
        numpy.ndarray: A vector of in-degrees.

    """
    return in_or_out_deg(G, AllowedDegs.IN, verts)


def out_deg(
    G: networkx.Graph | numpy.ndarray,
    verts: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Calculates the out-degree for each vertex.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.

    Returns:
        numpy.ndarray: A vector of out-degrees.

    """
    return in_or_out_deg(G, AllowedDegs.OUT, verts)
