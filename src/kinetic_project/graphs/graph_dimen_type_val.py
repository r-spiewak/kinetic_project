"""This file contains a function to validate
(and create) the types and dimensions of
adjacency matrices and active vertices vectors.
"""

from typing import Tuple

import networkx
import numpy

from kinetic_project.graphs.graph_mat_conversions import graph_to_mat


def dimen_type_val(
    G: networkx.Graph | numpy.ndarray,
    verts: numpy.ndarray | None = None,
) -> Tuple[numpy.ndarray,numpy.ndarray]:
    """This function validates the dimensions and
    types of input objects, and creates the verts
    array if it is not passed in.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.
    
    Returns:
        (numpy.ndarray, numpy.ndarray): The validated
            adjacency matrix and verts vector.

    Raises:
        ValueError: If the shape of the numpy.ndarray
            for verts is not (n by 1), where n is the
            dimension of the adjacency matrix for G.
    """
    A, dim = graph_dimen_type_val(G)
    if verts is None:
        verts = numpy.ones((dim,1))
    elif verts.shape != (dim,1):
        raise ValueError(
            f"{verts.shape} is not {(dim,1)}."
        )
    return (A, verts)


def graph_dimen_type_val(
    G: networkx.Graph | numpy.ndarray,
) -> Tuple[numpy.ndarray, int]:
    """This function validates the dimensions and type
    of the input graph.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
    
    Returns:
        (numpy.ndarray, int): The validated
            adjacency matrix and its dimension.

    Raises:
        ValueError: If the shape of the numpy.ndarray
            for G is not square.
    """
    if isinstance(G, networkx.Graph):
        A = graph_to_mat(G)
    else:
        A = G.copy()
    dim1, dim2 = A.shape
    if dim1 == dim2:
        dim = dim1
    else:
        raise ValueError(
            f"The dimensions of {A.shape} do not match."
        )
    return (A, dim)