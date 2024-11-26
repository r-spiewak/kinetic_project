"""This file contains code for the converting of
a networkx graph to its adjacency matrix and back."""

import networkx
import numpy


def graph_to_mat(
    G: networkx.Graph
) -> numpy.ndarray:
    """This function converts a networkx.Graph
    into its adjacency matrix representation.

    Args:
        G (networkx.Graph): Graph to convert.
        
    Returns:
        numpy.ndarray: Adjacency matrix of G.
    """
    return networkx.to_numpy_array(G)


def mat_to_graph(
    A: numpy.ndarray,
    vertex_labels: list | None = None,
) -> networkx.Graph:
    """This function converts a numpy.ndarray
    representation of a graph's adjacency matrix
    into a networkx.Graph.

    Args:
        A (numpy.ndarray): Adjacency matrix to
            convert.
        vertex_labels (list | None): labels to
            be used for the vertices. If None,
            labels vertices with consecutive
            integers. Defaults to None.
        
    Returns:
        networkx.Graph: networkx.Graph representation
            of of G.
    """
    return networkx.from_numpy_array(
        A,
        create_using=networkx.DiGraph,
        nodelist=vertex_labels,
    )
