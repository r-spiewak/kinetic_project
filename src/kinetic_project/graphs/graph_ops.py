"""This file contains functions to be performed
on graph adjacency matrices."""

import networkx
import numpy

from kinetic_project.graphs.deg_ops import in_deg, out_deg
from kinetic_project.graphs.graph_dimen_type_val import dimen_type_val


def iterate_subgraphs(
    G: numpy.ndarray | networkx.Graph,
    verts: numpy.ndarray | None = None,
    k: int | None = None,
    v: int | list[int] | None = None,
    hashmap: dict | None = None,
) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
    """This function validates conditions on a
    graph, and then iterates through and validates
    conditions on all the subgraphs recursively,
    by iteratively removing one edge from the graph.

    Args:
        G (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.
        k (int | None): 1 + maximum number of edges allowed
            in a valid subgraph. If None, set to one more
            than the number of edges in the graph. Defaults
            to None.
        v (in | list[int] | None): Vertices which
            must be included in subgraphs for the
            subgraphs to be considered valid. Each
            element represents the index, or label,
            of the corresponding vertex. If None, no
            specific vertex is required to be in a
            subgraph for the subgraph to be considered
            valid. Defaults to None.
        hashmap (dict | None): Hashmap of subgraphs that
            have been checked. If None, a hashmap is
            created. Defaults to None.

    Returns:
        list: list of tuples of valid subgraphs and
            their corresponding vertex lists.
    """
    A, verts = dimen_type_val(G, verts)
    if hashmap is None:
        hashmap = {}
    if k is None:
        k = sum(sum(A)) + 1
    subgraph_list: list = []
    A, verts = validate_graph_sink_source_condition(A, v=v)
    # key = (tuple(A), tuple(verts))
    # if key in hashmap:
    if (key := (tuple(A), tuple(verts))) in hashmap:
        return subgraph_list
    hashmap[key] = 1
    if 1 < sum(sum(A)) < k:
        subgraph_list.append(prune_graph(A, verts))
    for i, j in numpy.argwhere(A):
        # If I just reduce each individual edge to 0,
        # I may miss subgraphs in the case when there
        # are multiple edges going from one vertex to
        # another, i.e., when the value of the element
        # in the adjacency matrix is >1.
        this_A, this_verts = A.copy(), verts.copy()
        this_A[i, j] = 0
        this_A, this_verts = validate_graph_sink_source_condition(
            this_A, this_verts, v
        )
        subgraph_list.extend(
            iterate_subgraphs(this_A, this_verts, k, v, hashmap)
        )
    return subgraph_list


def prune_graph(
    A: numpy.ndarray | networkx.Graph,
    verts: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """This function prunes disconnected vertices
    from the adjacency matrix of a graph.

    Args:
        A (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray): A vector representing
            the active vertices in the graph.

    Returns:
        (numpy.ndarray, numpy.ndarray): The pruned
            adjacency matrix and the vertex labels.
    """
    A, verts = dimen_type_val(A, verts)
    active_verts = numpy.nonzero(verts)[0]
    active_eye = numpy.zeros((A.shape[0], len(active_verts)))
    for ind, vert in enumerate(active_verts):
        active_eye[vert, ind] = 1
    pruned_A = numpy.matmul(active_eye.T, A)
    pruned_A = numpy.matmul(pruned_A, active_eye)
    return (pruned_A, active_verts)


def validate_graph_sink_source_condition(
    A: numpy.ndarray | networkx.Graph,
    verts: numpy.ndarray | None = None,
    v: int | list[int] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """This function validates that a graph
    (represented by its adjacency matrix) does
    not have any vertices which are sinks or
    sources. Any such vertices are removed from
    the graph by zeroing out the corresponding
    rows and columns in the adjacency matrix,
    and setting the active status in the verts
    vector to zero. This function also creates
    the verts vector if it is not passed in.

    Args:
        A (networkx.Graph | numpy.ndarray): The graph,
            represented as a networkx.Graph or
            a numpy adjacency matrix.
        verts (numpy.ndarray | None): A vector
            representing the active vertices in the
            graph. If None, a vector of ones will
            be created. Defaults to None.
        v (in | list[int] | None): Vertices which
            must be included in subgraphs for the
            subgraphs to be considered valid. Each
            element represents the index, or label,
            of the corresponding vertex. If None, no
            specific vertex is required to be in a
            subgraph for the subgraph to be considered
            valid. Defaults to None.

    Returns:
        (numpy.ndarray, numpy.ndarray): The validated
            adjacency matrix and verts vector.
    """
    A, verts = dimen_type_val(A, verts)
    # print(f"Before: {A, verts}")
    verts = numpy.logical_and(verts, in_deg(A)).astype(int)
    verts = numpy.logical_and(verts, out_deg(A)).astype(int)
    A_new = zero_rows_and_cols(A.copy(), verts)
    # print(f"After: {A, verts}")
    # Check that necessary vertices are still included in the subgraph:
    if not validate_vertices(verts, v):
        dim = A.shape[0]
        return (numpy.zeros((dim, dim)), numpy.zeros((dim, 1)))
    if numpy.array_equal(A_new, A):
        return (A_new, verts)
    return validate_graph_sink_source_condition(A_new, verts, v)


def validate_vertices(
    active_verts: numpy.ndarray,
    v: int | list[int] | None = None,
) -> bool:
    """This function validates that a subgraph contains
    specific vertices.

    Args:
        active_verts (numpy.ndarray): A vector
            representing the active vertices in the
            graph.
        v (in | list[int] | None): Vertices which
            must be included in subgraphs for the
            subgraphs to be considered valid. Each
            element represents the index, or label,
            of the corresponding vertex. If None, no
            specific vertex is required to be in a
            subgraph for the subgraph to be considered
            valid. Defaults to None.

    Returns:
        bool: If all the required vertices are active
            in active_verts (True), or not (False).
    """
    if v is None:
        return True
    if isinstance(v, int):
        v = [v]
    if any(active_verts[vert] == 0 for vert in v):
        return False
    return True


def zero_rows_and_cols(
    mat: numpy.ndarray,
    vec: numpy.ndarray,
    in_place: bool = False,
) -> numpy.ndarray:
    """This function clears rows and columns
    in a matrix corresponding to the index
    where values in a vector are 0.

    Args:
        mat (numpy.ndarray): The n by n matrix
            whose rows and columns should be zeroed.
        vec (numpy.ndarray): The n by 1 vector whose
            values correspond to rows and columns in
            the matrix.
        in_place (bool): Whether to operate on the
            original matrix (True), or a copy (False).
            Defaults to False.

    Returns:
        numpy.ndarray: The matrix with the
            zeroed rows and columns corresponding
            to the zero values in the vector.
    """
    if not in_place:
        mat = mat.copy()
    inds = numpy.where(vec == 0)[0]
    mat[inds, :] = 0
    mat[:, inds] = 0
    return mat
