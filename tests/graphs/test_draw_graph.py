"""This file contains tests for the functions in the draw_graph.py module."""

import networkx

from kinetic_project.graphs.draw_graph import draw_graph


def test_draw_graph_with_simple_graph(mocker):
    """Test draw_graph with a simple graph."""
    # Create a simple graph:
    G = networkx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Mock `networkx.draw`:
    mock_draw = mocker.patch("kinetic_project.graphs.draw_graph.networkx.draw")
    # Mock `matplotlib.pyplot.show`:
    mock_matplotlib_show = mocker.patch(
        "kinetic_project.graphs.draw_graph.matplotlib.pyplot.show"
    )

    # Call `draw_graph`:
    draw_graph(G)

    # Verify `networkx.draw` is called with the correct arguments:
    mock_draw.assert_called_once_with(G, with_labels=True)

    # Verify `matplotlib.pyplot.show` is called:
    mock_matplotlib_show.assert_called_once()


def test_draw_graph_with_empty_graph(mocker):
    """Test draw_graph with an empty graph."""
    # Create an empty graph:
    G = networkx.Graph()

    # Mock `networkx.draw`:
    mock_draw = mocker.patch("kinetic_project.graphs.draw_graph.networkx.draw")
    # Mock `matplotlib.pyplot.show`:
    mock_matplotlib_show = mocker.patch(
        "kinetic_project.graphs.draw_graph.matplotlib.pyplot.show"
    )

    # Call `draw_graph`:
    draw_graph(G)

    # Verify `networkx.draw` is called, even if the graph is empty:
    mock_draw.assert_called_once_with(G, with_labels=True)

    # Verify `matplotlib.pyplot.show` is called:
    mock_matplotlib_show.assert_called_once()


def test_draw_graph_with_directed_graph(mocker):
    """Tests draw_graph with a directed graph."""
    # Create a directed graph:
    G = networkx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Mock `networkx.draw`:
    mock_draw = mocker.patch("kinetic_project.graphs.draw_graph.networkx.draw")
    # Mock `matplotlib.pyplot.show`:
    mock_matplotlib_show = mocker.patch(
        "kinetic_project.graphs.draw_graph.matplotlib.pyplot.show"
    )

    # Call `draw_graph`:
    draw_graph(G)

    # Verify `networkx.draw` is called with the correct arguments:
    mock_draw.assert_called_once_with(G, with_labels=True)

    # Verify `matplotlib.pyplot.show` is called:
    mock_matplotlib_show.assert_called_once()


def test_draw_graph_with_multigraph(mocker):
    """Tests draw_graph with a multigraph."""
    # Create a MultiGraph:
    G = networkx.MultiGraph()
    G.add_edges_from([(0, 1), (0, 1), (1, 2)])

    # Mock `networkx.draw`:
    mock_draw = mocker.patch("kinetic_project.graphs.draw_graph.networkx.draw")
    # Mock `matplotlib.pyplot.show`:
    mock_matplotlib_show = mocker.patch(
        "kinetic_project.graphs.draw_graph.matplotlib.pyplot.show"
    )

    # Call `draw_graph`:
    draw_graph(G)

    # Verify `networkx.draw` is called with the correct arguments:
    mock_draw.assert_called_once_with(G, with_labels=True)

    # Verify `matplotlib.pyplot.show` is called:
    mock_matplotlib_show.assert_called_once()


def test_draw_graph_with_attributes(mocker):
    """Tests draw_graph with graph attributes."""
    # Create a graph with attributes
    G = networkx.Graph()
    G.add_node(0, label="Node 0")
    G.add_node(1, label="Node 1")
    G.add_edge(0, 1, weight=5)

    # Mock `networkx.draw`:
    mock_draw = mocker.patch("kinetic_project.graphs.draw_graph.networkx.draw")
    # Mock `matplotlib.pyplot.show`:
    mock_matplotlib_show = mocker.patch(
        "kinetic_project.graphs.draw_graph.matplotlib.pyplot.show"
    )

    # Call `draw_graph`:
    draw_graph(G)

    # Verify `networkx.draw` is called with the correct arguments:
    mock_draw.assert_called_once_with(G, with_labels=True)

    # Verify `matplotlib.pyplot.show` is called:
    mock_matplotlib_show.assert_called_once()
