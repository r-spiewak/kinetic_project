"""This file contains code for the drawing of
a networkx graph."""

import matplotlib.pyplot
import networkx


def draw_graph(
    G: networkx.Graph, 
) -> None:
    """This function plots a networkx.Graph object
    using matplotlib.pyplot.
    
    Args:
        G (networkx.Graph): The graph to plot.
    """
    networkx.draw(G, with_labels=True)
    matplotlib.pyplot.show()
