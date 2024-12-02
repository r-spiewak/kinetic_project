""""""

# import pytest

# # Mock `matplotlib.pyplot.show` to prevent actual plotting
# @pytest.fixture(autouse=True)
# def mock_matplotlib_show(mocker):
#     mocker.patch("kinetic_project.graphs.draw_graph.matplotlib.pyplot.show")
#     return mocker


# # Mock the external dependencies (in_deg, out_deg, dimen_type_val)
# @pytest.fixture(autouse=True)
# def mock_dependencies(mocker):
#     mocker.patch("your_module.in_deg", side_effect=lambda A: np.any(A, axis=1))
#     mocker.patch("your_module.out_deg", side_effect=lambda A: np.any(A, axis=0))
#     mocker.patch("your_module.dimen_type_val", side_effect=lambda G, verts: (G, verts if verts is not None else np.ones(G.shape[0], dtype=int)))

# # Mock the `dimen_type_val` function
# @pytest.fixture(autouse=True)
# def mock_dimen_type_val(mocker):
#     def mock_func(G, verts):
#         if isinstance(G, nx.Graph):
#             A = nx.to_numpy_array(G)
#         else:
#             A = G
#         if verts is None:
#             verts = np.ones(A.shape[0], dtype=int)
#         return A, verts
#     mocker.patch("your_module.dimen_type_val", side_effect=mock_func)

# # Mock the `graph_to_mat` function
# @pytest.fixture(autouse=True)
# def mock_graph_to_mat(mocker):
#     mocker.patch("your_module.graph_to_mat", side_effect=nx.to_numpy_array)
