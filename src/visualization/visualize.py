import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import click
import networkx as nx

from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from networkx import draw, draw_spectral, draw_planar, draw_spring, draw_shell, adjacency_matrix, draw_kamada_kawai, \
    kamada_kawai_layout, spectral_layout, bipartite_layout, spring_layout
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA

project_dir = Path(os.getcwd())


def show_alliance_matrix(df_alliance_matrix_A):
    fig = px.imshow(df_alliance_matrix_A.values,
                    labels=dict(x="Nuances listes municipales", y="Nuances candidat législative",
                                color="Nombre d'alliance total"),
                    x=df_alliance_matrix_A.columns,
                    y=df_alliance_matrix_A.index
                    )
    fig.update_xaxes(side="top")
    fig.show()


def show_projection(df_alliance_matrix_A, method='tsne', n_components=2):
    assert 0 < n_components <= 2
    arr_alliance_matrix = df_alliance_matrix_A.values

    if method == 'tsne':
        lst_projectors = [(TSNE(n_components=n_components, perplexity=x), f"TSNE: perplexity:{x}") for x in [3, 4]]
    elif method == "isomap":
        lst_projectors = [(Isomap(n_components=n_components, n_neighbors=x), f"Isomap: neighbors:{x}") for x in
                          [3, 4, 5]]
    elif method == "pca":
        lst_projectors = [(PCA(n_components=n_components), "PCA")]
    else:
        raise ValueError(f"Unknown projection method {method}")

    for projector, projector_desc in lst_projectors:
        proj_arr_alliance_matrix = projector.fit_transform(arr_alliance_matrix)
        fig = go.Figure()
        if n_components == 1:
            x_vals = proj_arr_alliance_matrix[:, 0]
            y_vals = proj_arr_alliance_matrix[:, 0]
        else:
            x_vals = proj_arr_alliance_matrix[:, 0]
            y_vals = proj_arr_alliance_matrix[:, 1]
        fig.add_trace(go.Scatter(x=x_vals,
                                 y=y_vals,
                                 mode='markers+text',
                                 # marker_color=df_alliance_matrix_A.index,
                                 marker=dict(
                                     size=20,
                                     color='LightSkyBlue',
                                     line=dict(
                                         color="MediumPurple",
                                         width=2
                                     )
                                 ),
                                 text=df_alliance_matrix_A.index,
                                 textposition='top center'))

        fig.update_layout(title=projector_desc)
        fig.show()


def show_graph(df_alliance_matrix_A):
    G = nx.Graph()
    for party_name in df_alliance_matrix_A.index:
        G.add_node(party_name)
        for i_col, nb_alliance_first_party in enumerate(df_alliance_matrix_A.loc[party_name]):
            if nb_alliance_first_party == 0:
                continue
            list_name = df_alliance_matrix_A.columns[i_col]
            G.add_edge(party_name, list_name, weight=nb_alliance_first_party)
            # for i_row, nb_alliance_2nd_party in enumerate(df_alliance_matrix_A.loc[:, list_name]):
            #     second_party_name = df_alliance_matrix_A.index[i_row]
            #     cumulated_alliances = nb_alliance_first_party + nb_alliance_2nd_party
            #     G.add_edge(party_name, second_party_name, weight=cumulated_alliances)

    # draw_kamada_kawai(G)
    # A = adjacency_matrix(G).toarray()

    dct_layout = spring_layout(G)
    fig = go.Figure()
    for node in dct_layout:
        if node in df_alliance_matrix_A.columns:
            color = "red"
        else:
            color = "green"
        fig.add_trace(go.Scatter(
            x=(dct_layout[node][0],),
            y=(dct_layout[node][1],),
            name=node,
            text=node,
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=20,
                color=color
            )))
    fig.show()
    print()


@ click.command()
@ click.argument("method", type=click.Choice(["matrix", "pca", "tsne", "isomap", "graph"]))
def main(method):
    path_output_alliances_matrix = project_dir / "data/processed/alliance_matrix.csv"
    df_alliance_matrix_A = pd.read_csv(path_output_alliances_matrix, index_col=0)
    df_alliance_matrix_A /= np.sum(df_alliance_matrix_A.values, axis=1).reshape(-1, 1)
    # df_alliance_matrix_A = df_alliance_matrix_A.drop("EXG")
    # df_alliance_matrix_A = df_alliance_matrix_A.drop("FN")
    if method == "matrix":
        show_alliance_matrix(df_alliance_matrix_A)
    elif method == "graph":
        show_graph(df_alliance_matrix_A)
    else:
        show_projection(df_alliance_matrix_A, method=method, n_components=2)


def run():
    global project_dir
    load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()


if __name__ == "__main__":
    run()
