import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA

def show_alliance_matrix(df_alliance_matrix_A):
    fig = px.imshow(df_alliance_matrix_A.values,
                    labels=dict(x="Nuances listes municipales", y="Nuances candidat législative", color="Nombre d'alliance total"),
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
        lst_projectors = [(Isomap(n_components=n_components, n_neighbors=x), f"Isomap: neighbors:{x}") for x in [3, 4, 5]]
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


def main():
    path_output_alliances_matrix = project_dir / "data/processed/alliance_matrix.csv"
    df_alliance_matrix_A = pd.read_csv(path_output_alliances_matrix, index_col=0)
    df_alliance_matrix_A /= np.sum(df_alliance_matrix_A.values, axis=1).reshape(-1, 1)
    # df_alliance_matrix_A = df_alliance_matrix_A.drop("EXG")
    # df_alliance_matrix_A = df_alliance_matrix_A.drop("FN")
    show_alliance_matrix(df_alliance_matrix_A)
    show_projection(df_alliance_matrix_A, method='tsne', n_components=2)
    show_projection(df_alliance_matrix_A, method='pca', n_components=2)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()