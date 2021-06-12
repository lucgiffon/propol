import os
import numpy as np
import plotly.express as px
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from pathlib import Path


def show_alliance_matrix(df_alliance_matrix_A):
    fig = px.imshow(df_alliance_matrix_A.values,
                    labels=dict(x="Nuances listes municipales", y="Nuances candidat législative", color="Nombre d'alliance total"),
                    x=df_alliance_matrix_A.columns,
                    y=df_alliance_matrix_A.index
                    )
    fig.update_xaxes(side="top")
    fig.show()


def main():
    path_output_alliances_matrix = project_dir / "data/processed/alliance_matrix.csv"
    df_alliance_matrix_A = pd.read_csv(path_output_alliances_matrix, index_col=0)
    show_alliance_matrix(df_alliance_matrix_A)



if __name__ == "__main__":
    load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()