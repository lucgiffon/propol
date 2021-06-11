# -*- coding: utf-8 -*-
import click
import os
import pandas as pd
import numpy as np
import unidecode


from loguru import logger
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def create_id_from_row(row, cols_names_for_id, col_name_for_label):
    """
    Function to be applied row-wise on a dataframe to produce the pairs (id_candidate, party)

    id_candidate is produced by concatenating the column values in `cols_names_for_id` and interspace them with "_".
    :param row:
    :param cols_names_for_id:
    :param col_name_for_label:
    :return:
    """
    id_candidate = "_".join(map(lambda x: str.lower(unidecode.unidecode(str(row[x]))),
                                cols_names_for_id))

    return {"id_candidate": id_candidate,
            "party": row[col_name_for_label]}

@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    path_candidates_list = project_dir / "data/external/Leg_2017_candidatures_T1_c2.csv"
    path_output_sanitized_candidates_list = project_dir / "data/processed/sanitized_candidates.csv"
    path_output_sanitized_candidates_list.parent.mkdir(parents=True, exist_ok=True)
    with open(path_candidates_list, 'rb') as candidates_file:
        # _str = candidates_file.read()
        df_candidates = pd.read_csv(candidates_file, sep=",", skiprows=0)

    cols_for_id_candidates = (
        "Code du département",
        "Nom candidat",
        "Prénom candidat"
    )
    col_for_label = "Nuance candidat"
    df_final = df_candidates.apply(lambda x: create_id_from_row(x, cols_for_id_candidates, col_for_label), axis=1, result_type='expand')
    assert df_final["id_candidate"].unique().size == len(df_final)  # check there is no duplicate name
    print(df_candidates)

    df_final.to_csv(path_output_sanitized_candidates_list)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    dct_env = load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()
