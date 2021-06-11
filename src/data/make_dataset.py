# -*- coding: utf-8 -*-
import click
import os
import pandas as pd
import numpy as np
import unidecode


from loguru import logger
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def create_id_from_row(row, cols_names_for_id, dct_col_names_as_is):
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

    dct_other_cols = dict((key, row[value]) for key, value in dct_col_names_as_is.items())

    return dict(id_candidate=id_candidate, **dct_other_cols)

@click.command()
def main():
    prepare_sanitized_list_of_candidates()
    prepare_sanitized_list_of_lists()


def prepare_sanitized_list_of_candidates():
    """Write processed file of candidates with their party.
    """
    path_candidates_list = project_dir / "data/external/Leg_2017_candidatures_T1_c2.csv"
    # path_nuance_association = project_dir / "data/external/Leg_2017_candidatures_T1_c2_nuances.csv"

    path_output_sanitized_candidates_list = project_dir / "data/interim/sanitized_candidates.csv"
    path_output_sanitized_candidates_list.parent.mkdir(parents=True, exist_ok=True)
    with open(path_candidates_list, 'rb') as candidates_file:
        df_candidates = pd.read_csv(candidates_file, sep=",", skiprows=0)
    # with open(path_nuance_association, 'rb') as nuance_file:
    #     df_nuance_association = pd.read_csv(nuance_file, sep=",", skiprows=0)
    #
    # dct_nuance_association = dict(line for line in df_nuance_association.values)

    cols_for_id_candidates = (
        "Code du département",
        "Nom candidat",
        "Prénom candidat"
    )
    dct_col_for_label = {
        "party_short": "Nuance candidat"
    }
    df_final = df_candidates.apply(lambda x: create_id_from_row(x, cols_for_id_candidates, dct_col_for_label), axis=1, result_type='expand')
    assert df_final["id_candidate"].unique().size == len(df_final)  # assert there is no duplicate name
    # df_final["party_long"] = df_final.apply(lambda row: dct_nuance_association[row["party_short"]], axis=1)
    df_final.to_csv(path_output_sanitized_candidates_list, index=False)


def prepare_sanitized_list_of_lists():
    """
    Write processed file of the list candidates with their list.
    :return:
    """
    path_candidates_list = project_dir / "data/external/livre-des-listes-et-candidats.txt"
    path_output_sanitized_candidates_list = project_dir / "data/interim/sanitized_lists.csv"
    path_output_sanitized_candidates_list.parent.mkdir(parents=True, exist_ok=True)
    with open(path_candidates_list, 'rb') as list_file:
        df_lists = pd.read_csv(list_file, sep="\t", encoding="ISO-8859-1", skiprows=2)

    df_lists = df_lists[:1000]
    cols_for_id_candidates = (
        "Code du département",
        "Nom candidat",
        "Prénom candidat"
    )
    dct_col_for_label = {
        "party": "Nuance Liste",
        # "short_list_name": "Libellé abrégé liste",
        # "long_list_name": "Libellé Etendu Liste"
    }
    df_final = df_lists.apply(lambda x: create_id_from_row(x, cols_for_id_candidates, dct_col_for_label), axis=1,
                                   result_type='expand')
    assert df_final["id_candidate"].unique().size == len(df_final)  # assert there is no duplicate name

    df_final.to_csv(path_output_sanitized_candidates_list, index=False)




if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    dct_env = load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()
