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
        "party_candidate": "Nuance candidat"
    }
    df_final = df_candidates.apply(lambda x: create_id_from_row(x, cols_for_id_candidates, dct_col_for_label), axis=1, result_type='expand')
    assert df_final["id_candidate"].unique().size == len(df_final)  # assert there is no duplicate name
    # df_final["party_long"] = df_final.apply(lambda row: dct_nuance_association[row["party_short"]], axis=1)
    df_final.set_index("id_candidate", inplace=True)
    df_final.to_csv(path_output_sanitized_candidates_list, index=True)


def prepare_sanitized_list_of_lists():
    """
    Write processed file of the list candidates with their list.
    :return:
    """
    path_candidates_list = project_dir / "data/external/livre-des-listes-et-candidats.txt"
    path_output_sanitized_list_of_lists = project_dir / "data/interim/sanitized_lists.csv"
    path_output_sanitized_list_of_lists.parent.mkdir(parents=True, exist_ok=True)
    with open(path_candidates_list, 'rb') as list_file:
        df_lists = pd.read_csv(list_file, sep="\t", encoding="ISO-8859-1", skiprows=2)

    cols_for_id_candidates = (
        "Code du département",
        "Nom candidat",
        "Prénom candidat"
    )
    dct_col_for_label = {
        "party_list": "Nuance Liste",
        # "short_list_name": "Libellé abrégé liste",
        # "long_list_name": "Libellé Etendu Liste"
    }
    df_final = df_lists.apply(lambda x: create_id_from_row(x, cols_for_id_candidates, dct_col_for_label), axis=1,
                                   result_type='expand')

    duplicated = df_final.duplicated(subset=["id_candidate"], keep=False)
    # I remove the lists without any party (cities with less than 3500 capita)
    df_final.dropna(inplace=True)
    # there is no way to identify the duplicate candidates so I just drop them in order to keep the data clean
    df_final = df_final[~duplicated]
    assert df_final["id_candidate"].unique().size == len(df_final)  # assert there is no duplicate name

    df_final.set_index("id_candidate", inplace=True)
    df_final.to_csv(path_output_sanitized_list_of_lists, index=True)


def prepare_alliance_matrix():

    path_output_sanitized_candidates_list = project_dir / "data/interim/sanitized_candidates.csv"
    path_output_sanitized_list_of_lists = project_dir / "data/interim/sanitized_lists.csv"

    path_output_alliances_matrix = project_dir / "data/processed/alliance_matrix.csv"
    path_output_alliances_matrix.parent.mkdir(parents=True, exist_ok=True)

    with open(path_output_sanitized_candidates_list, 'r') as in_candidates:
        df_candidates = pd.read_csv(in_candidates)
    with open(path_output_sanitized_list_of_lists, 'r') as in_lists:
        df_lists = pd.read_csv(in_lists)

    dct_candidate_party_list = dict(df_lists.values)
    del df_lists
    df_candidates["party_list"] = df_candidates.apply(lambda row: dct_candidate_party_list.get(row["id_candidate"], np.nan), axis=1)
    logger.debug(f"Total number of considered candidates: {len(df_candidates)}")
    df_candidates.dropna(inplace=True)
    logger.debug(f"Number of candidates for both city councel and national assembly: {len(df_candidates)}")

    unique_party_list_L = list(df_candidates["party_list"].unique())
    unique_party_candidate_C = list(df_candidates["party_candidate"].unique())

    alliance_matrix_A = np.zeros((len(unique_party_candidate_C), len(unique_party_list_L)))

    df_count_by_alliance = df_candidates.groupby(by=["party_candidate", "party_list"]).count()
    for (party_candidate, party_list), row_count in df_count_by_alliance.iterrows():
        idx_party_candidate = unique_party_candidate_C.index(party_candidate)
        idx_party_list = unique_party_list_L.index(party_list)
        alliance_matrix_A[idx_party_candidate, idx_party_list] = row_count.values[0]

    df_alliance_matrix_A = pd.DataFrame(alliance_matrix_A)
    df_alliance_matrix_A.rename(inplace=True, columns=lambda x: unique_party_list_L[x], index=lambda x: unique_party_candidate_C[x])

    df_alliance_matrix_A.to_csv(path_output_alliances_matrix, index=True)


@click.command()
def main():
    logger.info("Creating sanitized list of candidates.")
    prepare_sanitized_list_of_candidates()
    logger.info("Creating sanitized list of city election lists.")
    prepare_sanitized_list_of_lists()
    logger.info("Building the alliance matrix")
    prepare_alliance_matrix()


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    project_dir = Path(os.environ["project_dir"])
    main()
