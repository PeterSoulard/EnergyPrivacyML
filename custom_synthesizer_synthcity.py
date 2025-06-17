import pandas as pd

import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# CUSTOM_ARGUMENTS = [
#     {"n_iter": 10, "target_column": "income"},
#     {"n_iter": 20, "target_column": "income"},
#     {"n_iter": 30, "target_column": "income"},
#     {"n_iter": 40, "target_column": "income"},
#     {"n_iter": 50, "target_column": "income"},
# ]

CUSTOM_ARGUMENTS = [{"n_iter": n_iter, "target_column": "income"} for n_iter in range(1, 101, 1) for _ in range(10)]

# TODO: Replace this variable with the ISO code of the country in which
#       the experiments will be run. For example:
#       COUNTRY_ISO_CODE = "NLD"
COUNTRY_ISO_CODE = "NLD"

def generate_data(training_data: pd.DataFrame, arguments: dict) -> pd.DataFrame:
    """
    This function generates a synthetic dataset of the same size as the training data.

    Parameters:
    training_data (pd.DataFrame): The original data that the synthesizer model must be trained on.
    arguments (dict): A dictionary with configuration variables used in this run.

    Returns:
    pd.DataFrame: A synthetic data dataframe of the same size as the training data dataframe.
    """

    loader = GenericDataLoader(
        training_data,
        target_column=arguments.get("target_column"))
    
    syn_model = Plugins().get("adsgan",
                              n_iter = arguments.get("n_iter"),
                              n_iter_min = arguments.get("n_iter"),
                              patience = arguments.get("n_iter")
                              )

    syn_model.fit(loader)
    synthetic_data = syn_model.generate(len(training_data)).dataframe()

    return synthetic_data
