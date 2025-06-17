import pandas as pd

# TODO: Import your synthesizer here
# import ...

# TODO: Replace this variable with a list, where each element is a
#       dictionary containing a configuration of arguments. For example:
#       CUSTOM_ARGUMENTS = [
#                              {
#                                  "synthesizer_mode": "univariate",
#                                  "target_column": "education",
#                                  "iterations": 200
#                              },
#                              {
#                                  "synthesizer_mode": "multivariate",
#                                  "target_column": "income",
#                                  "iterations": 5000
#                              },
#                          ]
CUSTOM_ARGUMENTS = None

# TODO: Replace this variable with the ISO code of the country in which
#       the experiments will be run. For example:
#       COUNTRY_ISO_CODE = "NLD"
COUNTRY_ISO_CODE = None

def generate_data(training_data: pd.DataFrame, arguments: dict) -> pd.DataFrame:
    """
    This function generates a synthetic dataset of the same size as the training data.

    Parameters:
    training_data (pd.DataFrame): The original data that the synthesizer model must be trained on.
    arguments (dict): A dictionary with configuration variables used in this run.

    Returns:
    pd.DataFrame: A synthetic data dataframe of the same size as the training data dataframe.
    """

    # TODO: Replace this by code that generates and returns the synthetic data.
    return training_data
