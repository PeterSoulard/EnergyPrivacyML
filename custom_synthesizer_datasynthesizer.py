import pandas as pd
import numpy as np

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
import os

CUSTOM_ARGUMENTS = [{"epsilon": epsilon, "k": k}
                    for epsilon in np.linspace(0.1, 2.0, 10)
                    for k in range(1, 3) for _ in range(30)]

# TODO: Replace this variable with the ISO code of the country in which
#       the experiments will be run. For example:
#       COUNTRY_ISO_CODE = "NLD"
COUNTRY_ISO_CODE = "NLD"


def generate_data(training_data: pd.DataFrame,
                  arguments: dict) -> pd.DataFrame:
    """
    This function generates a synthetic dataset of the same size as the
    training data.

    Parameters:
    training_data (pd.DataFrame): The original data that the synthesizer model
        must be trained on.
    arguments (dict): A dictionary with configuration variables used in
        this run.

    Returns:
    pd.DataFrame: A synthetic data dataframe of the same size as the training
        data dataframe.
    """

    training_data.to_csv("temp.csv")

    describer = DataDescriber(category_threshold=42)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file="temp.csv",
        epsilon=arguments.get("epsilon"),
        k=arguments.get("k")
    )

    describer.save_dataset_description_to_file("description_temp.json")

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(
        len(training_data),
        "description_temp.json"
    )

    generator.save_synthetic_data("synthetic_data_temp.csv")

    synthetic_data = pd.read_csv("synthetic_data_temp.csv")

    synthetic_data = synthetic_data.iloc[:, 1:]

    synthetic_data.columns = training_data.columns

    for f in ["temp.csv", "description_temp.json",
              "description.json", "synthetic_data_temp.csv"]:
        if os.path.exists(f):
            os.remove(f)

    return synthetic_data
