from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, \
                                  InferenceEvaluator
from scipy.spatial.distance import jensenshannon
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
import random
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

ORIGINALDATA_FILEPATH = "data/census_train.csv"
VALIDATIONDATA_FILEPATH = "data/census_val.csv"
EMISSIONS_FILEPATH = "emissions.csv"

# ===== ENERGY TRACKING =====


def start_tracking_energy(country_iso_code: str) -> OfflineEmissionsTracker:
    # Start measuring the new energy usage
    tracker = OfflineEmissionsTracker(country_iso_code=country_iso_code)
    tracker.start()
    return tracker


def stop_tracking_energy(tracker) -> None:
    # Stop the measurements
    tracker.stop()


def get_energy() -> list:
    df = pd.read_csv(EMISSIONS_FILEPATH)

    return df["gpu_energy"].to_list()

# ===== MEASURING PRIVACY RISKS =====


def measure_singling_out_risk(training_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame) -> float:
    evaluator = SinglingOutEvaluator(
        ori=training_data,
        syn=synthetic_data,
        n_attacks=min(1_000, len(training_data) // 10),
        max_attempts=10_000
    )

    evaluator.evaluate(mode="multivariate")
    risk, _ = evaluator.risk()
    return risk


def measure_linkability_risk(training_data: pd.DataFrame,
                             synthetic_data: pd.DataFrame) -> float:
    random.seed()

    columns = list(training_data.columns)

    left = random.sample(columns, len(columns) // 2)
    right = [col for col in columns if col not in left]

    aux_cols_tuple = [left, right]

    evaluator = LinkabilityEvaluator(
        ori=training_data,
        syn=synthetic_data,
        aux_cols=aux_cols_tuple,
        n_attacks=min(1_000, len(training_data) // 10),
        n_neighbors=3
    )

    evaluator.evaluate()
    risk, _ = evaluator.risk()
    return risk


def measure_inference_risk(training_data: pd.DataFrame,
                           synthetic_data: pd.DataFrame) -> float:
    random.seed()

    columns = list(training_data.columns)

    secret = random.choice(columns)
    aux_cols = [col for col in columns if col != secret]

    evaluator = InferenceEvaluator(
        ori=training_data,
        syn=synthetic_data,
        aux_cols=aux_cols,
        secret=secret,
        regression=None,
        n_attacks=min(1_000, len(training_data) // 10)
    )

    evaluator.evaluate()
    risk, _ = evaluator.risk()
    return risk


def measure_privacy_risk(training_data: pd.DataFrame,
                         synthetic_data: pd.DataFrame) -> tuple:
    singling_out = measure_singling_out_risk(training_data=training_data,
                                             synthetic_data=synthetic_data)
    linkability = measure_linkability_risk(training_data=training_data,
                                           synthetic_data=synthetic_data)
    inference = measure_inference_risk(training_data=training_data,
                                       synthetic_data=synthetic_data)

    return singling_out, linkability, inference

# ===== CALCULATING RESEMBLANCE AND ACCURACY =====


def resemblance(training_data: pd.DataFrame,
                synthetic_data: pd.DataFrame) -> float:

    distances = []

    for column in training_data.columns:

        column_data_real = training_data[column]
        column_data_synth = synthetic_data[column]

        categories_real = column_data_real.unique()
        categories_synth = column_data_synth.unique()

        categories = set(categories_real).union(set(categories_synth))

        probabilities_real = []
        probabilities_synth = []

        for i, category in enumerate(categories):
            probabilities_real.append(
                column_data_real.tolist().count(category))
            probabilities_synth.append(
                column_data_synth.tolist().count(category))

        total_real = sum(probabilities_real)
        total_synth = sum(probabilities_synth)

        for i in range(len(probabilities_real)):
            probabilities_real[i] /= total_real
            probabilities_synth[i] /= total_synth

        distances.append(jensenshannon(probabilities_real,
                                       probabilities_synth))

    return np.mean(distances)


def preprocess(data: pd.DataFrame) -> tuple:
    data["income"] = data["income"].str.replace("<=50K", "0")
    data["income"] = data["income"].str.replace(">50K", "1")
    data["income"] = data["income"].astype(int)

    labels = data["income"].copy()
    data = data.drop(["income"], axis = 1)

    continuous = ["age","capital_gain","capital_loss","hr_per_week"]
    data[continuous] = MinMaxScaler().fit_transform(data[continuous])

    categorial = [feat for feat in data.columns.values if feat not in continuous]
    data = pd.get_dummies(data, columns = categorial)

    binary_columns = data.columns[(data == False).all() | (data == True).all()]
    data[binary_columns] = data[binary_columns].astype(int)

    return data, labels


def accuracy(synthetic_data: pd.DataFrame, synthetic_labels: pd.DataFrame,
             validation_data: pd.DataFrame, validation_labels: pd.DataFrame) -> float:

    validation_data = validation_data[synthetic_data.columns]

    # KNN
    NeighbourModel = KNeighborsClassifier(n_neighbors = 2)
    NeighbourModel.fit(synthetic_data, synthetic_labels.values.ravel())
    report = classification_report(validation_labels, NeighbourModel.predict(validation_data), output_dict=True)
    knn = report['accuracy']

    # LR
    LogReg = LogisticRegression(max_iter=1000)
    LogReg.fit(synthetic_data, synthetic_labels.values.ravel())
    report = classification_report(validation_labels, LogReg.predict(validation_data), output_dict=True)
    lr = report['accuracy']

    # NN
    NeuralNet = Sequential()
    NeuralNet.add(Dense(32, activation = 'relu'))
    NeuralNet.add(Dense(64, activation = 'relu'))
    NeuralNet.add(Dense(32, activation = 'relu'))
    NeuralNet.add(Dense(8, activation = 'relu'))
    NeuralNet.add(Dense(1, activation = 'sigmoid'))
    NeuralNet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    binary_columns = synthetic_data.columns[(synthetic_data == False).all() | (synthetic_data == True).all()]
    synthetic_data[binary_columns] = synthetic_data[binary_columns].astype(int)
    NeuralNet.fit(synthetic_data, synthetic_labels, batch_size = 150, epochs = 50, verbose=0)
    validation_data = tensorflow.convert_to_tensor(validation_data, dtype=tensorflow.float32)
    validation_labels = tensorflow.convert_to_tensor(validation_labels, dtype=tensorflow.int32)
    _, nn = NeuralNet.evaluate(validation_data, validation_labels, verbose=0)

    return (knn + lr + nn) / 3

# ===== PLOTTING THE RESULTS =====


def plot_energy_privacy(energy_usage: list, privacy_risk: list, risk_type: str) -> None:
    plt.scatter(energy_usage, privacy_risk)
    plt.xlabel("Energy Consumption")
    plt.ylabel("Privacy Risk")
    plt.title(f"Privacy Risk vs Energy Consumption ({risk_type})")
    plt.savefig(f"plots/{risk_type}_risks_per_energy.png")
    plt.close()


def plot_energy_distance(energy_usage: list, distances: list) -> None:
    plt.scatter(energy_usage, distances)
    plt.xlabel("Energy Consumption")
    plt.ylabel("Jensen-Shannon distance")
    plt.title(f"Jensen-Shannon distance vs Energy Consumption")
    plt.savefig(f"plots/distance_per_energy.png")
    plt.close()


def plot_energy_accuracies(energy_usage: list, accuracies: list) -> None:
    plt.scatter(energy_usage, accuracies)
    plt.xlabel("Energy Consumption")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Energy Consumption")
    plt.savefig(f"plots/accuracy_per_energy.png")
    plt.close()


def plot_results(energy_usage: list, privacy_risks: dict, distances: list, accuracies: list) -> None:
    for risk_type in privacy_risks.keys():
        privacy_risk = privacy_risks.get(risk_type)
        plot_energy_privacy(energy_usage=energy_usage,
                            privacy_risk=privacy_risk,
                            risk_type=risk_type)

    plot_energy_distance(energy_usage, distances)

    plot_energy_accuracies(energy_usage, accuracies)


def main() -> None:
    # Import the custom synthesizer file
    import custom_synthesizer as cs

    country_iso_code = cs.COUNTRY_ISO_CODE

    if not cs.CUSTOM_ARGUMENTS:
        print("*** NO ARGUMENTS PROVIDED ***")
        return

    # Clear the emissions file
    if os.path.exists(EMISSIONS_FILEPATH):
        os.remove(EMISSIONS_FILEPATH)

    # Read the original data
    original_data = pd.read_csv(ORIGINALDATA_FILEPATH)

    # Read the validation data
    validation_data = pd.read_csv(VALIDATIONDATA_FILEPATH)
    validation_data, validation_labels = preprocess(validation_data)

    # A dict of lists to store the privacy measurements
    privacy_risks = {
        "singling_out": [],
        "linkability": [],
        "inference": []
    }

    n_configurations = len(cs.CUSTOM_ARGUMENTS)

    distances = []
    accuracies = []

    # Loop over each arguments configuration.
    for i, arguments in enumerate(cs.CUSTOM_ARGUMENTS):
        print(f"Running configuration {i+1}/{n_configurations}:\n\t{arguments}\n")

        sample_size = min(10_000, original_data.shape[0] // 10)

        # Randomly sample the training data from the original data
        training_data = original_data.sample(n=sample_size, replace=False)

        # Start tracking energy
        tracker = start_tracking_energy(country_iso_code)

        # Generate the data
        synthetic_data = cs.generate_data(training_data=training_data,
                                          arguments=arguments)

        # Stop tracking energy
        stop_tracking_energy(tracker)

        singling_out, linkability, inference = \
            measure_privacy_risk(training_data=training_data,
                                 synthetic_data=synthetic_data)

        privacy_risks.get("singling_out").append(singling_out)
        privacy_risks.get("linkability").append(linkability)
        privacy_risks.get("inference").append(inference)

        distances.append(resemblance(training_data, synthetic_data))

        synthetic_data, synthetic_labels = preprocess(synthetic_data)
        accuracies.append(accuracy(synthetic_data, synthetic_labels, validation_data, validation_labels))

    energy_usage = get_energy()

    print(f"\tEnergy usage:\n{energy_usage}\n\tPrivacy risks:\n{privacy_risks}")
    print(f"\tResemblances:\n{distances}\n\tAccuracies:\n{accuracies}")

    plot_results(energy_usage, privacy_risks, distances, accuracies)


if __name__ == "__main__":
    main()
