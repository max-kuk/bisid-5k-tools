import os


import numpy as np
import argparse


from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# set random seed
seed = 0
np.random.seed(seed)


def set_size(width: float, fraction: int = 1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "two-column":
        # width_pt = 8.25 * 28.346
        width_pt = 3.25 * 72.27
    elif width == "one-column":
        # width_pt = 17.46 * 28.346
        width_pt = 6.875 * 72.27
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def load_spectra(src_dir):
    data = []
    labels = []
    for i, c in enumerate(classes):
        print(f"Loading data for class {c}")
        for f in os.listdir(os.path.join(src_dir, c)):
            spectra = np.load(os.path.join(src_dir, c, f))

            data.append(spectra)
            labels.append(i)

    return np.array(data), np.array(labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, default="dataset_segmented/spectroscopic_dataset"
    )
    parser.add_argument(
        "--save_dir", type=str, default="paper_logs_1d", help="save results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multilayer_perceptron",
        help="model name (logistic_regression, decision_tree, random_forest, multilayer_perceptron)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)

    args = parser.parse_args()

    # create target folder
    os.makedirs(args.save_dir, exist_ok=True)

    classes = {
        "Polygonum aviculare",
        "Stellaria media",
        "Silene dioica",
        "Alopecurus pratensis",
        "Brassica napus",
        "Geranium robertianum",
        "Trifolium pratense",
        "Galeopsis tetrahit",
        "Alopecurus myosuroides",
        "Bistorta officinalis",
    }

    models = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "multilayer_perceptron": MLPClassifier,
    }

    model = models[args.model]()

    # load data
    data, labels = load_spectra(args.src_dir)

    n_runs = 3

    f1_scores = np.zeros(n_runs)
    precision_scores = np.zeros(n_runs)
    recall_scores = np.zeros(n_runs)
    accuracy_scores = np.zeros(n_runs)

    for i in range(n_runs):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.3,  # random_state=seed
        )

        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)

        # calculate metrics
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, target_names=classes)

        # multiply by 100 to get percentage
        f1 *= 100
        precision *= 100
        recall *= 100
        accuracy *= 100

        # round to 2 decimal places
        f1 = round(f1, 2)
        precision = round(precision, 2)
        recall = round(recall, 2)
        accuracy = round(accuracy, 2)

        f1_scores[i] = f1
        precision_scores[i] = precision
        recall_scores[i] = recall
        accuracy_scores[i] = accuracy

        print(f"Run {i+1}")
        print(f"F1: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(clf_report)

    # calculate mean and std
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    precision_mean = np.mean(precision_scores)
    precision_std = np.std(precision_scores)

    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)

    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)

    # round to 2 decimal places
    f1_mean = round(f1_mean, 2)
    f1_std = round(f1_std, 2)

    precision_mean = round(precision_mean, 2)
    precision_std = round(precision_std, 2)

    recall_mean = round(recall_mean, 2)
    recall_std = round(recall_std, 2)

    accuracy_mean = round(accuracy_mean, 2)
    accuracy_std = round(accuracy_std, 2)

    print(f"Accuracy mean: {accuracy_mean}±{accuracy_std}")
    print(f"Precision mean: {precision_mean}±{precision_std}")
    print(f"Recall mean: {recall_mean}±{recall_std}")
    print(f"F1 mean: {f1_mean}±{f1_std}")

    # save metrics to file
    with open(f"{args.save_dir}/classification_report_{args.model}.txt", "w") as f:
        f.write(f"Accuracy mean: {accuracy_mean}±{accuracy_std}\n")
        f.write(f"Precision mean: {precision_mean}±{precision_std}\n")
        f.write(f"Recall mean: {recall_mean}±{recall_std}\n")
        f.write(f"F1 mean: {f1_mean}±{f1_std}\n")
