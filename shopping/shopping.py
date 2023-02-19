import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as csvfile:
        data = csv.DictReader(csvfile)

        evidence = []
        labels = []

        for row in data:
            evidence_values = []
            for key in row.keys():  # for all keys except revenue, append corrected table value to evidence
                if key == "Revenue":
                    if row[key] == "TRUE":
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    table_value = row[key]
                    evidence_values.append(get_num_value_of_key(table_value, key))
            evidence.append(evidence_values)

    return evidence, labels

def get_num_value_of_key(table_value, key):

    integers = ["Administrative", "Informational", "ProductRelated",
                "OperatingSystems", "Browser", "Region", "TrafficType"]

    floats = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
              "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if key in integers:
        return int(table_value)
    elif key == "Month":
        return months.index(table_value)
    elif key in floats:
        return float(table_value)
    elif key == "Weekend":
        if table_value == "TRUE":
            return 1
        else:
            return 0
    elif key == "VisitorType":
        if table_value == "Returning_Visitor":
            return 1
        else:
            return 0


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    n_sens = 0
    n_spec = 0
    for predicted, real in zip(predictions, labels):
        if real == 1:
            n_sens += 1
        else:
            n_spec += 1

        if predicted == real:
            if predicted == 1:
                sensitivity += 1
            else:
                specificity += 1

    return sensitivity/n_sens, specificity/n_spec


if __name__ == "__main__":
    main()
