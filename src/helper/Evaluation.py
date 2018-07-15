from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import difflib


def get_model_accuracy(actual_result, predicted_result):
    return accuracy_score(actual_result, predicted_result, normalize=True)


def show_confusion_matrix(actual_result, predicted_result):
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actual_result, predicted_result).ravel()
    return true_negative, false_positive, false_negative, true_positive


def show_classification_report(actual_result, predicted_result, target_names):
    return classification_report(actual_result, predicted_result, target_names)


def calculate_accuracy(actual_results, predicted_results):
    correct_count = 0.0

    for actual, predicted in zip(actual_results, predicted_results):
        correct_count += difflib.SequenceMatcher(None, sorted(actual), sorted(predicted)).ratio()

    print("Total Prediction : ", len(predicted_results))
    print("Total Correct Prediction : ", correct_count)

    return correct_count / len(predicted_results)

