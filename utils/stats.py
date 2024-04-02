import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

def format_metric(data):
    """Return mean and stdev in formatted string."""
    return f"{np.mean(data):.3f} ± {np.std(data):.3f}"

def print_table(sensitivities, specificities, auc_rocs, class_names):
    max_class_name_length = max([len(name) for name in class_names])

    print(f"{'Class name':<{max_class_name_length}} \t| Sensitivity\t| Specificity\t| AUC-ROC")
    print("-" * (max_class_name_length + 76))
    for i, class_name in enumerate(class_names):
        sensitivity_str = format_metric(sensitivities[:, i])
        specificity_str = format_metric(specificities[:, i])
        auc_roc_str = format_metric(auc_rocs[:, i])
        
        print(f"{class_name:<{max_class_name_length}} \t\t| {sensitivity_str}\t| {specificity_str}\t| {auc_roc_str}")
    print("\n")


def calculate_metrics(Y, test_probs, num_classes):
    """
    Calculate sensitivities, specificities, and AUCs for a single fold and an arbitrary number of classes.

    :param Y: True labels.
    :param test_preds: Test predictions for the fold.
    :param test_probs: Test probabilities for the fold.
    :param num_classes: Number of classes.
    :return: Tuple of lists containing sensitivities, specificities, and AUCs for each class.
    """
    class_sensitivities = []
    class_specificities = []
    class_AUCs = []

    test_preds = np.argmax(test_probs, axis=-1)

    # Create a confusion matrix
    cm = confusion_matrix(Y, test_preds)

    for i in range(num_classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp

        # Calculate True Positive Rate (Sensitivity) and False Positive Rate
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        class_sensitivities.append(tpr)
        class_specificities.append(1 - fpr)

        # Calculate AUC for each class
        bin_true = [1 if x == i else 0 for x in Y]
        class_AUCs.append(roc_auc_score(bin_true, test_probs[:, i]))

    f1 = f1_score(Y, test_preds, average='macro')

    return class_sensitivities, class_specificities, class_AUCs, f1