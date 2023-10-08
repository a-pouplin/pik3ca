from sklearn.metrics import roc_auc_score, recall_score, precision_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle

def load_pickle(path):
    with open(path, "rb") as file:
        myfile = pickle.load(file)
    return myfile

def to_npy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    return tensor


def compute_metric_for_folds(results_dict, metric_func):
    """
    Computes a specified metric for each fold in the results_dict.
    
    :param results_dict: Dictionary containing results for each fold.
    :param metric_func: scikit-learn function to compute a specific metric.
    :return: A dictionary where each fold contains an array of metric values.
    """
    metric_values = {}
    
    for fold, results in results_dict.items():
        fold_metric_values = []
        if metric_func == "loss":
            metric_values[fold] = results['loss']
        else:
            for labels, predictions in zip(results['labels'], results['predictions']):
                labels = to_npy(labels)
                predictions = to_npy(predictions)
                fold_metric_values.append(metric_func(labels, predictions))
            metric_values[fold] = fold_metric_values        
    return metric_values


def plot_k_folds_train_and_val(values_training, values_validation, y_label="axis"):
    num_folds = len(values_training)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plotting for Training
    for i in range(num_folds):
        axes[0].plot(values_training[i], label=f"Fold {i+1}")
    axes[0].set_title("Training")
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel(y_label)
    axes[0].legend()

    # Plotting for Validation
    for i in range(num_folds):
        axes[1].plot(values_validation[i], label=f"Fold {i+1}")
    axes[1].set_title("Validation")
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel(y_label)
    axes[1].legend()

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve_for_fold(results_validation):
    """
    Plots the ROC curve for a specific fold.
    
    :param results_validation: Dictionary containing results for each fold.
    :param fold_number: The fold number for which to plot the ROC curve.
    """
    num_fold = len(results_validation)

    plt.figure()
    for i in range(num_fold):
        labels = results_validation[i]['labels'][-1]
        predictions = results_validation[i]['predictions'][-1]
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve of fold {0} (area = {1:0.2f})'.format(i+1, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def compute_metrics_per_epoch(results, metric_func, rounding=True):
    """
    Computes a specified metric from the results_dict.
    
    :param results_dict: Dictionary containing results for each fold.
    :param metric_func: scikit-learn function to compute a specific metric.
    :return: mean and std of the metric, in array for each epoch.
    """
    num_epochs = len(results[0][0])
    mean_epochs = []
    std_epochs = []
    metric = []

    for epoch in range(num_epochs):
        if metric_func == "loss":
            metric = [results[seed][fold][epoch]["Loss"] for seed in results for fold in results[seed]]

        else:
            for seed in results:
                for fold in results[seed]:
                    # Get labels and predictions for the last epoch 
                    labels = results[seed][fold][epoch]["Labels"]
                    predictions = results[seed][fold][epoch]["Predictions"]
                    if rounding:
                        predictions = np.round(predictions)
                    metric.append(metric_func(labels, predictions))

        mean_epochs.append(np.mean(metric))
        std_epochs.append(np.std(metric))
     
    return mean_epochs, std_epochs

def plot_mean_std(ax, results_mean, results_std, y_label="axis", color='blue', label=None): 
    epochs = range(len(results_mean))
    ax.plot(epochs, results_mean, color=color, label=label)
    ax.fill_between(epochs, 
                    [mean - std for mean, std in zip(results_mean, results_std)], 
                    [mean + std for mean, std in zip(results_mean, results_std)], 
                    color=color, alpha=0.1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)
    return ax

def plot_roc_curve_mean_std(ax, results, label=None, color='orange'):
    """
    Plots the ROC curve on average
    
    :param results: Dictionary containing results for each fold.
    :param fold_number: The fold number for which to plot the ROC curve.
    """
    epoch = len(results[0][0]) - 1
    fprs, tprs, thresholds, aucs = [], [], [], []
    for seed in results:
        for fold in results[seed]:
            # Get labels and predictions for the last epoch 
            true_labels = results[seed][fold][epoch]["Labels"]
            predicted_scores = results[seed][fold][epoch]["Predictions"]
            
            fpr, tpr, threshold = roc_curve(true_labels, predicted_scores)
            aucs.append(roc_auc_score(true_labels, predicted_scores))
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)

    # Define standard FPR values (from 0 to 1)
    standard_fpr = np.linspace(0, 1, 100)

    interpolated_tprs = []
    for i in range(len(fprs)):
        interp_tpr = interp1d(fprs[i], tprs[i], kind='linear', fill_value=(0, 1), bounds_error=False)(standard_fpr)
        interpolated_tprs.append(interp_tpr)

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)

    ax.plot(standard_fpr, mean_tpr, color=color, label=label)
    ax.fill_between(standard_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.1)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return ax, aucs


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    result_dir = Path("results/baseline/")

    # with open(result_dir / "results_training.pkl", "rb") as file:
    #     results_training = pickle.load(file)

    # with open(result_dir / "results_validation.pkl", "rb") as file:
    #     results_validation = pickle.load(file)

    with open(result_dir / "results_test.pkl", "rb") as file:
        results_validation = pickle.load(file)


    # auc_values_training = compute_metric_for_folds(results_training, roc_auc_score)
    # auc_values_validation = compute_metric_for_folds(results_validation, roc_auc_score)

    # loss_values_training = compute_metric_for_folds(results_training, "loss")
    # loss_values_validation = compute_metric_for_folds(results_validation, "loss")

    # plot_k_folds_train_and_val(auc_values_training, auc_values_validation, y_label="AUC")
    # plot_k_folds_train_and_val(loss_values_training, loss_values_validation, y_label="Loss")
    # plot_roc_curve_for_fold(results_validation)

    auc_mean, auc_std = compute_metrics_per_epoch(results_validation, roc_auc_score, rounding=False)
    sens_mean, sens_std = compute_metrics_per_epoch(results_validation, recall_score)
    loss_mean, loss_std = compute_metrics_per_epoch(results_validation, "loss")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0] = plot_mean_std(axes[0], auc_mean, auc_std, label="AUC", color='orange')
    axes[0] = plot_mean_std(axes[0], sens_mean, sens_std, label="Sensitivity", color='purple')
    axes[1] = plot_mean_std(axes[1], loss_mean, loss_std, y_label="Loss")
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax, aucs = plot_roc_curve_mean_std(ax, results_validation)
    print(aucs)
    plt.title("ROC curve | AUC mean: {:.2f} (std: {:.2f})".format(np.mean(aucs), np.std(aucs)))
    plt.tight_layout()
    plt.show()
