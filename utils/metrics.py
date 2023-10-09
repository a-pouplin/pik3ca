from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle, os, json
from datetime import datetime 
from pathlib import Path
import random


def load_pickle(path):
    with open(path, "rb") as file:
        myfile = pickle.load(file)
    return myfile


def save_pickle(path, myfile):
    with open(path, "wb") as file:
        pickle.dump(myfile, file)


def load_json(path):
    with open(path, "r") as file:
        myfile = json.load(file)
    return myfile


def save_json(path, myfile):
    with open(path, "w") as file:
        json.dump(myfile, file)


def to_npy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    return tensor


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def create_folder_run_id(result_dir, model, metric):
    # Create a folder for the run_id "[model]_[metric]_run_[time]"
    now = datetime.now()
    time_id = now.strftime('%d%H%M')
    time_id = int(time_id)
    run_id = f'{model.lower()}_{metric}_run_{time_id}'
    os.makedirs(result_dir / run_id, exist_ok=True)
    return Path(result_dir / run_id)
    

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


def compute_metrics_per_epoch(results, metric_func, rounding=False):
    """
    Computes a specified metric from the results_dict.
    
    :param results_dict: Dictionary containing results for each fold.
    :param metric_func: scikit-learn function to compute a specific metric.
    :return: mean and std of the metric, in array for each epoch.
    """
    num_epochs =  len(list(list(results.values())[0].values())[0])
    mean_epochs = []
    std_epochs = []
    metric = []

    for epoch in range(num_epochs):
        if metric_func == "loss":
            metric = [results[seed][fold][epoch]["Loss"] for seed in results for fold in results[seed]]

        else:
            for seed in results:
                for fold in results[seed]:
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
    epoch = len(list(list(results.values())[0].values())[0]) - 1
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
    # train_additive = load_pickle("../results/runs/additivemil_67_run_90956/results_training.pkl")
    val_additive = load_pickle("results/runs/additivemil_67_run_90956/results_validation.pkl")
    val_mean, val_std = compute_metrics_per_epoch(val_additive, roc_auc_score, rounding=False)
    print(val_mean)
    print(len(val_mean))
