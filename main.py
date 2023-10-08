import torch
from torch.utils.data import DataLoader
from data.dataset import BreastCancerDataset
from models.attention import AdditiveMIL, AttentionMIL
from models.baseline import Baseline
from models.losses import ranking_loss
from sklearn.model_selection import GroupKFold
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
import pickle
import numpy as np
import time, os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--num_splits", default=1, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_bags", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--model", default="AdditiveMIL", type=str, choices=["AdditiveMIL", "AttentionMIL", "Baseline"])
    parser.add_argument("--results_dir", default="results/", type=str)
    opts = parser.parse_args()
    return opts


def create_data_loaders(dataset, train_idx, val_idx, batch_size):
    """Create training and validation data loaders from indices."""
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, loss_function):
    """Train model for one epoch and return loss, labels, and predictions."""
    train_loss, train_labels, train_predictions = 0, [], []
    model.train()
    for batch in train_loader:
        # Extract data from batch
        features, label = batch["features"], batch["label"]
        
        optimizer.zero_grad()
        prediction = model(features)
        
        # Compute and backpropagate loss
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()

        # Accumulate results
        train_loss += loss.item()
        train_labels.append(label)
        train_predictions.append(prediction)
    
    # convert in numpy array
    train_loss = train_loss/len(train_loader)
    train_labels = torch.cat(train_labels, dim=0).squeeze().detach().numpy()
    train_predictions = torch.cat(train_predictions, dim=0).squeeze().detach().numpy()

    return train_loss, train_labels, train_predictions


def evaluate(model, val_loader, loss_function):
    """Evaluate model on validation set and return loss, labels, and predictions."""
    model.eval()
    val_loss, val_labels, val_predictions = 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            features, label = batch["features"], batch["label"]
            prediction = model(features)
            loss = loss_function(prediction, label)

            val_labels.append(label)
            val_predictions.append(prediction)
            val_loss += loss.item()
    
    # convert in numpy array
    val_loss = val_loss/len(val_loader)
    val_labels = torch.cat(val_labels, dim=0).squeeze().numpy()
    val_predictions = torch.cat(val_predictions, dim=0).squeeze().numpy()

    return val_loss, val_labels, val_predictions


if __name__ == '__main__':     
    opts = get_args()
    data_dir = Path(opts.data_dir)

    if opts.model == "Baseline":
        result_dir = Path(opts.results_dir) / "baseline"
    elif opts.model == "AdditiveMIL":
        result_dir = Path(opts.results_dir) / "additive"
    elif opts.model == "AttentionMIL":
        result_dir = Path(opts.results_dir) / "attention"

    
    dataset = BreastCancerDataset(data_dir, mode="train", num_bags=opts.num_bags)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    results_training = {}
    results_validation = {}
    val_auc = []

    all_data = [dataset[i]["features"] for i in range(len(dataset))]
    all_labels = [dataset[i]["label"] for i in range(len(dataset))]
    all_samples = [dataset[i]["sample"] for i in range(len(dataset))]
    all_centers = [dataset[i]["center"] for i in range(len(dataset))]

    for seed in range(opts.num_splits):
        group_kfold = GroupKFold(n_splits=opts.num_folds)
        split = group_kfold.split(X=all_labels, y=all_labels, groups=all_samples)

        for fold, (train_idx, val_idx) in enumerate(split):
            train_loader, val_loader = create_data_loaders(dataset, train_idx, val_idx, batch_size=opts.batch_size)

            # initialize results dict
            if seed not in results_validation: results_validation[seed] = {}
            if seed not in results_training: results_training[seed] = {}
            if fold not in results_validation[seed]: results_validation[seed][fold] = {}
            if fold not in results_training[seed]: results_training[seed][fold] = {}

            # Init model and optimizer at each fold
            model_class = getattr(__import__(__name__), opts.model)
            model = model_class()   
            loss_function = torch.nn.BCELoss() 
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
            
            for epoch in range(opts.num_epochs):

                # Train model 
                train_loss, train_labels, train_predictions = train_one_epoch(model, train_loader, optimizer, loss_function)

                results_training[seed][fold][epoch] = {
                    "Loss": train_loss,
                    "Labels": train_labels,
                    "Predictions": train_predictions
                }
                
                # Evaluate model 
                val_loss, val_labels, val_predictions = evaluate(model, val_loader, loss_function)

                results_validation[seed][fold][epoch] = {
                    "Loss": val_loss,
                    "Labels": val_labels,
                    "Predictions": val_predictions
                }

                # Print results for each epoch
                print(
                    f"Seed {seed}, Fold {fold+1}, Epoch {epoch+1} : "
                    f"Validation Loss: {val_loss:.4f}, "
                    f"Validation AUC: {roc_auc_score(val_labels, val_predictions):.4f}, "
                )

            val_auc.append(roc_auc_score(val_labels, val_predictions))

    print(f"Validation AUC -- mean: {np.mean(val_auc):.2f}, std: {np.std(val_auc):.2f}")

    run_id = int(time.time())
    os.makedirs(result_dir / f'run_{run_id}', exist_ok=True)
    run_id_dir = Path(result_dir / f'run_{run_id}')

    with open(run_id_dir / "results_validation.pkl", "wb") as file:
        pickle.dump(results_validation, file)

    with open(run_id_dir / "results_training.pkl", "wb") as file:
        pickle.dump(results_training, file)

    print(f"results saved in {run_id_dir}")

    torch.save(model.state_dict(),run_id_dir / 'model_weights.pth')
    print(f"model saved in {run_id_dir}")
