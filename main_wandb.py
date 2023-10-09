import torch
from torch.utils.data import DataLoader
from data.dataset import BreastCancerDataset
from models.attention import AdditiveMIL, AttentionMIL, GatedAttentionMIL
from models.baseline import Baseline
from sklearn.model_selection import GroupKFold
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
from utils.metrics import create_folder_run_id, save_pickle, save_json
import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_bags", default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--hidden_dim", default=64, type=float)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--model", default="AdditiveMIL", type=str, 
                        choices=["AdditiveMIL", "AttentionMIL", "GatedAttentionMIL", "Baseline"])
    parser.add_argument("--results_dir", default="results/runs/", type=str)
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

def experiment():
    run = wandb.init(project="pik3ca")

    opts = get_args()
    data_dir = Path(opts.data_dir)

    # hyperparams
    opts.lr = wandb.config.lr
    opts.batch_size = wandb.config.batch_size
    opts.model = wandb.config.model
    opts.hidden_dim = wandb.config.hidden_dim
    opts.fold = wandb.config.fold
    
    dataset = BreastCancerDataset(data_dir, mode="train", num_bags=opts.num_bags)
    all_data = [dataset[i]["features"] for i in range(len(dataset))]
    all_labels = [dataset[i]["label"] for i in range(len(dataset))]
    all_patients = [dataset[i]["patient"] for i in range(len(dataset))]

    group_kfold = GroupKFold(n_splits=opts.num_folds)
    split = group_kfold.split(X=all_data, y=all_labels, groups=all_patients)

    train_idx, val_idx = list(split)[opts.fold]
    train_loader, val_loader = create_data_loaders(dataset, train_idx, val_idx, batch_size=opts.batch_size)

    # Init model and optimizer at each fold
    model_class = getattr(__import__(__name__), opts.model)
    model = model_class(hidden_dim=opts.hidden_dim)   
    loss_function = torch.nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    
    for epoch in range(opts.num_epochs):

        # Train model 
        train_loss, train_labels, train_predictions = train_one_epoch(model, train_loader, optimizer, loss_function)
        train_auc = roc_auc_score(train_labels, train_predictions)

        # Evaluate model 
        val_loss, val_labels, val_predictions = evaluate(model, val_loader, loss_function)
        val_auc = roc_auc_score(val_labels, val_predictions)

        # Print results for each epoch
        print(
            f"Epoch {epoch+1} | "
            f"Training loss: {train_loss:.2f}, "
            f"auc: {train_auc:.2f} | "
            f"Validation loss: {val_loss:.2f}, "
            f"auc: {val_auc:.2f}, "
        )

        wandb.log({"val_auc": val_auc, "val_loss": val_loss, "train_auc": train_auc, "train_loss": train_loss})



if __name__ == '__main__':     
    

    # Define sweep config
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_auc"},
        "parameters": {
            "batch_size": {"values": [8, 16, 32]},
            "hidden_dim": {"values": [8, 16, 32, 64, 128]},
            "lr": {"max": 0.001, "min": 0.0001},
            "model": {"values": ["AdditiveMIL"]},
            "fold": {"values": [1]},
        },
    }

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="pik3ca")
    wandb.agent(sweep_id, function=experiment, count=50)







