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
from utils.metrics import create_folder_run_id, save_pickle, save_json, set_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=16, type=int) 
    parser.add_argument("--num_bags", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=32, type=float)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--model", default="Baseline", type=str, 
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


if __name__ == '__main__': 
    seed = 0
    set_seed(seed)
    
    opts = get_args()
    data_dir = Path(opts.data_dir)
    result_dir = Path(opts.results_dir)
    
    dataset = BreastCancerDataset(data_dir, mode="train", num_bags=opts.num_bags)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size)

    results_training = {}
    results_validation = {}
    val_aucs = []

    all_data = [dataset[i]["features"] for i in range(len(dataset))]
    all_labels = [dataset[i]["label"] for i in range(len(dataset))]
    all_patients = [dataset[i]["patient"] for i in range(len(dataset))]
    # all_centers = [dataset[i]["center"] for i in range(len(dataset))]

    group_kfold = GroupKFold(n_splits=opts.num_folds)
    split = group_kfold.split(X=all_data, y=all_labels, groups=all_patients)
    for fold, (train_idx, val_idx) in enumerate(split):
        all_patients = np.array(all_patients)
        assert set(all_patients[train_idx]).isdisjoint(set(all_patients[val_idx])), "Train and val sets should be disjoint"
        train_loader, val_loader = create_data_loaders(dataset, train_idx, val_idx, batch_size=opts.batch_size)

        # initialize results dict
        if seed not in results_validation: results_validation[seed] = {}
        if seed not in results_training: results_training[seed] = {}
        if fold not in results_validation[seed]: results_validation[seed][fold] = {}
        if fold not in results_training[seed]: results_training[seed][fold] = {}

        # Init model and optimizer at each fold
        model_class = getattr(__import__(__name__), opts.model)
        model = model_class(hidden_dim=opts.hidden_dim)   
        loss_function = torch.nn.BCELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        
        for epoch in range(opts.num_epochs):

            # Train model 
            train_loss, train_labels, train_predictions = train_one_epoch(model, train_loader, optimizer, loss_function)
            train_auc = roc_auc_score(train_labels, train_predictions)

            results_training[seed][fold][epoch] = {
                "Loss": train_loss,
                "Labels": train_labels,
                "Predictions": train_predictions
            }
            
            # Evaluate model 
            val_loss, val_labels, val_predictions = evaluate(model, val_loader, loss_function)
            val_auc = roc_auc_score(val_labels, val_predictions)

            results_validation[seed][fold][epoch] = {
                "Loss": val_loss,
                "Labels": val_labels,
                "Predictions": val_predictions
            }

            # Print results for each epoch
            print(
                f"Fold {fold}, Epoch {epoch+1} | "
                f"Training loss: {train_loss:.2f}, "
                f"auc: {train_auc:.2f} | "
                f"Validation loss: {val_loss:.2f}, "
                f"auc: {val_auc:.2f}, "
            )

        val_aucs.append(val_auc)

    print(f"Validation AUC -- mean: {np.mean(val_aucs):.2f}, std: {np.std(val_aucs):.2f}")

    # Save results and params
    run_id_dir = create_folder_run_id(result_dir, opts.model, int(np.mean(val_aucs)*100))
    save_json(run_id_dir / "params.json", vars(opts))
    save_pickle(run_id_dir / "results_validation.pkl", results_validation)
    save_pickle(run_id_dir / "results_training.pkl", results_training)
    print(f"results saved in {run_id_dir}")

    # Save model weights and structure
    torch.save(model.state_dict(), run_id_dir / 'model_weights.pth')
    with open(run_id_dir /'model_structure.txt', 'w') as f: f.write(str(model))
    print(f"model saved in {run_id_dir}")
