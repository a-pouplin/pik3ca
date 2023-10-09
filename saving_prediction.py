import torch
import pandas as pd
from models.attention import AdditiveMIL, AttentionMIL
from models.baseline import Baseline
from data.dataset import BreastCancerDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from utils.metrics import load_json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--results_dir", default="results/", type=str)
    parser.add_argument("--run_id", default="baseline_70_run_91043", type=str)
    opts = parser.parse_args()
    return opts

def load_testset(data_dir):
    """
    Loads test set from data directory
    """
    testset = BreastCancerDataset(data_dir, mode="test")
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    return test_loader

def save_predictions(model, test_loader, submission_path):
    """
    Saves predictions to csv file
    """
    assert(len(test_loader) == 1) # we get all the test set in one batch
    model.eval()
    with torch.no_grad():
        testset = next(iter(test_loader))
        features = testset["features"]
        sample = testset["sample"]
        prediction = model(features)
    
    
    submission = pd.DataFrame({"Sample ID": sample, 
                               "Target": prediction.numpy()}).sort_values("Sample ID")
     
    # sanity checks
    assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
    assert submission.shape == (149, 2), "Your submission file must be of shape (149, 2)"
    assert list(submission.columns) == ["Sample ID","Target",], "Your submission file must have columns `Sample ID` and `Target`"

    print(f"Saving predictions in {submission_path}")
    submission.to_csv(submission_path, index=None)
    print("Done!")

if __name__ == '__main__':
    opts = get_args()
    data_dir = Path(opts.data_dir)
    result_dir = Path(opts.results_dir)

    # load test data
    test_loader = load_testset(data_dir)

    # directory where model is saved
    run_dir = result_dir / f"runs/{opts.run_id}"
    params = load_json(run_dir / 'params.json')

    # load model structure and weights
    model_class = getattr(__import__(__name__), params['model'])
    model = model_class(hidden_dim=params['hidden_dim']) 
    model.load_state_dict(torch.load(run_dir / 'model_weights.pth'))
    
    # save predictions
    submission_path = result_dir / f"submissions/{opts.run_id}.csv"
    save_predictions(model, test_loader, submission_path)
