import torch
import pandas as pd
from models.attention import AdditiveMIL
from models.baseline import Baseline
from data.dataset import BreastCancerDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--results_dir", default="results/", type=str)
    parser.add_argument("--model_dir", default="results/baseline/", type=str)
    opts = parser.parse_args()
    return opts

def load_testset(data_dir):
    """
    Loads test set from data directory
    """
    testset = BreastCancerDataset(data_dir, mode="test")
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    return test_loader

def save_predictions(model, test_loader, result_dir):
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

    print("Saving predictions to csv file...")
    submission.to_csv(result_dir/"test_baseline.csv", index=None)
    print("Done!")

if __name__ == '__main__':
    opts = get_args()
    data_dir = Path(opts.data_dir)
    result_dir = Path(opts.results_dir)
    model_dir = Path(opts.model_dir)

    test_loader = load_testset(data_dir)

    # model = AdditiveMIL()
    model = Baseline()
    model.load_state_dict(torch.load(model_dir / 'model_weights.pth'))
    
    save_predictions(model, test_loader, result_dir)