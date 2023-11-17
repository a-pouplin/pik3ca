import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import os


def download_data():
    # url not working, need to download manually 
    data_dir = Path("data/")
    challenge_url = "https://challengedata.ens.fr/participants/challenges/98/download/"

    data = {'train': {'url': challenge_url + "x-train.zip",
                      'output_dir': data_dir / "train_input"},
            'test': {'url': challenge_url + "x-test",
                     'output_dir': data_dir / "test_input"}}
    
    for mode in ["train", "test"]:
        url = data[mode]["url"]
        output_dir = data[mode]["output_dir"]
        if os.path.exists(output_dir):
            print(f"Data already exists in {output_dir}. Skipping download.")
        else:
            print(f"Downloading {mode} data to {output_dir}.")
            print(f"URL: {url}")
            command = ["wget", "-P", output_dir, url]
            subprocess.run(command, capture_output=True, text=True)
            print("Done.")

class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, mode="train", num_bags=None):
        assert mode in ["train", "test"], "Mode should be either 'train' or 'test'"
        
        self.data_dir = data_dir
        self.mode = mode
        self.features_dir = self.data_dir / f"{mode}_input" / "moco_features"
        self.num_bags = num_bags
        
        df = pd.read_csv(self.data_dir / "supplementary_data" / f"{mode}_metadata.csv")
        
        if mode == "train":
            y = pd.read_csv(self.data_dir / "train_output.csv")
            df = df.merge(y, on="Sample ID")

        elif mode == "test":
            df["Target"] = -1
        
        self.samples = df[["Sample ID", "Target", "Center ID", "Patient ID"]].values
        self.labels = df["Target"].values
        self.centers = df["Center ID"].values

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label, center, patient = self.samples[idx]
        _features = np.load(self.features_dir / sample)
        coordinates, features = _features[:, :3], _features[:, 3:]

        # choose randomly a subset of tiles (feature) accross the 1000 features
        if self.num_bags is not None:
            idx = np.random.choice(len(features), size=self.num_bags, replace=False)
            features = features[idx]
        
        return {"features": torch.tensor(features, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32),
                "patient": patient,
                "coordinates": torch.tensor(coordinates, dtype=torch.float32),
                "center": center,
                "sample": sample}



def test_BreastCancerDataset(): 
    data_dir = Path("data/")

    dataset = BreastCancerDataset(data_dir, mode="train")
    test_dataset = BreastCancerDataset(data_dir, mode="test", num_bags=100)

    # training set consisting of 344 samples, each sample of dim 1000 x 2048
    assert(len(dataset) == 344)
    assert(dataset[0]["features"].shape == torch.Size([1000, 2048])) 

    # test set consisting of 149 images. Here, num_bags = 100.
    assert(len(test_dataset) == 149)
    assert(test_dataset[0]["features"].shape == torch.Size([100, 2048]))

if __name__ == '__main__': 
    # download_data()
    test_BreastCancerDataset()
