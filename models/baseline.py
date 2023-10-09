import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

import unittest

class Baseline(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=1, hidden_dim=256):
        super(Baseline, self).__init__()
        self.feature_dim = feature_dim 
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Simple Linear Layer
        self.linear_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU()
            )

        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        # x shape (batch_size, num_bag, feature_dim)
        processed_features = self.linear_layer(x)
        bag_aggregate = torch.mean(processed_features, dim=1) # (batch_size, feature_dim)
        
        # Classify
        logits = self.classifier(bag_aggregate).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return probabilities

class TestModels(unittest.TestCase):

    def setUp(self):
        self.feature_dim = 2048
        self.num_bags = 100
        self.num_classes = 1
        self.batch_size = 16

        self.x = torch.rand((self.batch_size, self.num_bags, self.feature_dim))

    def test_Baseline(self):
        model = Baseline(self.feature_dim, self.num_classes)
        output = model(self.x)
        self.assertEqual(output.shape, torch.Size([self.batch_size]))

if __name__ == '__main__':
    unittest.main()