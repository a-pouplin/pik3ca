import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

import unittest

class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim 
        self.hidden_dim = hidden_dim

        # Simple Linear Layer
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape (batch_size, num_bag, feature_dim)
        return self.attention(x) # (batch_size, num_bag, 1)


class ClassifierModule(nn.Module):
    """
    Reduce the dimension of the features to 1.
    """
    def __init__(self, feature_dim, hidden_dim):
        super(ClassifierModule, self).__init__()
        self.feature_dim = feature_dim 
        self.hidden_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        # x shape (batch_size, feature_dim)
        return self.classifier(x)


class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=128):
        super(AttentionMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.attention = AttentionModule(self.feature_dim, self.hidden_dim)
        self.classifier = ClassifierModule(self.feature_dim, self.hidden_dim)

    def forward(self, x):
        # Attention --> alpha_i(x)
        attention = self.attention(x) # (batch_size, num_bag, 1)

        # Sum pooling over bags --> sum_i alpha_i(x) * f(x_i)
        aggregate = torch.einsum('bic,bif->bcf', attention, x) # (batch_size, 1, feature_dim)
        aggregate = torch.squeeze(aggregate, dim=1) # (batch_size, feature_dim)

        # Classify --> phi_p (sum_i m_i(x))
        logits = self.classifier(aggregate).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return probabilities  
    
class GatedAttentionMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=32):
        super(GatedAttentionMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.attention = AttentionModule(self.feature_dim, self.hidden_dim)
        self.gate = AttentionModule(self.feature_dim, self.hidden_dim)
        self.classifier = ClassifierModule(self.feature_dim, self.hidden_dim)

    def forward(self, x):        
        # Attention --> alpha_i(x)
        attention = self.attention(x) # (batch_size, num_bag, 1)

        # Applying gate
        gating_weights = self.gate(x)
        gated_attention = attention * gating_weights # (batch_size, num_bag, 1)

        # Sum pooling over bags --> sum_i alpha_i(x) * f(x_i)
        aggregate = torch.einsum('bic,bif->bcf', gated_attention, x) # (batch_size, 1, feature_dim)
        aggregate = torch.squeeze(aggregate, dim=1) # (batch_size, feature_dim)

        # Classify --> phi_p (sum_i m_i(x))
        logits = self.gate(aggregate).squeeze(-1)
        probabilities = torch.sigmoid(logits) 
        return probabilities  


class AdditiveMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=128):
        super(AdditiveMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.attention = AttentionModule(self.feature_dim, self.hidden_dim)
        self.classifier = ClassifierModule(self.feature_dim, self.hidden_dim)

    def forward(self, x):
        # Attention --> alpha_i(x)
        attention = self.attention(x) # (batch_size, num_bag, 1)

        # Classifier w/ attention and features --> phi_p (m_i(x))
        mul_attention = attention * x # (batch_size, num_bag, feature_dim)
        logits_attention = self.classifier(mul_attention) # (batch_size, num_bag, 1)

        # Sum pooling over bags --> sum_i phi_p (m_i(x))
        aggregate = torch.sum(logits_attention, dim=1) # (batch_size, 1)
        aggregate = torch.squeeze(aggregate, dim=1) # (batch_size,)
        probabilities = torch.sigmoid(aggregate)
        return probabilities  


class TestModels(unittest.TestCase):

    def setUp(self):
        self.hidden_dim = 128
        self.feature_dim = 2048
        self.num_bags = 100
        self.batch_size = 16

        self.x = torch.rand((self.batch_size, self.num_bags, self.feature_dim))
        self.aggregate = torch.rand((self.batch_size, self.feature_dim))
    
    def test_ClassifierModule(self):
        model = ClassifierModule(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)
        # logits = model(self.aggregate)
        self.assertEqual(model(self.aggregate).shape, torch.Size([self.batch_size, 1]))
        self.assertEqual(model(self.x).shape, torch.Size([self.batch_size, self.num_bags, 1]))

    def test_AttentionModule(self):
        model = AttentionModule(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)
        attention = model(self.x)
        self.assertEqual(attention.shape, torch.Size([self.batch_size, self.num_bags, 1]))
        self.assertTrue(torch.allclose(torch.sum(attention, dim=1), torch.ones(self.batch_size, 1)))

    def test_AttentionMIL(self):        
        model = AttentionMIL(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)
        attention = model.attention(self.x)
        probabilities = model(self.x)

        self.assertTrue(torch.allclose(torch.sum(attention, dim=1), torch.ones(self.batch_size, 1)))
        self.assertEqual(attention.shape, torch.Size([self.batch_size, self.num_bags, 1]))
        self.assertEqual(probabilities.shape, torch.Size([self.batch_size]))

    def test_GatedAttentionMIL(self):
        model = GatedAttentionMIL(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)
        attention = model.attention(self.x)
        probabilities = model(self.x)

        self.assertTrue(torch.allclose(torch.sum(attention, dim=1), torch.ones(self.batch_size, 1)))
        self.assertEqual(attention.shape, torch.Size([self.batch_size, self.num_bags, 1]))
        self.assertEqual(probabilities.shape, torch.Size([self.batch_size]))

    def test_AdditiveMIL(self):
        model = AdditiveMIL(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)
        attention = model.attention(self.x)
        logit = model.classifier(self.x)
        probabilities = model(self.x)

        self.assertTrue(torch.allclose(torch.sum(attention, dim=1), torch.ones(self.batch_size, 1)))
        self.assertEqual(attention.shape, torch.Size([self.batch_size, self.num_bags, 1]))
        self.assertEqual(logit.shape, torch.Size([self.batch_size, self.num_bags, 1]))
        self.assertEqual(probabilities.shape, torch.Size([self.batch_size]))


if __name__ == '__main__':
    unittest.main()
