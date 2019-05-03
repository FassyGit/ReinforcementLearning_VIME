import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy_MLP(nn.Module):
    """
    n_features: dim of observations, input_dim
    n_actios : dim of action space, output_dim
    """
    def __init__(self, n_features,n_actions ):
        super(Policy_MLP, self).__init__()
        self.affine1 = nn.Linear(n_features, 10)
        self.affine2 = nn.Linear(10, n_actions)
        self.saved_log_probs = []

    def forward(self, x):
        x = self.affine1(x)
        x = torch.tanh(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
