import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Policynet_Pytorch import Policy_MLP
from torch.distributions import Categorical
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)

class PolicyGradient(object):
    def __init__(self, n_actions, n_features, n_hidden=32, learning_rate=0.01, reward_decay=0.95, device="cpu"):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.device = device

        # observations, actions, and rewards
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # add additional to store next observation, and original reward
        self.ep_next_obs, self.ep_naive_rs, self.ep_kls = [], [], []

        #_build_net
        self.policy_net = Policy_MLP(n_features = self.n_features, n_actions = self.n_actions, n_hidden=n_hidden)
        self.policy_net.apply(self.init_weights)
        self.policy_net.to(device)
        # initialize weights
        

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.1)

    def choose_action(self, observation):
        """
        choose action based on observation
        """
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(observation)
        m = Categorical(probs)
        # sample from action space
        action = m.sample()
        self.policy_net.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def train(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        policy_loss = []
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
#        discounted_ep_rs_norm = torch.tensor(discounted_ep_rs_norm, dtype = torch.float32, device =self.device)
#        saved_log_probs = torch.tensor(self.policy_net.saved_log_probs, dtype=torch.float32, device=self.device)
        for log_prob, R in zip(self.policy_net.saved_log_probs, discounted_ep_rs_norm):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).mean()
        policy_loss.backward()
        optimizer.step()
        del self.policy_net.saved_log_probs[:]
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_next_obs, self.ep_naive_rs, self.ep_kls = [], [], [], [], [], []
        return discounted_ep_rs_norm
