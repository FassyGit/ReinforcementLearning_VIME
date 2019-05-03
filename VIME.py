import gym
from RL_brain_pytorch import PolicyGradient
import matplotlib.pyplot as plt
import bnn
from trainer import Trainer, batch_loader
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import numpy as np
from replay_pool import replay_pool
from collections import deque
import copy

# parameters for RL
DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater than this threshold
GAME = 'CartPole-v0'
RANDOM_SEED = 1
LR_POLICY = 0.02
GAMMA = 0.99
NUM_EPISODE = 500
ETA = 1e-4
BATCH_SIZE_REPLAY = 10
NORMALIZE = True
MIN_SIZE = 500
MAX_SIZE = 50000
# previous KL used for normalization
KL_Q_L = 10
# whether to normalize the intrinsic reward based on previous KLs
USE_KL_Q = True
# whether to use VIME
USE_VIME = False

# following are the parameters for bnn
N_HIDDEN = [32]
BIAS_BNN = True
PRIOR_STD_BNN = 0.5
LIKELIHOOD_DS_BNN = 5.0

LR_BNN = 0.0001
BATCH_SIZE_BNN = 1000
MAX_EPOCH_BNN = 500
EXTRA_W_KL = 0.1
LAMDA = 0.01
EXPERIMENT_ID = 'test for CartPole-v0'

def build_test_set(ob, ac, ob_, device):
    X = np.array(np.append(ob, ac))
    Y = np.array(ob_)
    X = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(dim=0)
    Y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(dim=0)
    return X, Y

def update_klnormalize(kls,pre_kl_medians):
    pre_kl_medians.append(np.median(kls))
    return np.mean(pre_kl_medians)

def main():
    RENDER = False

    log_cfg = {
        'root': './snap/%s' % EXPERIMENT_ID,
        'display_interval': 99999,
        'val_interval': 99999,
        'snapshot_interval': 99999,
        'writer': SummaryWriter(log_dir='./tboard/%s' % (EXPERIMENT_ID)),
    }
    os.makedirs(log_cfg['root'], exist_ok=True)

    extra_arg = {
        'external_criterion': nn.MSELoss(),
    }

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

#     make env
    env = gym.make(GAME)
    env.seed(RANDOM_SEED)
    env = env.unwrapped
    n_act = env.action_space.n
    n_state = env.observation_space.shape[0]

#     intialize the RL class
    RL = PolicyGradient(
        n_actions= n_act,
        n_features=n_state,
        learning_rate=LR_POLICY,
        reward_decay=GAMMA,
        device = device
    )

#     intialize bnn
    n_in = n_state + 1
    n_out = n_state
    model_bnn = bnn.BNN(
        n_in=n_in,
        n_hidden=N_HIDDEN,
        n_out= n_out,
        bias=BIAS_BNN,
        prior_std=PRIOR_STD_BNN,
        likelihood_sd=LIKELIHOOD_DS_BNN,
        nla=bnn.ReLU()
    )
    optimizer_bnn = torch.optim.Adam(model_bnn.parameters(),
                                     lr = LR_BNN
                                     )
    train_set = None
    test_set = None

#     initialize replaypool
    pool = replay_pool(min_size= MIN_SIZE,
                       max_size= MAX_SIZE
                        )
#     Algorithm
    kl_normalize = 1
    pre_kl_medians = deque(maxlen= KL_Q_L)
    for i_episode in range(NUM_EPISODE):
        observation = env.reset()
        while True:
            if RENDER: env.render()

            action = RL.choose_action(observation= observation)
            observation_, reward, done, _ = env.step(action)

            RL.ep_next_obs.append(observation_)
            RL.ep_naive_rs.append(reward)

#            calculate the kl
            if pool.current_size > pool.min_size and USE_VIME:
#                 add intrinsic reward
                testX, testY = build_test_set(observation, action, observation_, device)
                intrinsic_reward = model_bnn.kl_second_order_approx(
                    step_size=LAMDA,
                    inputs = testX,
                    targets= testY
                )
            else:
                intrinsic_reward = 0
            RL.ep_kls.append(copy.deepcopy(intrinsic_reward))
            if USE_KL_Q and kl_normalize!= 0:
                intrinsic_reward /= kl_normalize
            reward += ETA* intrinsic_reward
            RL.store_transition(observation, action, reward)

            if done:
                if USE_VIME:
                    pool.fill_in(RL.ep_obs, RL.ep_as, RL.ep_next_obs)
                    if USE_KL_Q:
                        kl_normalize = update_klnormalize(RL.ep_kls, pre_kl_medians)
                    if pool.current_size > pool.min_size :
                        train_set = pool.sample_data(
                            batch_size=BATCH_SIZE_REPLAY,
                            normalize= NORMALIZE
                        )
                        # train_bnn
                        trainer_bnn = Trainer(model_bnn, train_set, BATCH_SIZE_BNN,
                                              optimizer_bnn, MAX_EPOCH_BNN, log_cfg,
                                              EXTRA_W_KL, device, test_set, extra_arg)
                        trainer_bnn.train()

                        # train policy net
                ep_rs_sum = sum(RL.ep_naive_rs)

                if 'running_reward' not in locals():
                    running_reward = ep_rs_sum
                else:
                    # running_reward = ep_rs_sum
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  weighted reward sum:", int(running_reward), " reward:", ep_rs_sum,
                      " number of data in pool: ", pool.current_size)

                vt = RL.train()

                # if i_episode == 0:
                #     plt.plot(vt)    # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break

            observation = observation_



if __name__ == '__main__':
    main()