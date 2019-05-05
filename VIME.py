import gym
from RL_brain_pytorch import PolicyGradient
import matplotlib.pyplot as plt
import bnn
from trainer import Trainer, batch_loader, timestr
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import numpy as np
from replay_pool import replay_pool
from collections import deque
import copy
import shutil
import pandas as pd

# parameters for RL
DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater than this threshold
GAME = 'CartPole-v1'
is_act_discrete = True
RANDOM_SEED = 1
LR_POLICY = 0.01
GAMMA = 0.99
NUM_EPISODE = 1000
ETA = 1e-4
sample_pool_size = 5000
NORMALIZE = True
MIN_SIZE = 1000
MAX_SIZE = 50000
# previous KL used for normalization
KL_Q_L = 10
# whether to normalize the intrinsic reward based on previous KLs
USE_KL_Q = True
# whether to use VIME
USE_VIME = True
# specify your device here
device = 'cpu'

# following are the parameters for bnn
N_HIDDEN = [32]
BIAS_BNN = True
PRIOR_STD_BNN = 0.5
LIKELIHOOD_DS_BNN = 1.0

LR_BNN = 0.01
BATCH_SIZE_BNN = 10
MAX_EPOCH_BNN = 5 
EXTRA_W_KL = 0.1
LAMDA = 0.01
EXPERIMENT_ID = '%s_%s_%s_lrbnn%.4f_eta%s_%sSeed' % (GAME, 'VIME' if USE_VIME else'Naive', 
                timestr('mdhm'), LR_BNN, str(ETA), 'random'if RANDOM_SEED is None else'fix')

def build_test_set(ob, ac, ob_, device, pool, is_norm):
    if is_norm:
        eps = 1e-8
        obs_mean, obs_std, act_mean, act_std = pool.mean_std()
        ob = (ob - obs_mean) / (obs_std + eps)
        ob_ = (ob_ - obs_mean) / (obs_std + eps)
        ac = (ac - act_mean) / (act_std + eps)
    X = np.array(np.append(ob, ac))
    Y = np.array(ob_)
    X = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(dim=0)
    Y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(dim=0)
    return X, Y

def update_klnormalize(kls, pre_kl_medians):
    pre_kl_medians.append(np.median(kls))
    return np.mean(pre_kl_medians)

def main():
    RENDER = False
    tb_writer = SummaryWriter(log_dir='./tboard/%s' % (EXPERIMENT_ID))
    
    log_cfg = {
        'root': './snap/%s' % EXPERIMENT_ID,
        'display_interval': 99999,
        'val_interval': MAX_EPOCH_BNN,
        'snapshot_interval': 99999,
        'writer': None,
    }
    os.makedirs(log_cfg['root'], exist_ok=True)
    shutil.copy('./VIME.py', os.path.join(log_cfg['root'], 'VIME.py'))

    extra_arg = {
        'external_criterion_val': nn.MSELoss(),
    }

    # gym.envs.register(
    #     id = 'CartPole-Long-v0',
    #     entry_point = 'gym.envs.classic_control:CartPoleEnv',
    #     max_episode_steps = 5000,
    #     reward_threshold = 4500.0
    # )
    # env = gym.make('CartPole-Long-v0')
    gym.envs.register(
        id = 'MountainCarContinuous-Long-v0',
        entry_point = 'gym.envs.classic_control:Continuous_MountainCarEnv',
        max_episode_steps = 999,
        reward_threshold = 90.0
    )
    env = gym.make('MountainCarContinuous-Long-v0')
    # env._max_episode_seconds = 200
    env.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    n_act = env.action_space.n
    n_state = env.observation_space.shape[0]
    # print(env._max_episode_steps)
#
#     intialize the RL class
    running_reward_history = []
    RL = PolicyGradient(
        n_actions= n_act,
        n_features=n_state,
        n_hidden=32,
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
                       max_size= MAX_SIZE,
                       is_act_discrete=is_act_discrete
                        )
#     Algorithm
    kl_normalize = 1
    pre_kl_medians = deque(maxlen= KL_Q_L)

    for i_episode in range(NUM_EPISODE):
        observation = env.reset()
        step_to_go = 0

        while True:
#           if RENDER: env.render()
            action = RL.choose_action(observation= observation)
            step_to_go += 1
            observation_, reward, done, info = env.step(action)

            RL.ep_next_obs.append(observation_)
            RL.ep_naive_rs.append(reward)
            naive_all, intrinsic_all = [], []
            naive_all.append(reward)
#            calculate the kl
            if pool.current_size > pool.min_size and USE_VIME:
#                 add intrinsic reward
                testX, testY = build_test_set(observation, action, observation_, device, pool, NORMALIZE) # pass pool to collect statistics
                intrinsic_reward = model_bnn.kl_second_order_approx(
                    step_size=LAMDA,
                    inputs = testX,
                    targets= testY
                ).item()
            else:
                intrinsic_reward = 0
            RL.ep_kls.append(copy.deepcopy(intrinsic_reward))
            if USE_KL_Q and kl_normalize!= 0:
                intrinsic_reward /= kl_normalize
            reward += ETA* intrinsic_reward
            intrinsic_all.append(intrinsic_reward)
            RL.store_transition(observation, action, reward)
            observation = observation_

            if done:
                break

        if USE_VIME:
            pool.fill_in(RL.ep_obs, RL.ep_as, RL.ep_next_obs)
            if USE_KL_Q:
                kl_normalize = update_klnormalize(RL.ep_kls, pre_kl_medians)
            if pool.current_size > pool.min_size:
                train_set = pool.sample_data(
                    sample_pool_size=sample_pool_size,
                    normalize= NORMALIZE
                )
                # train_bnn
                trainer_bnn = Trainer(model_bnn, train_set, BATCH_SIZE_BNN,
                                    optimizer_bnn, MAX_EPOCH_BNN, log_cfg,
                                    EXTRA_W_KL, device, test_set, extra_arg)
                loss, metric_train, kl_train = trainer_bnn.train()
                tb_writer.add_scalar('loss', loss, i_episode)
                tb_writer.add_scalar('train/metric', metric_train, i_episode)
                tb_writer.add_scalar('train/kl', kl_train, i_episode)
                if metric_train < 0.01:
                    print('Lets rock')

        # train policy net
        ep_rs_sum = sum(RL.ep_naive_rs)

        if 'running_reward' not in locals():
                running_reward = ep_rs_sum
        else:
            running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
        running_reward_history.append(running_reward)
        if running_reward > DISPLAY_REWARD_THRESHOLD:
            RENDER = True  # rendering
        if running_reward > env.spec.reward_threshold:
            print("problem solved")
            # break
        print("episode:", i_episode, "  weighted reward sum:", int(running_reward), " reward:", ep_rs_sum,
                " number of data in pool: ", pool.current_size)
        naive_avg, intrinsic_avg = np.array(naive_all).mean(), np.array(intrinsic_all).mean()*ETA
        print('Average naive reward is %f, average intrinsic reward is %f' % (naive_avg, intrinsic_avg))
        tb_writer.add_scalar('reward', ep_rs_sum, i_episode)
        tb_writer.add_scalar('n_data_pool', pool.current_size, i_episode)
        tb_writer.add_scalar('reward/naive_avg', naive_avg, i_episode)
        tb_writer.add_scalar('reward/intrinsic_avg', intrinsic_avg, i_episode)

        vt = RL.train()

    env.close()

    window = int(NUM_EPISODE / 20)
    fig, ((ax1), (ax2)) = plt.subplots(2,1,sharey=True, figsize = [9,9]);
    rolling_mean = pd.Series(RL.naivie_reward_history).rolling(window).mean()
    std = pd.Series(RL.naivie_reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(RL.naivie_reward_history)), rolling_mean-std, rolling_mean+std, color = 'orange', alpha = 0.2)
    ax1.set_title('Episode Length Moving Average({}--episode window'.format(window))
    ax1.set_xlabel('Episode');
    ax1.set_ylabel('Episode_Length')

    ax2.plot(RL.naivie_reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode');
    ax2.set_ylabel('Episode_Length')
    plt.tight_layout(pad = 2)
    plt.show()

    ax3 = plt.subplot()
    ax3.plot(running_reward_history)
    ax3.set_title('running reward')
    ax3.set_xlabel('Episode');
    ax3.set_ylabel('running reward')
    plt.show()

if __name__ == '__main__':
    main()