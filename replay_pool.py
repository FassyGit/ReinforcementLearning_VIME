import numpy as np

class replay_pool(object):
    def __init__(self, min_size=5000, max_size=500000, is_act_discrete=True):
        self.min_size = min_size
        self.max_size = max_size

        self.obs, self.act, self.next_obs = [], [], []
        self.current_size = 0
        self.is_act_discrete = is_act_discrete

    def fill_in(self,observations, actions, next_observations):
        self.obs = self.obs + observations
        self.act = self.act + actions
        self.next_obs = self.next_obs + next_observations
        self.current_size = len(self.obs)
        # remove the early data if full
        if self.current_size > self.max_size:
            self.obs = self.obs[-self.max_size:]
            self.act = self.act[-self.max_size:]
            self.next_obs = self.next_obs[-self.max_size:]
            self.current_size = len(self.obs) # should be equal to max_size

    def mean_std(self):
        obs_mean = np.mean(self.obs, axis=0)
        obs_std = np.std(self.obs, axis=0)
        if self.is_act_discrete:
            act_mean = 0
            act_std = 1
        else:
            act_mean = np.mean(self.act, axis=0)
            act_std = np.std(self.act, axis=0)
        
        return obs_mean, obs_std, act_mean, act_std

    def sample_data(self, sample_pool_size=1000, normalize=True):
        """

        :param normalize: NORMALIZE the input for BNN
        :return:
        """
        # check if enough data to be sampled
        assert self.current_size > self.min_size
        if self.current_size >= sample_pool_size:
            idx = np.random.randint(self.current_size,size = sample_pool_size)
            obs_ = np.take(np.array(self.obs),idx,axis=0)
            act_ = np.take(np.array(self.act),idx,axis=0)
            bnn_output = np.take(np.array(self.next_obs),idx,axis=0)
        else:
            obs_ = np.array(self.obs[-sample_pool_size:])
            act_ = np.array(self.act[-sample_pool_size:])
            bnn_output = np.array(self.next_obs[-sample_pool_size:])

        # normalize input if required
        if normalize:
            obs_mean, obs_std, act_mean, act_std = self.mean_std()

            obs_ = (obs_ - obs_mean) / (obs_std + 1e-8)
            act_ = (act_ - act_mean) / (act_std + 1e-8)
            bnn_output = (bnn_output - obs_mean) / (obs_std + 1e-8)
            
            print('Current obs mean/std is %s+/-%s' % (str(obs_mean), str(obs_std)))
            print('Current act mean/std is %s+/-%s' % (str(act_mean), str(act_std)))
        
        bnn_input = np.column_stack((obs_,act_))
        return bnn_input, bnn_output

