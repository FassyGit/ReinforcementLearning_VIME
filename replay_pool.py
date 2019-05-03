import numpy as np

class replay_pool:
    def __init__(self, min_size=5000, max_size=500000,batch_size=10):
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size

        self.obs, self.act, self.next_obs = [], [], []
        self.current_size = 0

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


    def sample_data(self,normalize=True):
        # check if enough data to be sampled
        assert self.current_size > self.batch_size
        idx = np.random.randint(self.current_size,size=self.batch_size)
        obs_ = np.take(np.array(self.obs),idx,axis=0)
        act_ = np.take(np.array(self.act),idx,axis=0)
        bnn_output = np.take(np.array(self.next_obs),idx,axis=0)

        # normalize input if required
        if normalize:
            obs_mean = np.mean(self.obs, axis=0)
            obs_std = np.std(self.obs, axis=0)
            act_mean = np.mean(self.act, axis=0)
            act_std = np.std(self.act, axis=0)

            obs_ = (obs_ - obs_mean) / (obs_std + 1e-8)
            act_ = (act_ - act_mean) / (act_std + 1e-8)
        
        bnn_input = np.column_stack((obs_,act_))
        return bnn_input, bnn_output

