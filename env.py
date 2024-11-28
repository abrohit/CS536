from typing import Optional
import gymnasium as gym
import numpy as np
from Network import Network
import networkx as nx

class NetworkEnv(gym.Env):

    def __init__(self, num_switches, num_links, num_flows, alpha, w_min, w_max, episode_length):
        self.network = Network()
        self.episode_length = episode_length
        self.alpha = alpha
        self.num_switches = num_switches

        self.observation_space = gym.spaces.Dict(
                {
                    "time": gym.spaces.Box(low=0, high=episode_length, shape=(num_switches, 1), dtype=np.float32),
                    "system_capacity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32),
                    "queue_occupation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32),
                    "arrival_rate": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32),
                    "loss_traffic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32),
                    "utilization": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32),
                    "delay": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_switches, 1), dtype=np.float32)
                }
        )

        self.action_space = gym.spaces.Box(low=w_min, high=w_max, shape=(num_links,1), dtype=np.float32)



    def _get_obs(self):
        state = {
                "time": [self.network.t for i in range(self.num_switches)],
                "system_capacity": self.network.K,
                "queue_occupation": self.network.e_n,
                "arrival_rate": self.network.agg_lam,
                "loss_traffic": self.network.e_l,
                "utilization": self.network.rho,
                "delay": self.network.e_d
        }
        return state


    def _get_info(self):
        pass


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        #super().reset(seed=seed)

        #self.network.reset()
        self.network = Network()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info


    def step(self, actions):
        self.network.update_weights(actions)
        observation = self._get_obs()
        info = self._get_info()

        if observation["time"][0] == self.episode_length:
            terminated = True
        else:
            terminated = False

        truncated = False
        reward = self._calc_reward(observation)

        return observation, reward, terminated, truncated, info


    def _calc_reward(self, observation):
        longest_path = nx.dag_longest_path(self.network.Graph)
        rd_denom = 0
        for n in longest_path:
            rd_denom += observation["system_capacity"][n] / self.network.u[n]
        rd = 1 - (self.network.d_k_e2e.mean() / rd_denom)

        rp = 1 - (np.sum(observation["loss_traffic"])/np.sum(observation["arrival_rate"]))

        reward = self.alpha * rd + (1 - self.alpha) * rp
        return reward

