from typing import Optional
import gymnasium as gym
import numpy as np
from Network import Network
import networkx as nx

class NetworkEnv(gym.Env):

    def __init__(self, topology, alpha, w_min, w_max, episode_length):
        self.topology = topology
        self.network = Network(topology)
        self.network.show_network()
        self.episode_length = episode_length
        self.alpha = alpha
        self.num_switches = self.network.num_of_switches

        self.observation_space = gym.spaces.Dict(
                {
                    "time": gym.spaces.Box(low=0, high=episode_length, shape=(self.num_switches, 1), dtype=np.float32),
                    "system_capacity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32),
                    "queue_occupation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32),
                    "arrival_rate": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32),
                    "loss_traffic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32),
                    "utilization": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32),
                    "delay": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_switches, 1), dtype=np.float32)
                }
        )

        self.action_space = gym.spaces.Box(low=w_min, high=w_max, shape=(len(self.network.Graph.edges),1), dtype=np.float32)


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
        self.network = Network(self.topology)
        self.network.show_network()

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
        rd_denom = 0
       
        longest_path = []
        for flow in self.network.f:
            path = self.network.shortest_path(flow[0], flow[1])
            if len(path) > len(longest_path):
                longest_path = path
        print("LONGEST PATH LEN: " + str(len(longest_path)))
    

        for n in longest_path:
            rd_denom += observation["system_capacity"][n] / self.network.u[n]
            #print(rd_denom)

        #print("REWARD BREAKDOWN: RD NUMERATOR")
        #print(self.network.d_k_e2e.mean())
        rd = 1 - (self.network.d_k_e2e.mean() / rd_denom)

        rp = 1 - (np.sum(observation["loss_traffic"])/np.sum(observation["arrival_rate"]))

        #print("REWARD BREAKDOWN FINAL")
        #print(rd)
        #print(rp)

        reward = self.alpha * rd + (1 - self.alpha) * rp
        return reward

