import gymnasium as gym
import numpy as np


class NetworkEnv(gym.Env):

    def __init__(self, num_switches, num_links, num_flows, alpha, w_min, w_max, episode_length):
        self.network = Network()
        self.episode_length = episode_length

        self.observation_space = gym.spaces.Dict(
                {
                    "time":,
                    "system_capacity":,
                    "queue_occupation":,
                    "arrival_rate":,
                    "loss_traffic":,
                    "utilization":,
                    "delay":
                }
        )

        self.action_space = gym.spaces.Box(low=w_min, high=w_max, shape=(num_links,1), dtype=np.float32)

    

    def _get_obs(self):
        state = self.network.get_state()
        #TODO: translate state values to observation space dict

        # Assuming getting timestep from this function returning with observation
        return state


    def _get_info(self):
        pass


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.network.reset()

        obs, _ = self._get_obs()
        info = self._get_info()


    def step(self, actions):
        self.network.update(actions)
        observation = self._get_obs()
        info = self._get_info()

        if observation["time"][0] = self.episode_length:
            terminated = True
        else:
            terminated = False

        truncated = False
        reward = self._calc_reward()

        return observation, reward, terminated, truncated, info


    def _calc_reward(self):
        #TODO
        pass
        




    
