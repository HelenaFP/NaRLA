import torch
import narla
import gymnasium as gym
from typing import Tuple
from narla.environments import Environment


class GymEnvironment(Environment):
    def __init__(self, name: str, render: bool = False):
        super().__init__(
            name=name,
            render=render
        )

        self._gym_environment = self._build_gym_environment(
            name=name,
            render=render
        )
        self._action_space = narla.environments.ActionSpace(
            number_of_actions=self._gym_environment.action_space.n
        )

    @staticmethod
    def _build_gym_environment(name: str, render: bool) -> gym.Env:
        render_mode = ""
        if render:
            render_mode = "human"

        gym_environment = gym.make(
            id=name,
            render_mode=render_mode
        )

        return gym_environment

    def reset(self) -> torch.Tensor:
        observation, info = self._gym_environment.reset()
        observation = self._cast_observation(observation)

        return observation

    @property
    def state_size(self) -> int:
        observation = self.reset()

        return observation.shape[-1]

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        observation, reward, terminated, truncated, info = self._gym_environment.step(action.item())

        observation = self._cast_observation(observation)
        reward = self._cast_reward(reward)

        return observation, reward, terminated
