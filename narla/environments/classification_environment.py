from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
import narla
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple
from narla.environments import Environment


class ClassificationEnvironment(Environment):
    """
    
    """
    def __init__(self, name: narla.environments.AvailableEnvironments, render: bool = False, learn_sequence: bool = True):
        super().__init__(
            name=name,
            render=render
        )

        self._learn_sequence = learn_sequence

        self._dataloader_train, self._dataloader_test = self._build_classification_environment(name=name, learn_sequence=learn_sequence)

        self._action_space = narla.environments.ActionSpace(
            number_of_actions=3
        )

    def _build_classification_environment(self, name: narla.environments.AvailableEnvironments, learn_sequence: bool) -> Tuple[DataLoader, DataLoader]:
        
        if name.value=="Iris":
            iris = load_iris()
            X = iris.data
            y = iris.target
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_tensor = torch.Tensor(X_train)
            y_train_tensor = torch.Tensor(y_train)
            X_test_tensor = torch.Tensor(X_test)
            y_test_tensor = torch.Tensor(y_test)
            self._dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
            self._dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
            if learn_sequence:
                dataloader_train = DataLoader(self._dataset_train, batch_size=1, shuffle=False)
            else:
                dataloader_train = DataLoader(self._dataset_train, batch_size=1, shuffle=True)
            dataloader_test = DataLoader(self._dataset_test , batch_size=1, shuffle=False)
            return dataloader_train, dataloader_test
       
        return None, None

    @property
    def observation_size(self) -> int:
        observation = self.reset()

        return observation.shape[-1]

    def reset(self, train=True) -> torch.Tensor:
        self._episode_reward = 0

        if train:
            if self._learn_sequence:
                dataloader_train = DataLoader(self._dataset_train, batch_size=1, shuffle=False)
            else:
                dataloader_train = DataLoader(self._dataset_train, batch_size=1, shuffle=True)
            self._data_iter = iter(self._dataloader_train)
            batch = next(self._data_iter)
            input, label = batch
            self._current_label = label.to(device=narla.experiment_settings.trial_settings.device)
        else:
            self._dataloader_test = DataLoader(self._dataset_test, batch_size=1, shuffle=False)
            self._data_iter = iter(self._dataloader_test)
            batch = next(self._data_iter)
            input, label = batch
            self._current_label = label.to(device=narla.experiment_settings.trial_settings.device)
            
        return input.to(device=narla.experiment_settings.trial_settings.device)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        action = int(action.item())
        reward = action == self._current_label.item()
        # print("action: ", action, flush=True)
        # print("label: ", self._current_label.item(), flush=True)
        # print("reward: ", reward, flush=True)
        self._episode_reward += reward
        reward = self._cast_reward(reward)
        
        try:
            batch = next(self._data_iter)
            input, label = batch
            terminated = False
            self._current_label = label.to(device=narla.experiment_settings.trial_settings.device)
            return input.to(device=narla.experiment_settings.trial_settings.device), reward, terminated
        except StopIteration:
            terminated = True
            return None, reward, terminated
        