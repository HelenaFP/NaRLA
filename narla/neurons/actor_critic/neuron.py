from __future__ import annotations

import torch
import narla
import numpy as np
from typing import Tuple
from narla.neurons import Neuron as BaseNeuron


GAMMA = 0.99


class Neuron(BaseNeuron):
    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        learning_rate: float = 1e-4
    ):
        super().__init__()

        self._network = narla.neurons.actor_critic.Network(
            input_size=observation_size,
            output_size=number_of_actions
        ).to(narla.Settings.device)

        self._loss_function = torch.nn.SmoothL1Loss()
        self._optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=learning_rate,
            amsgrad=True
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        action_probabilities, value = self._network(observation)

        distribution = torch.distributions.Categorical(action_probabilities)
        action = distribution.sample()

        self._history.record(
            observation=observation,
            log_probability=distribution.log_prob(action),
            state_value=value,
            action=action,
        )

        return action

    def learn(self):
        value_losses = []
        policy_losses = []

        returns = self.get_returns()
        state_values = self._history.get("state_value")
        log_probabilities = self._history.get("log_probability")

        for return_value, state_value, log_probability in zip(returns, state_values, log_probabilities):
            advantage = return_value - state_value

            # calculate actor (policy) loss
            policy_losses.append(-log_probability * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_loss = torch.nn.functional.smooth_l1_loss(state_value, return_value)
            value_losses.append(value_loss)

        # reset gradients
        self._optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self._optimizer.step()

        self._history.clear()

    def get_returns(self) -> Tuple[torch.Tensor, ...]:
        rewards = self._history.get("reward")

        returns = []
        return_value = 0
        for reward in reversed(rewards):
            return_value = reward + GAMMA * return_value
            returns.insert(0, return_value)

        returns = torch.tensor(returns, device=narla.Settings.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        return returns