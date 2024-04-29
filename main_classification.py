import narla
import plotly.express as px
import torch
import numpy as np
import os

import narla.settings.trial_settings

# Parse command line args into narla.settings
settings = narla.settings.parse_args()

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Create the Environment
environment = narla.environments.ClassificationEnvironment(
    name=narla.environments.ClassificationEnvironments.IRIS,
    learn_sequence=True
)
observation = environment.reset()

print("Created environment", flush=True)

# Build the MultiAgentNetwork based on settings
network = narla.multi_agent_network.MultiAgentNetwork(
    observation_size=environment.observation_size,
    number_of_actions=environment.action_space.number_of_actions,
    network_settings=settings.multi_agent_network_settings,
)

print("Built network", flush=True)

for episode_number in range(1, settings.trial_settings.maximum_episodes + 1):

    observation = environment.reset()
    for count in narla.count():
        # The network computes an action based on the observation
        action = network.act(observation)

        # Execute the action in the environment
        observation, reward, terminated = environment.step(action)

        network.compute_biological_rewards()

        # Distribute reward information to all layers
        network.distribute_to_layers(**{
            narla.rewards.RewardTypes.TASK_REWARD: reward,
            narla.history.saved_data.TERMINATED: terminated,
        })

        if terminated:
            print("Episode:", episode_number, "total reward:", environment.episode_reward, flush=True)

            # Record the reward history for the episode
            network.record(**{
                narla.history.saved_data.EPISODE_REWARD: environment.episode_reward,
                narla.history.saved_data.EPISODE_COUNT: episode_number,
            })
            break

    # Network learns based on episodes
    network.learn()

    # stop when agent has sufficiently learned
    if episode_number > 1000 and np.mean(network.history.get(narla.history.saved_data.EPISODE_REWARD)[-100:]) >= 120: 
        print("Network has learned, stopping training.", flush=True)
        break

    if episode_number % settings.trial_settings.save_every:
        narla.io.save_history_as_data_frame(
            name="results",
            history=network.history
        )


narla.io.save_history_as_data_frame(
    name="results",
    history=network.history
)



# clear history
network.history.clear()
# test performance on test data
observation = environment.reset(train=False)
print("Testing network", flush=True)
for count in narla.count():
    # The network computes an action based on the observation
    action = network.act(observation)

    # Execute the action in the environment
    observation, reward, terminated = environment.step(action)

    network.compute_biological_rewards()

    # Distribute reward information to all layers
    network.distribute_to_layers(**{
        narla.rewards.RewardTypes.TASK_REWARD: reward,
        narla.history.saved_data.TERMINATED: terminated,
    })

    if terminated:
        print("Episode:", episode_number, "total reward:", environment.episode_reward, flush=True)

        # Record the reward history for the episode
        network.record(**{
            narla.history.saved_data.EPISODE_REWARD: environment.episode_reward,
            narla.history.saved_data.EPISODE_COUNT: episode_number,
        })
        break

narla.io.save_history_as_data_frame(
    name="results_test",
    history=network.history
)


print("done", flush=True)