import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from non_linear_function import Net
from torch.utils.tensorboard.writer import SummaryWriter
import typing as tt
from dataclasses import dataclass


# declaring and assigning some constants for our neural network
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


# dataclass representing one single step in a episode
@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

# dataclass representing one single episode
@dataclass
class Episode:
    # Note: the reward in our case is undiscounted i.e gamma = 1
    reward: float
    steps: tt.List[EpisodeStep]


def create_batches(env: gym.Env, net: Net, batch_size: int):
    # holds all the episodes needed for this batch
    batch = []
    # total reward for each episode
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim = 1)

    while True:
        # converting our observation so that we can pass it ot out neural network
        obs_t = torch.tensor(obs, dtype=torch.float32)
        # getting the action probabilities using softmax
        action_probs_ = sm(net(obs_t.unsqueeze(0)))
        action_probs = action_probs_.data.numpy()[0]

        # getting the action to be taken using the probabilities we got from our network
        action = np.random.choice(len(action_probs), p = action_probs)
        # getting the next observation, reward, is_done and is_truncated flags from the environment
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        # incrementing the total reward for this current episode
        episode_reward += float(reward)
        # a single step in the current episode
        step = EpisodeStep(obs, action)
        episode_steps.append(step)
        # we also have to check whether the current episode is done or truncated
        if is_done or is_trunc:
            # the current episode
            episode = Episode(episode_reward, episode_steps)
            # the appending the current episode to the batch
            batch.append(episode)
            # since the episode is done we have to reset the total reward counter
            episode_reward = 0.0
            # also resetting the episode step list
            episode_steps = []
            # now we have to reset the environment
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                # yielding the batch if the number of required episodes are generated
                yield batch
                # cleaning the batch list
                batch = []
        # updating our observation with the current observation
        obs = next_obs



def elite_episodes(batch: tt.List[Episode], percentile: float):
    # getting the rewards from each episode 
    rewards = list(map(lambda s: s.reward, batch))
    # getting the reward bound using np.percentile()
    reward_bound = float(np.percentile(rewards, percentile))
    # we also want the reward mean so that we can use it for monitoring
    reward_mean = np.mean(rewards)
    elite_obs: tt.List[np.ndarray] = []
    elite_actions: tt.List[int] = []
    for episode in batch:
        # filtering the episodes, which have total reward greater than the reward bound
        # print("Rewards:", rewards)
        # print("Reward bound:", reward_bound)
        if episode.reward >= reward_bound:
            elite_obs.extend(map(lambda e: e.observation, episode.steps))
            elite_actions.extend(map(lambda e: e.action, episode.steps))
    # print("Number of elite observations:", len(elite_obs))
    # print("Number of elite actions:", len(elite_actions))
    obs_tensor = torch.FloatTensor(elite_obs)
    act_tensor = torch.LongTensor(elite_actions)
    return obs_tensor, act_tensor, reward_bound, reward_mean

if __name__ == "__main__":

    # creating the environment (one can also add render_mode = "human" argument to see the model learn)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="video")
    # shape of the current observation space
    obs_size = env.observation_space.shape[0]
    # number of actions
    n_actions = int(env.action_space.n)
    # our neural network
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    print(net)
    # The objective function
    objective = nn.CrossEntropyLoss()
    # the optimizer
    optimizer = optim.Adam(params=net.parameters(), lr = 0.01)
    # The summary writer for Tensorboard
    writer = SummaryWriter(comment = "cartpole")


    # The actual training loop
    for num, batch in enumerate(create_batches(env, net, BATCH_SIZE)):
        obs_t, acts_t, reward_bound, reward_mean = elite_episodes(batch, PERCENTILE)
        # We zero the gradients of our neural network
        optimizer.zero_grad()
        # passing the observation to our network
        # print(obs_t)
        output = net(obs_t)
        # The loss
        loss = objective(output, acts_t)
        # Now the gradients will be calculated on the loss
        loss.backward()
        # Now we will ask our optimizer to adjust our neural network
        optimizer.step()
        print(f"iteration: {num}, loss = {loss.item()}, reward_mean = {reward_mean}, reward_bound = {reward_bound}")
        writer.add_scalar("loss", loss.item(), num)
        writer.add_scalar("reward_bound", reward_bound, num)
        writer.add_scalar("reward mean", reward_mean, num)
        # condition showing that the environment has been solved
        if reward_mean > 475:
            print("solved!")
            break
    writer.close()



