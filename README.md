# Cross-Entropy Method for Reinforcement Learning

A small PyTorch implementation of the **Cross-Entropy Method (CEM)** for reinforcement learning, currently demonstrated on the **CartPole-v1** environment from Gymnasium.

The project is primarily intended as a learning-focused implementation of CEM: the agent generates complete episodes, keeps the highest-performing episodes, and trains a policy network to reproduce the actions taken in those elite trajectories.

## Demo

### Trained Agent — CartPole-v1

The following video shows the trained agent playing CartPole after learning with the Cross-Entropy Method.

<video src="https://raw.githubusercontent.com/anantlahamge07/cross-entropy-method-rl/main/video/rl-video-episode-729.mp4" controls width="700"></video>

[Watch the latest recorded episode](https://github.com/anantlahamge07/cross-entropy-method-rl/blob/main/video/rl-video-episode-729.mp4)

## Overview

The Cross-Entropy Method is a relatively simple policy-search approach:

1. Run the current policy in the environment and collect complete episodes.
2. Measure the total reward of every episode.
3. Select the best-performing episodes using a reward percentile.
4. Use the observations and actions from those elite episodes as supervised training data.
5. Update the policy network with cross-entropy loss.
6. Repeat until the policy reaches the desired performance.

For CartPole, the environment provides a 4-dimensional observation and two possible actions. The neural network maps an observation to two action logits, which are converted to probabilities with softmax during episode generation.

## Algorithm

For every batch of episodes, the implementation calculates a percentile-based reward threshold:

```text
reward_bound = percentile(episode_rewards, PERCENTILE)
```

Only episodes satisfying

```text
episode_reward >= reward_bound
```

are treated as **elite episodes**.

Their individual `(observation, action)` pairs form the training dataset for the current iteration. The policy is then optimized using `CrossEntropyLoss`.

A simplified view of the training loop is:

```text
Current policy
      │
      ▼
Generate episodes
      │
      ▼
Calculate episode rewards
      │
      ▼
Keep elite episodes
      │
      ▼
(observation, action) pairs
      │
      ▼
Cross-entropy training
      │
      ▼
Updated policy
      │
      └─────────────── repeat
```

## Project Structure

```text
cross-entropy-method-rl/
├── cross_entropy.py          # CEM training loop and episode generation
├── non_linear_function.py    # PyTorch policy network
├── README.md
├── LICENSE
├── runs/                     # TensorBoard logs
└── video/                    # Recorded environment videos
```

### `cross_entropy.py`

Contains the main implementation, including:

- `EpisodeStep` and `Episode` data classes
- batched episode generation with `create_batches()`
- elite trajectory selection with `elite_episodes()`
- the CEM training loop
- TensorBoard logging
- optional video recording through Gymnasium

### `non_linear_function.py`

Defines the policy network:

```text
Input observation
      ↓
Linear(obs_size → 128)
      ↓
ReLU
      ↓
Linear(128 → number of actions)
      ↓
Action logits
```

## Requirements

The project uses Python, PyTorch, NumPy, Gymnasium, and TensorBoard.

For video recording, Gymnasium's optional video dependencies are also required.

A typical environment setup is:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install torch numpy tensorboard
pip install "gymnasium[classic-control]"
pip install "gymnasium[other]"
```

> Depending on your Python/PyTorch setup, the exact PyTorch installation command may differ. The official PyTorch installation selector is recommended for GPU-specific installations.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/anantlahamge07/cross-entropy-method-rl.git
cd cross-entropy-method-rl
```

Run the training script:

```bash
python cross_entropy.py
```

The script prints the training progress for each iteration, including:

- iteration number
- training loss
- mean episode reward
- reward percentile bound

Training stops when the mean reward exceeds the configured threshold.

## Configuration

The main hyperparameters are defined near the top of `cross_entropy.py`:

```python
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
```

They control:

| Parameter | Meaning | Current value |
|---|---|---:|
| `HIDDEN_SIZE` | Number of neurons in the hidden layer | `128` |
| `BATCH_SIZE` | Episodes collected before each update | `16` |
| `PERCENTILE` | Reward percentile used to select elite episodes | `70` |

The optimizer currently uses Adam with a learning rate of `0.01`.

## TensorBoard

Training metrics are written with `SummaryWriter` to the `runs/` directory.

Launch TensorBoard with:

```bash
tensorboard --logdir runs
```

Then open the local TensorBoard address shown in the terminal.

The training script logs:

- `loss`
- `reward_bound`
- `reward mean`

These metrics make it easy to observe whether the policy is improving across iterations.

## Video Recording

The environment is initialized with an RGB render mode and wrapped with Gymnasium's `RecordVideo` wrapper. Recorded videos are written to:

```text
video/
```

If you do not need video recording, you can remove the `RecordVideo` wrapper and use a normal environment:

```python
env = gym.make("CartPole-v1")
```

For visual rendering in a supported desktop environment, the code can instead be adapted to use a human render mode.

## Why CEM?

CEM is useful as a compact introduction to **policy search** and **black-box optimization in reinforcement learning**.

Unlike value-based methods such as DQN, this implementation does not learn a value function. Instead, it directly improves the policy by learning from actions that were successful in previously sampled trajectories.

This makes the algorithm conceptually simple:

```text
sample → evaluate → select elites → imitate elites → repeat
```

## Current Scope

The repository is intentionally small and currently focuses on a single discrete-action control problem:

- **Environment:** Gymnasium CartPole-v1
- **Action space:** discrete
- **Policy:** feed-forward PyTorch neural network
- **Optimization:** Cross-Entropy Method + supervised cross-entropy training

The code is structured so that the same overall approach can be adapted to other environments, provided the policy output and action-selection logic are adjusted to match the environment's action space.

## Possible Extensions

Some natural next steps for the project are:

- support additional Gymnasium environments
- separate training and evaluation environments
- add deterministic evaluation episodes
- experiment with different batch sizes and elite percentiles
- add model checkpointing
- compare CEM against other policy-gradient or value-based methods
- support continuous action spaces
- add reproducible random seeds and experiment configuration files

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Anant Lahamge**

GitHub: [@anantlahamge07](https://github.com/anantlahamge07)
