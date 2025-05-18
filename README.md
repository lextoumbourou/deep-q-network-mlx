# Deep Q-Network

This repo implements the **Deep Q-Network** architecture and training procedure from the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by Mnih et all.

## Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependancy management. You can follow [these](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) instructions to install it.

## Training

In the paper, they train a model to play seven ATARI games: "Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders". So far I'm trained:

* Breakout
* Beam Rider

I trained this models on my MacBook Pro M3 Max with 48GB of unified memory. In the paper they use a replay buffer size of 1M, but I could only manage 100k before running out of memory.

### Breakout

```
uv run main.py --mode train --env BreakoutNoFrameskip-v4 
```

### Beam Rider

## Eval

To do.
