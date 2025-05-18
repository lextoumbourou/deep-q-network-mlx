# Deep Q-Network (MLX Implementation)

This repo implements the **Deep Q-Network** architecture and training procedure from the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by Mnih et al. using the [MLX array framework](https://github.com/ml-explore/mlx), which allows for training and evaluation locally on Apple Silicon devices (M1+).

Mainly written as an opportunity to learn both MLX and the DQN architecture at once 😄

## Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependancy management. You can follow [these](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) instructions to install it.

## Training

In the paper, they train a model to play seven ATARI games: "Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders". I'm currently training Breakout.

I will train the models on my MacBook Pro M3 Max with 48GB of unified memory.

One note: I've mostly tried to use the parameters exactly as per the paper, how they use a replay buffer size of 1M, but I could only manage 100k without running out of memory.

### Breakout

```
uv run main.py --mode train --env ALE/Breakout-v5
```

### Beam Rider

## Eval

To do.

## Linting and Formatting

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

To check for linting errors:

```bash
uv run ruff check .
```

To automatically fix linting errors and format the code:

```bash
uv run ruff format .
uv run ruff check . --fix
```

You can also configure your editor to use ruff for linting and formatting on save.

## Type Checking

This project uses [mypy](https://mypy-lang.org/) for static type checking.

To run mypy:

```bash
uv run mypy .
```
