# Deep Q-Network

This repo implements the **Deep Q-Network** architecture and training procedure from the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.

## Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependancy management. You can follow [these](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) instructions to install it.

## Train

```
uv run main.py --mode train
```

Note: after a few days working on this, I can't figure out why I keep hitting errors around the 5000th episode. Let me know if anyone has any ideas.

```bash
Traceback (most recent call last):
  File "/Users/lex/code/deep-q-network-mlx/main.py", line 365, in <module>
    main()
  File "/Users/lex/code/deep-q-network-mlx/main.py", line 347, in main
    model, rewards, avg_rewards, avg_max_qs = train_agent(
  File "/Users/lex/code/deep-q-network-mlx/main.py", line 210, in train_agent
    mx.eval(model.parameters(), loss)
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

## Eval

To do.
