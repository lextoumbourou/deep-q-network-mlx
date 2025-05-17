#!/usr/bin/env python
# Deep Q-Network Implementation with MLX
# Based on the architecture described in "Playing Atari with Deep Reinforcement Learning"

import argparse
from pathlib import Path
from dqn.train import train_agent, evaluate


def main():
    """Main function to parse arguments and start training or evaluation."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Network implementation with MLX"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--env", type=str, default="ALE/Breakout-v5", help="Atari environment name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for training or evaluation",
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load model weights for evaluation",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./weights",
        help="Path to save model weights during training",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_agent(
            env_name=args.env,
            num_episodes=args.episodes,
            render=args.render,
            save_path=Path(args.save) if args.save else None,
        )

    elif args.mode == "eval":
        if not args.load:
            print(
                "Must provide model path for evaluation using --load <path_to_model.npz>"
            )
            return

        evaluate(
            model_path=args.load,
            env_name=args.env,
            num_episodes=args.episodes,
            render=args.render,
        )


if __name__ == "__main__":
    main()
