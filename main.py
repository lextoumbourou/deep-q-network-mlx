#!/usr/bin/env python
# Deep Q-Network Implementation with MLX
# Based on the architecture described in "Playing Atari with Deep Reinforcement Learning"

import argparse
from pathlib import Path
from dqn.train import train_agent
from dqn.evaluate import record_episode_video, evaluate


def main():
    """Main function to parse arguments and start training or evaluation."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Network implementation with MLX"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "make_video"],
        help="Mode: train, eval, or make_video",
    )
    parser.add_argument(
        "--env", type=str, default="ALE/Breakout-v5", help="Atari environment name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes for training or evaluation",
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load model weights for evaluation or video generation",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./weights",
        help="Path to save model weights during training",
    )
    parser.add_argument(
        "--video_episode_num",
        type=int,
        default=0,
        help="Episode number for the output video filename (used in make_video mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_agent(
            env_name=args.env,
            save_path=Path(args.save),
            num_episodes=args.episodes
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

    elif args.mode == "make_video":
        if not args.load:
            print(
                "Must provide model path for video generation using --load <path_to_model.npz>"
            )
            return

        cleaned_env_name = args.env.replace("/", "_").replace("-", "_")

        video_dir = Path("videos") / cleaned_env_name

        video_filename = f"episode_{args.video_episode_num}.mp4"
        output_video_filepath = video_dir / video_filename

        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"Recording video for {args.env} to {output_video_filepath}...")

        record_episode_video(
            model_path=args.load,
            env_name=args.env,
            output_video_filepath=output_video_filepath,
            video_length=60000,
        )


if __name__ == "__main__":
    main()
