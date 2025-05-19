#!/usr/bin/env python

"""
Deep Q-Network Implementation using MLX.

Based on the architecture described in "Playing Atari with Deep Reinforcement Learning"
by Mnih et al.
"""

import argparse
from pathlib import Path

from dqn.evaluate import evaluate, record_episode_video
from dqn.plotting import load_metrics, plot_training_results
from dqn.train import train_agent


def main():
    """Main function to parse arguments and start training or evaluation."""
    parser = argparse.ArgumentParser(description="Deep Q-Network implementation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "make_video", "plot_metrics"],
        help="Mode: train, eval, make_video, or plot_metrics",
    )
    parser.add_argument(
        "--env", type=str, help="Atari environment name"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=10_000_000,
        help="Total number of steps to train for",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=50_000,
        help="Number of steps per epoch during training",
    )
    parser.add_argument(
        "--frameskip",
        type=int,
        default=4,
        help="Number of frames to skip",
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10_000,
        help="Number of steps to evaluate for",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to load model weights for evaluation or video generation",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./weights",
        help="Path to save model weights during training",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to metrics file for plotting",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_agent(
            env_name=args.env,
            save_path=Path(args.save_path),
            train_steps=args.train_steps,
            steps_per_epoch=args.steps_per_epoch,
            frameskip=args.frameskip,
        )

    elif args.mode == "eval":
        if not args.load_path:
            print("Must provide model path for evaluation")
            return

        evaluate(
            model_path=args.load_path,
            env_name=args.env,
            eval_steps=args.eval_steps,
            render=args.render,
        )

    elif args.mode == "make_video":
        if not args.load_path:
            print("Must provide model path for video generation")
            return

        cleaned_env_name = args.env.replace("/", "_").replace("-", "_")

        video_dir = Path("videos") / cleaned_env_name

        load_path = Path(args.load_path)
        epoch_num = load_path.name.split(".")[0].split("_")[-1]

        video_filename = f"epoch_{epoch_num}.mp4"
        output_video_filepath = video_dir / video_filename

        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"Recording video for {args.env} to {output_video_filepath}...")

        record_episode_video(
            model_path=args.load_path,
            env_name=args.env,
            output_video_filepath=output_video_filepath,
            video_length=60000,
        )

    elif args.mode == "plot_metrics":
        if not args.metrics_file:
            print("Must provide metrics file path for plotting")
            return

        metrics_path = Path(args.metrics_file)
        if not metrics_path.exists():
            print(f"Metrics file not found: {metrics_path}")
            return

        avg_rewards, avg_max_qs, env_name = load_metrics(metrics_path)
        plot_training_results(
            avg_rewards,
            avg_max_qs,
            metrics_path.parent,
            env_name,
        )


if __name__ == "__main__":
    main()
