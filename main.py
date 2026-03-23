from viz import run_live
import argparse
from config import load_config, build_world


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "-t", "--steps", type=int, default=500, help="Number of simulation steps to run"
    )
    parser.add_argument(
        "-c",
        "--claim-id",
        type=int,
        default=None,
        help="Claim ID to visualize (default: first claim from config)",
    )
    parser.add_argument(
        "-d",
        "--draw-every",
        type=int,
        default=1,
        help="Draw visualization every N steps",
    )
    parser.add_argument(
        "-g",
        "--layout-seed",
        type=int,
        default=0,
        help="Random seed for network layout",
    )
    parser.add_argument(
        "-p",
        "--pause-time",
        type=float,
        default=0.25,
        help="Pause time between animation frames (seconds)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    world = build_world(cfg)

    # Use provided claim_id or default to first claim from config
    claim_id = (
        args.claim_id
        if args.claim_id is not None
        else next(iter(cfg.world.truths.keys()))
    )

    run_live(
        world,
        steps=args.steps,
        claim_id=claim_id,
        draw_every=args.draw_every,
        layout_seed=args.layout_seed,
        pause_time=args.pause_time,
    )
