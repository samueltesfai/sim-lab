import argparse

from simlab.viz import run_viz
from simlab.config import load_config, build_world
from simlab.telemetry import Telemetry


def main() -> None:
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
    parser.add_argument(
        "-l",
        "--log-every",
        type=int,
        default=10,
        help="Print telemetry every N steps (default: 10)",
    )
    parser.add_argument(
        "--export-telemetry-csv",
        type=str,
        default=None,
        help="Export telemetry history to CSV file after run",
    )
    parser.add_argument(
        "--export-telemetry-jsonl",
        type=str,
        default=None,
        help="Export telemetry history to JSONL file after run",
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

    # Create telemetry object
    telemetry = Telemetry()

    run_viz(
        world,
        steps=args.steps,
        claim_id=claim_id,
        draw_every=args.draw_every,
        layout_seed=args.layout_seed,
        pause_time=args.pause_time,
        telemetry=telemetry,
        log_every=args.log_every,
    )

    # Export telemetry if requested
    if args.export_telemetry_csv:
        telemetry.export_csv(args.export_telemetry_csv)
        print(f"Telemetry exported to {args.export_telemetry_csv}")
    if args.export_telemetry_jsonl:
        telemetry.export_jsonl(args.export_telemetry_jsonl)
        print(f"Telemetry exported to {args.export_telemetry_jsonl}")


if __name__ == "__main__":
    main()
