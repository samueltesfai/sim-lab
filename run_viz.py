from viz import run_live
import argparse
from sim import init_world


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=15)
    parser.add_argument("-s", "--rng-seed", type=int, default=42)
    parser.add_argument("-t", "--steps", type=int, default=500)
    parser.add_argument("-c", "--claim-id", type=int, default=0)
    parser.add_argument("-d", "--draw-every", type=int, default=1)
    parser.add_argument("-g", "--layout-seed", type=int, default=0)
    parser.add_argument("-p", "--pause-time", type=float, default=0.001)
    args = parser.parse_args()

    world = init_world(num_agents=args.num_agents, rng_seed=args.rng_seed)
    run_live(world, steps=args.steps, claim_id=args.claim_id, draw_every=args.draw_every, layout_seed=args.layout_seed, pause_time=args.pause_time)
    