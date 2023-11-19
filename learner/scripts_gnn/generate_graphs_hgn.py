import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from representation import REPRESENTATIONS
from dataset.graphs_hgn import gen_graph_rep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str, help="domain to generate (useful for debugging)")
    parser.add_argument('--regenerate', action="store_true")
    parser.add_argument('--step', type=int, default=1, help="how deep is the pairs (default 1)")
    args = parser.parse_args()

    gen_graph_rep(representation="hgn",
                  regenerate=args.regenerate,
                  domain=args.domain,
                  step=args.step)
