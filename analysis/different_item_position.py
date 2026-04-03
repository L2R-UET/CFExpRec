import argparse
from analysis.helper import UserVectorPositionVisualizer, GraphPositionVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate consistency of explanations across different levels and recommenders.")
    parser.add_argument('--perturb_scope', type=str, choices=['vector', 'graph'], default='vector', help='Scope of perturbation for consistency evaluation: vector or graph.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.perturb_scope == "vector":
        visualizer = UserVectorPositionVisualizer()
    elif args.perturb_scope == "graph":
        visualizer = GraphPositionVisualizer()
    else:
        raise ValueError("Unknown perturbation scope: %s" % args.perturb_scope)
    visualizer.visualize()