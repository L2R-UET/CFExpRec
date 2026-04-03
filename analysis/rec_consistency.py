import argparse
from analysis.helper import UserVectorVisualizer, GraphVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate consistency of explanations across different levels and recommenders.")
    parser.add_argument('--perturb_scope', type=str, choices=['vector', 'graph'], default='vector', help='Scope of perturbation for consistency evaluation: vector or graph.')
    parser.add_argument('--dataset', type=str, default="Amazon", help='Dataset to use for evaluation (e.g., Amazon-Books, Yelp2018).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.perturb_scope == "vector":
        visualizer = UserVectorVisualizer(args.dataset)
    elif args.perturb_scope == "graph":
        visualizer = GraphVisualizer(args.dataset)
    else:
        raise ValueError("Unknown perturbation scope: %s" % args.perturb_scope)
    visualizer.visualize()