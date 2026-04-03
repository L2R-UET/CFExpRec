import argparse
from analysis.helper import DifferentGraphVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate consistency of explanations across different graph perturbation levels.")
    parser.add_argument('--level', type=str, choices=['item', 'list'], default='item', help='Evaluation level: item or list.')
    parser.add_argument('--dataset', type=str, default="Amazon", help='Dataset to use for evaluation (e.g., Amazon-Books, Yelp2018).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    visualizer = DifferentGraphVisualizer(args.dataset, args.level)
    visualizer.visualize()