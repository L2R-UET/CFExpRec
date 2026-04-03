import argparse

def parse_rec_args():
    parser = argparse.ArgumentParser(description="Rec Model Training")
    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model to train (e.g., LightGCN, NGCF)')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_train', type=int, default=-1, help='Batch size for training')
    parser.add_argument('--batch_test', type=int, default=-1, help='Batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--top_k', type=int, default=10, help='Top K for evaluation metrics')   
    parser.add_argument('--epoch_pbar', action='store_true', help='Progress bar for each batch')
    parser.add_argument('--patience', type=int, default=30, help='Patience for training')
    parser.add_argument('--wandb', action='store_true', help='Wandb for logging or not')
    
    args = parser.parse_args()
    return args

def parse_exp_args():
    parser = argparse.ArgumentParser(description="Exp Model Training")
    parser.add_argument('--rec_config', type=str, help='Path to the JSON config file for recommender')
    parser.add_argument('--exp_config', type=str, help='Path to the JSON config file for explainer')
    parser.add_argument('--rec_model', type=str, default='LightGCN', help='Rec model (e.g., LightGCN, NGCF)')
    parser.add_argument('--exp_model', type=str, default='LXR', help='Exp model to train (e.g. LXR, CF-GNNExplainer)')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--batch_train', type=int, default=1, help='Batch size for training')
    parser.add_argument('--batch_test', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--top_k', type=int, default=10, help='Top K for evaluation metrics')
    parser.add_argument('--level', type=str, choices=['item', 'list'], default='list', help='Explanation level: item or list')
    parser.add_argument('--graph_perturb', type=str, default='khop', choices=['full', 'khop', 'indirect', 'user_only'], help='Method for graph perturbation in explanation')
    parser.add_argument('--test_users', type=int, default=None)
    args = parser.parse_args()
    return args