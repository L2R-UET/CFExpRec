import json

import torch
from tqdm import tqdm
import numpy as np
import random
from parser import parse_rec_args
import wandb
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rec_model.mf import MF
from rec_model.vae import VAE
from rec_model.lightgcn import LightGCN
from rec_model.gformer import GFormer
from rec_model.simgcl import SimGCL
from rec_model.diffrec import DiffRec

from data_preprocessing import DataHandler

if __name__ == "__main__":

    seed_value = 42 
    
    load_dotenv()
    args = parse_rec_args()
    
    seed_value = args.seed if hasattr(args, 'seed') else 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    wandb_key = os.getenv("WANDB_API_KEY")
    if args.wandb and wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project="GraphCFRec",
            name=f"{args.model}_{args.dataset}",
            config=vars(args)
        )

    exp_dir = f"logs/{args.model}_{args.dataset}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Exp: {exp_dir} | Model: {args.model}")
    
    Path("checkpoints").mkdir(parents=True, exist_ok=True)


    data = DataHandler(args, mode='rec')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_REGISTRY = {
        'LightGCN': LightGCN, 
        'GFormer': GFormer, 
        'MF': MF, 
        'VAE': VAE, 
        'DiffRec': DiffRec, 
        'SimGCL': SimGCL
    }

    rec_model_name_lower = args.model.lower().replace("-", "").replace("_", "")
    config_path = args.config if args.config else f"config/rec_model/{args.dataset}/{rec_model_name_lower}_config.json"
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8-sig") as f:
            rec_model_config = json.load(f)
    else:
        rec_model_config = dict()

    model = MODEL_REGISTRY[args.model](data, device, args, config=rec_model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_list_epoch = []
    recall_list = []
    ndcg_list = []

    wait = 0
    best_ndcg = -1
    best_recall = -1

    pbar = tqdm(range(args.epochs), desc="Training")

    for epoch in pbar:
        final_loss_list = model.train_epoch(optimizer)
        avg_loss = final_loss_list if isinstance(final_loss_list, float) else np.mean(final_loss_list)
        
        loss_list_epoch.append(round(avg_loss, 4))
        test_recall, test_ndcg = model.test_epoch(args.top_k)
        
        recall_list.append(round(test_recall, 4))
        ndcg_list.append(round(test_ndcg, 4))

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            recall=f"{test_recall:.4f}",
            ndcg=f"{test_ndcg:.4f}",
        )

        if test_ndcg > best_ndcg and test_recall > best_recall:
            print(f"Epoch {epoch+1}: New best model found! NDCG@{args.top_k}: {test_ndcg:.4f}, Recall@{args.top_k}: {test_recall:.4f}")
            best_ndcg = test_ndcg
            best_recall = test_recall

            ckpt_path = f"checkpoints/{args.model}_{args.dataset}.pth"
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
        else:
            wait += 1

        if args.wandb and wandb_key:
            wandb.log({'loss': avg_loss,
                       f'recall_{args.top_k}': test_recall,
                       f'ndcg_{args.top_k}': test_ndcg})
        
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    ckpt_path = f"checkpoints/{args.model}_{args.dataset}.pth"
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best model: {ckpt_path}")

    print('-'*20)
    print(f'{args.model} with {args.dataset}:')
    for topk in [3, 5, 10]:
        r, n = model.test_epoch(topk)
        print(f'Recall@{topk}: {r:.4f}, NDCG@{topk}: {n:.4f}')
    
    if args.wandb and wandb_key:
        wandb.finish()