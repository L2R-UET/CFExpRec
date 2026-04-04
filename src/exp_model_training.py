import json
from pathlib import Path
import torch
import pickle as pkl
from tqdm import tqdm
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rec_model.lightgcn import LightGCN
from rec_model.gformer import GFormer
from rec_model.simgcl import SimGCL
from rec_model.mf import MF
from rec_model.vae import VAE
from rec_model.diffrec import DiffRec
from data_preprocessing import DataHandler

from exp_model.accent import ACCENT
from exp_model.cf_gnnexplainer import CFGNNExplainer
from exp_model.cf2 import CF2
from exp_model.clear import CLEAR
from exp_model.lime_rs import LIME_RS
from exp_model.c2explainer import C2Explainer
from exp_model.shap import SHAP
from exp_model.grease import GREASE
from exp_model.lxr import LXR
from exp_model.prince import PRINCE
from exp_model.unrexplainer import UNRExplainer

from parser import parse_exp_args
from metrics.exp_metrics import (
    pn_s_item_one_instance, pn_s_list_one_instance, pn_r_one_instance, exp_size_one_instance, 
    pos_p_one_instance, neg_p_one_instance, gini_one_instance
)


if __name__ == "__main__":
    args = parse_exp_args()
    Path("logs").mkdir(parents=True, exist_ok=True)
    data = DataHandler(args, mode='exp')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    REC_MODEL_REGISTRY = {
        "MF": MF,
        "VAE": VAE,
        "DiffRec": DiffRec,
        "LightGCN": LightGCN,
        "GFormer": GFormer,
        "SimGCL": SimGCL,
    }

    EXP_MODEL_REGISTRY = {
        "LIME_RS": LIME_RS,
        "SHAP": SHAP,
        "PRINCE": PRINCE,
        "ACCENT": ACCENT,
        "LXR": LXR,
        "GREASE": GREASE,
        "CF-GNNExplainer": CFGNNExplainer,
        "CF2": CF2,
        "CLEAR": CLEAR,
        "C2Explainer": C2Explainer,
        "UNR-Explainer": UNRExplainer,
    }

    rec_model_name_lower = args.rec_model.lower().replace("-", "").replace("_", "")
    rec_config_path = args.rec_config if args.rec_config else f"config/rec_model/{args.dataset}/{rec_model_name_lower}_config.json"
    with open(rec_config_path, "r", encoding="utf-8-sig") as f:
        rec_model_config = json.load(f)
    rec_model = REC_MODEL_REGISTRY[args.rec_model](data, device, args, config=rec_model_config, explain=True)

    state_dict = torch.load(str(Path.cwd()) +f"/checkpoints/{args.rec_model}_{args.dataset}.pth", 
                            weights_only=True, map_location=device)
    rec_model.load_state_dict(state_dict)
    rec_model.to(device)
    rec_model.eval()

    exp_model_name_lower = args.exp_model.lower().replace("-", "").replace("_", "")
    exp_config_path = args.exp_config if args.exp_config else f"config/exp_model/{args.dataset}/{args.rec_model}/{exp_model_name_lower}_config.json"
    if Path(exp_config_path).exists():
        with open(exp_config_path, "r", encoding="utf-8-sig") as f:
            exp_model_config = json.load(f)
    else:
        exp_model_config = dict()

    exp_model = EXP_MODEL_REGISTRY[args.exp_model](rec_model, device, args, config=exp_model_config)

    print(f"Evaluating {args.exp_model} on {args.dataset} with {args.rec_model} in {args.level}-level explanation and top-{args.top_k} recommendation.")

    if exp_model.mode == 'explicit' or exp_model.mode == 'hybrid':
        explicit_output_eval = []
        for _, batch in tqdm(enumerate(data.test_loader)):
            user_id = batch.item()
            cf_exp, time = exp_model.explain(user_id, mode="explicit", level=args.level, graph_perturb=args.graph_perturb)
            if args.level == 'item':
                topk_with_cf_exp = []
                for cf in cf_exp:
                    pred_indices = rec_model.predict(user_id, args.top_k, 1 - cf)[1]
                    if pred_indices.dim() > 1:
                        pred_indices = pred_indices.squeeze(0)
                    topk_with_cf_exp.append(pred_indices)
                pn_s = [
                    pn_s_item_one_instance(
                        exp_model.y_pred_indices[user_id].tolist(),
                        topk_with_cf_exp[i].tolist(),
                        i,
                        args.top_k,
                    )
                    for i in range(len(topk_with_cf_exp))
                ]
                pn_r = [0 for _ in topk_with_cf_exp]
                exp_size = [exp_size_one_instance(cf) for cf in cf_exp]
            else:
                topk_with_cf_exp = rec_model.predict(user_id, args.top_k, 1 - cf_exp)[1]
                pn_s = pn_s_list_one_instance(exp_model.y_pred_indices[user_id].tolist(), topk_with_cf_exp.tolist(), args.top_k)
                pn_r = pn_r_one_instance(exp_model.y_pred_indices[user_id].tolist(), topk_with_cf_exp.tolist(), args.top_k)
                exp_size = exp_size_one_instance(cf_exp)
            explicit_output_eval.append({
                "pn_s": pn_s,
                "pn_r": pn_r,
                "#perturb": exp_size,
                "time": time,
            })

        final_result = {
            "pn_s": [],
            "pn_r": [],
            "#perturb": [],
            "time": [],
        }
        for i in range(len(explicit_output_eval)):
            for key in final_result:
                value = explicit_output_eval[i][key]
                if isinstance(value, list):
                    final_result[key].extend(value)
                else:
                    final_result[key].append(value)

        print(f"Final Explicit Performance:")
        for key, values in final_result.items():
            print(f"  {key.capitalize()}: {np.mean(values)} +- {np.std(values)}")

        with open(f"logs/{args.rec_model}_{args.exp_model}_{args.dataset}_top{args.top_k}_{args.level}_exp.pkl", "wb") as f:
            pkl.dump(explicit_output_eval, f)

    if exp_model.mode == 'implicit' or exp_model.mode == 'hybrid':
        implicit_output_eval = []
        for _, batch in tqdm(enumerate(data.test_loader)):
            user_id = batch.item()
            cf_output, time = exp_model.explain(user_id, mode="implicit", level=args.level, graph_perturb=args.graph_perturb)
            history_mask = exp_model.ui_mat[user_id] > 0
            if args.level == 'item':
                pos_p = [pos_p_one_instance(rec_model, exp_model.y_pred_indices[user_id].tolist(), cf_output[i], user_id, i) for i in range(len(cf_output))]
                neg_p = [neg_p_one_instance(rec_model, exp_model.y_pred_indices[user_id].tolist(), cf_output[i], user_id, i) for i in range(len(cf_output))]
                gini = [gini_one_instance(cf_output[i]) for i in range(len(cf_output))]
            else:
                pos_p = pos_p_one_instance(rec_model, exp_model.y_pred_indices[user_id].tolist(), cf_output, user_id)
                neg_p = neg_p_one_instance(rec_model, exp_model.y_pred_indices[user_id].tolist(), cf_output, user_id)
                gini = gini_one_instance(cf_output)
            implicit_output_eval.append({
                "pos_p": pos_p,
                "neg_p": neg_p,
                "gini": gini,
                "time": time,
            })
        
        final_result = {
            "pos_p": [],
            "neg_p": [],
            "gini": [],
            "time": [],
        }
        for i in range(len(implicit_output_eval)):
            for key in final_result:
                value = implicit_output_eval[i][key]
                if isinstance(value, list):
                    final_result[key].extend(value)
                else:
                    final_result[key].append(value)

        print("Final Implicit Performance:")
        for key, values in final_result.items():
            print(f"  {key.capitalize()}: {np.mean(values)} +- {np.std(values)}")

        with open(f"logs/{args.rec_model}_{args.exp_model}_{args.dataset}_top{args.top_k}_{args.level}_imp.pkl", "wb") as f:
            pkl.dump(implicit_output_eval, f)