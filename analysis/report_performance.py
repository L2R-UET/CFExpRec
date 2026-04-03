import argparse
import pickle as pkl

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate explanation performance of recommendation models.")
    parser.add_argument('--rec_model', type=str, required=True, help='Recommendation model to evaluate (e.g., LightGCN, GFormer).')
    parser.add_argument('--exp_model', type=str, required=True, help='Explanation model to evaluate (e.g., ACCENT, LXR).')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for evaluation (e.g., Amazon-Books, Yelp2018).')
    parser.add_argument('--top_k', type=int, default=5, help='Top K items to consider for evaluation metrics.')
    parser.add_argument('--level', type=str, choices=['item', 'list'], default='item', help='Evaluation level: item or list.')
    parser.add_argument('--format', type=str, choices=['exp', 'imp'], default='exp', help='Format of explanation: exp (explicit) or imp (implicit).')
    parser.add_argument('--graph_perturb', type=str, choices=['full', 'khop', 'indirect', 'user'], default="khop", help='Different graph perturbation.')
    parser.add_argument('--export_log', action='store_true', help='Whether to export the evaluation log to a file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Evaluating {args.exp_model} explanations for {args.rec_model} on {args.dataset} with top_k={args.top_k}, level={args.level}, format={args.format}, graph_perturb={args.graph_perturb}.")
    
    '''
    Format:
    [{"pn_hit": 1, "pn_ndcg": 1, "gini": 0.5, "#perturb": 10, "time": 0.5}, 
     {"pn_hit": 0, "pn_ndcg": 0, "gini": 0.3, "#perturb": 8, "time": 0.4}]
    '''

    graph_perturb = "" if args.graph_perturb == "khop" else args.graph_perturb

    with open(f"logs/{args.rec_model}_{args.exp_model}_{args.dataset}_top{args.top_k}_{args.level}_{args.format}_{graph_perturb}.pkl", "rb") as f:
        result_by_user = pkl.load(f)
    
    final_log = {metric: 0 for metric in result_by_user[0].keys()}

    if args.level == 'item':
        final_log_by_item = {metric: [0] * args.top_k for metric in result_by_user[0].keys()}

    for user_results in result_by_user:
        for metric, value in user_results.items():
            value_agg = sum(value) / len(value) if isinstance(value, list) else value
            final_log[metric] += value_agg
            if args.level == 'item':
                for i in range(args.top_k):
                    final_log_by_item[metric][i] += value[i]
    
    num_users = len(result_by_user)
    for metric in final_log.keys():
        final_log[metric] /= num_users
        if args.level == 'item':
            final_log_by_item[metric] = [x / num_users for x in final_log_by_item[metric]]

    if args.export_log:
        log_filename = f"logs/{args.rec_model}_{args.exp_model}_{args.dataset}_top{args.top_k}_{args.level}_{args.format}_{graph_perturb}_agg.pkl"
        print(f"Exporting evaluation log to {log_filename}.")
        output_pkl = {"overall": final_log, "by_item": final_log_by_item} if args.level == 'item' else {"overall": final_log}
        print(output_pkl)
        with open(log_filename, "wb") as f:
            pkl.dump(output_pkl, f)
            
    else:
        print("Final Evaluation Log:")
        print(final_log)
        if args.level == 'item':
            print("Evaluation Log by Item Position:")
            for metric, values in final_log_by_item.items():
                print(f"{metric}: {values}")