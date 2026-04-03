# From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems
**Accepted at SIGIR'26 - Reproducibility Track**
<!-- ## Guide for running -->

## Overview 
This repository contains the implementation of recommendation models, counterfactual explanation methods, and their evaluation used in our research paper accepted at SIGIR 2026 (Reproducibility Track).

The codebase supports multiple recommendation backbones and explanation models across several benchmark datasets.

```python
.
├── checkpoints/                      # Stores trained model checkpoints (recommendation & explanation models)
├── processed_data/                   # Contains preprocessed datasets ready for training and evaluation
├── logs/                             # Contains evaluation logs after running explainers
├── config/                           # Configuration of recommenders and explainers
│   ├── exp_model/                    # Explainer config
│   │   ├── aaa/                      # Explainer config with regard to dataset aaa
│   │   │   ├── bbb/                  # Explainer config with regard to recommender bbb
│   │   │   │   ├── ccc_config.json   # Explainer config for explainers ccc
│   │   │   │   └── ...
│   │   │   └── ...  
│   │   └── ...    
│   └── rec_model/                    # Recommender config
│       ├── mmm/                      # Recommender config with regard to dataset aaa
│       │   ├── nnn_config.json       # Explainer config for recommender nnn
│       │   └── ...  
│       └── ...                 
├── analysis/
│   ├── report_performance.py         # Report overall performance
│   ├── consistency.py                # Report consistency over different evaluation level/ recommenders
│   ├── different_item_position.py    # Report stability over different item position
│   └── graph_perturb.py              # Report performance with different graph perturb
├── src/
│   ├── exp_model/                    # Counterfactual explanation model implementations
│   │   ├── base_model.py             # Base class defining the common interface for explanation models
│   │   ├── xxx.py                    # Implementation of explainers xxx (e.g. clear.py is for CLEAR)
│   │   └── ...                  
│   ├── metrics/                      # Evaluation metrics for both recommendation and explanation tasks
│   │   ├── exp_metrics.py            # Metrics for evaluating explanation quality
│   │   └── rec_metrics.py            # Metrics for evaluating recommendation performance (e.g., Recall, NDCG)
│   ├── rec_model/                    # Recommendation model implementations
│   │   ├── base_model.py             # Base class defining the common interface for recommendation models
│   │   ├── zzz.py                    # Implementation of recommendation model zzz (e.g. lightgcn.py is for LightGCN)
│   │   └── ...                  
│   ├── data_preprocessing.py         # Dataset loading, preprocessing, and graph construction utilities
│   ├── exp_model_training.py         # Script for training counterfactual explainers
│   ├── parser.py                     # Centralized configuration and hyperparameter management
│   └── rec_model_training.py         # Script for training recommendation models
├── .env                              # Environment configuration file (e.g., WANDB_API_KEY)
├── requirements.txt                  # Python library requirements
└── README.md                         # Project documentation and reproduction instructions

```

## Dataset
Three datasets (Amazon, ML1M, Pinterest) used in the study are available in the folder ```./processed_data```. The datasets are preprocessed and ready to use. For ML1M and Pinterest, we follow the data preprocessing code in the [LXR repository](https://github.com/DeltaLabTLV/LXR). For Amazon, we include the data preprocessing code in ```./processed_data/Amazon/preprocessing.ipynb``` file.

Each dataset contains a file named ```interaction.csv``` to store user-item interaction, with each row (user_id, item_id) corresponding to an interaction.

## Environment Setup
### 1. Create Environment File
Create `.env` file in the root directory and add your Weights & Biases API key:

```env
WANDB_API_KEY=your_api_key_here
```
Make sure all required dependencies are installed before running experiments.

### 2. Install necessary packages
Python version 3.11 is recommended to avoid unexpected bugs.

Run the following command to install the dependencies:
```bash
pip install -r requirements.txt
```

## Recommendation Model Training
All recommendation models can be trained using:
```bash
python src/rec_model_training.py [arguments]
```

### 1. Amazon

```bash
python src/rec_model_training.py --model MF --dataset Amazon --epochs 100 --lr 0.01 --batch_train 64
python src/rec_model_training.py --model VAE --dataset Amazon --epochs 200 --lr 0.001 --batch_train 1024
python src/rec_model_training.py --model DiffRec --dataset Amazon --epochs 100 --lr 0.01 --batch_train 512 --batch_test 2048
python src/rec_model_training.py --model LightGCN --dataset Amazon --epochs 100 --lr 0.01 --batch_train 256
python src/rec_model_training.py --model GFormer --dataset Amazon --batch_train 1024 --lr 0.001 --epochs 100
python src/rec_model_training.py --model SimGCL --dataset Amazon --batch_train 64 --lr 0.001
```

### 2. ML1M

```bash
python src/rec_model_training.py --model MF --dataset ML1M --epochs 100 --lr 0.001 --batch_train 2048 --batch_test 2048
python src/rec_model_training.py --model VAE --dataset ML1M --epochs 120 --lr 0.001 --batch_train 2048 --batch_test 2048
python src/rec_model_training.py --model DiffRec --dataset ML1M --epochs 30 --lr 0.001 --batch_train 512 --batch_test 2048
python src/rec_model_training.py --model LightGCN --dataset ML1M --lr 0.01 --batch_train 2048
python src/rec_model_training.py --model GFormer --dataset ML1M --batch_train 65536
python src/rec_model_training.py --model SimGCL --dataset ML1M --batch_train 2048
```

### 3. Yahoo

```bash
python src/rec_model_training.py --model MF --dataset Yahoo --epochs 100 --lr 0.001 --batch_train 2048 --batch_test 2048
python src/rec_model_training.py --model VAE --dataset Yahoo --epochs 120 --lr 0.001 --batch_train 2048 --batch_test 2048
python src/rec_model_training.py --model DiffRec --dataset Yahoo --epochs 30 --lr 0.001 --batch_train 512 --batch_test 2048
python src/rec_model_training.py --model LightGCN --dataset Yahoo --lr 0.01 --batch_train 2048
python src/rec_model_training.py --model GFormer --dataset Yahoo --batch_train 65536
python src/rec_model_training.py --model SimGCL --dataset Yahoo --batch_train 2048
```

## Counterfactual Explanation Evaluation

Run the following command to get evaluation logs for the explainer with the default configuration:

```bash
python src/exp_model_training.py --exp_model XXX --rec_model YYY --dataset ZZZ --top_k {3/5} --level {item/list} --graph_perturb {full/khop/indirect/user_only}
```

If configuration changes are required, please modify the settings in ```./config``` folder, where all default configurations are provided. All the evaluation logs will be stored under the ```./logs``` folder.

## Guide for Reproducibility
Before running the below commands, please first generate the evaluation logs for all counterfactual explainers with the guide above.

### 1. Overall Performance and Evaluation Level Consistency (Section 4.1, 4.2 and 4.6)

Run the following command to generate the final results for each experiment:

```bash
python analysis/report_performance.py --dataset XXX --rec_model YYY --exp_model ZZZ --top_k {3/5} --level {item/list} --format {exp/imp} --graph_perturb {full/khop/indirect/user}
```

To run visualization for other sections, you need to first export logs by running the above command with ```--export_log``` added.

### 2. Recommender Consistency Visualization (Section 4.3)

For the visualization with different recommenders (Section 4.3), run the following command:

```bash
python analysis/rec_consistency.py --perturb_scope {vector/graph}
```

### 3. Stability across Different Item Position (Section 4.4)

```bash
python analysis/different_item_position.py --perturb_scope {vector/graph}
```

### 4. Subgraph vs Full Graph (Section 4.5)

```bash
python analysis/graph_perturb.py --level {item/list}
```


## Notes
- Ensure the recommendation model is trained before running explanation models.
- For large-batch configurations (e.g., GFormer), sufficient GPU memory is required.
- Results are logged via Weights & Biases.
- For reproducibility, we provide all recommendation model checkpoints in `checkpoints/` folder.
