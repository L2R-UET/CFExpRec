# Documentation for explanation model implementation

## Initialization
Inherit ```UserVectorExpBaseModel``` class for user vector as input, ```GraphExpBaseModel``` class for user-item graph as input (in ```base_model.py```).

## Compulsory components
- In the ```__init__``` function, specify the explanation type: define ```self.mode``` as ```implicit```, ```explicit``` or ```hybrid``` (support both implicit and explicit). Provide the configuration under each JSON file under the ```./config/exp_model``` folder. That config will be passed through the ```config``` variable when explainer is initialized.
- Implement ```get_implicit_explanation``` function for implicit explanation, ```get_explicit_explanation``` function for explicit explanation. At the start of the function, specify the interaction scope via ```get_historical_interactions``` function:

```python
# UserVectorExpBaseModel
def get_explicit_explanation(self, user_id, item_ids, **kwargs):
    interaction = self.get_historical_interactions(user_id, item_ids)
    ...

# GraphExpBaseModel
def get_explicit_explanation(self, user_id, item_ids, **kwargs):
    interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
    ...
```

## Details
### 1. ```rec_model/base_model.py```
#### RecBaseModel

PROPERTIES:
- ```args```: ```Namespace``` object storing comment arguments of all recommenders passing from command line (e.g. ```args.epochs, args.lr```)
- ```config```: ```dict``` object loaded from a JSON config file, storing the hyperparameters for individual recommender
- ```data_handler```: ```DataHandler``` object storing preprocessed data and dataloader
- ```n_users```: ```int``` storing the number of users in the data
- ```n_items```: ```int``` storing the number of items in the data
- ```ui_mat```: ```torch.Tensor``` or ```torch.sparse.Tensor``` storing user-item interaction matrix of shape (n_users, n_items)

FUNCTIONS:

```apply_mask(ui_mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor```

Apply masking to the user-item matrix to remove some interactions

Params:
- ```ui_mat``` (```torch.Tensor```): user-item interaction matrix
- ```mask``` (```torch.Tensor```): a 0/1 matrix with the same shape as ```ui_mat```, 1 indicating keeping interaction, 0 indicating removing interaction

```predict(users, topk: int, mask: torch.Tensor)```

Give prediction scores for all items and/or top-k predicted items for each user in ```users```

Params:
- ```users``` (```None``` or ```int``` or ```list```): 
user(s) to predict; if ```None```, default is prediction for all users
- ```topk``` (```None``` or ```int```): the number of items to obtain in top; if ```None```, only output the prediction score, top-k prediction is not outputed
- ```mask``` (```None``` or ```torch.Tensor```): masking over original interaction; if ```None```, the original interaction is used

Return:
```torch.Tensor``` with shape ```(#users to predict, self.n_items)``` or ```(self.n_items, )``` (for 1 user only), and/or ```torch.Tensor``` with shape ```(#users to predict, topk)``` or ```(topk, )``` (for 1 user only)


### 2. ```exp_model/base_model.py```
#### ExpBaseModel

PROPERTIES:
- ```rec_model```: ```RecBaseModel``` object storing the recommendation model
- ```args```: ```Namespace``` object storing common arguments of all explainers passing from command line (e.g. ```args.dataset, args.graph_perturb```)
- ```config```: ```dict``` object loaded from a JSON config file, storing the hyperparameters for individual explainer
- ```mode```: explanation type that explainer can support, either ```explicit```, ```implicit```, or ```hybrid```(MUST BE SPECIFIED!!!)
- ```ui_mat```: ```torch.Tensor``` or ```torch.sparse.Tensor``` storing user-item interaction matrix of shape (n_users, n_items)
- ```y_pred_scores```: ```torch.Tensor``` of shape (n_users, n_items) storing the prediction scores of each item corresponding to each user
- ```y_pred_indices```: ```torch.Tensor``` of shape (n_users, top_k) storing the top-k item indices of all users

FUNCTIONS:

```get_implicit_explanation(user_id, item_ids, **kwargs) -> torch.Tensor```

Return implicit explanation for target user and target item or top-k item list

Params:
- ```user_id``` (```int```): target user to be explained
- ```item_ids``` (```int``` or ```list```): target item(s) to be explained
- ```graph_perturb``` (```str```, for ```GraphExpBaseModel```): ```full```, ```khop```, ```indirect``` or ```user_only```, specifying the perturbation scope for graph explainer

Return:
```torch.Tensor``` with shape ```(n_items, )``` for ```UserVectorExpBaseModel```, ```(n_users, n_items)``` for ```GraphExpBaseModel```, in which value in one entry > 0 indicates the importance of the historical interaction with regard to the target item/list, and = 0 indicates the entry not belonging to any historical interaction.

```get_explicit_explanation(user_id, item_ids, **kwargs) -> torch.Tensor```

Return explicit explanation for target user and target item or top-k item list

Params:
- ```user_id``` (```int```): target user to be explained
- ```item_ids``` (```int``` or ```list```): target item(s) to be explained
- ```graph_perturb``` (```str```, for ```GraphExpBaseModel```): ```full```, ```khop```, ```indirect``` or ```user_only```, specifying the perturbation scope for graph explainer

Return:
A 0/1 ```torch.Tensor``` with shape ```(n_items, )``` for ```UserVectorExpBaseModel```, ```(n_users, n_items)``` for ```GraphExpBaseModel```, where value 1 indicates that the interaction is counterfactual to be found, and value 0 otherwise.

```convert_cf_list_to_mask(cf_list: list) -> torch.Tensor```

Convert a list of indices into a multi-hot vector/matrix, useful when the output is a list of counterfactual item IDs or a list of interactions in a graph

Params:
- ```cf_list``` (```list```): list of indices, shape ```(#indices, )``` for ```UserVectorExpBaseModel```, shape ```(2, #indices)``` for ```GraphExpBaseModel```

Return:
A 0/1 ```torch.Tensor``` with 1 for indices in ```cf_list```, 0 otherwise

```flip_mask(cf: torch.Tensor) -> torch.Tensor```

Flip 0 to 1 and 1 to 0 in a mask

Params:
- ```cf``` (```torch.Tensor```): 0/1 tensor

Return: 0/1 ```torch.Tensor``` with value flipped

## Notes
- Only implement your explainer(s), DO NOT MAKE ANY CHANGES TO OTHER CLASSES OR MAIN FUNCTION. Please create an issue or directly contact the administrator (huynq2k4) if you require to make any changes or fix any bugs in other code snippets.
- For implicit explanations, for all entries corresponding to the historical interaction, set value to 1e-9 if the importance score of that interaction is exactly 0. The value 0 is reserved for entries not corresponding to any historical interaction.
