'''
Based on the original implementation of paper "Towards Explaining Recommendations Through Local Surrogate Models"
and the implementation from the LXR repository: https://github.com/DeltaLabTLV/LXR
Original repository: https://github.com/caionobrega/explaining-recommendations
'''

import torch
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from .base_model import UserVectorExpBaseModel

class LIME_RS(UserVectorExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        if config is None:
            config = {}
        super(LIME_RS, self).__init__(rec_model, device, args, config)
        self.n_items = rec_model.n_items
        self.mode = 'implicit'
        self.n_samples = config.get('n_samples', 150)
        self.kernel_width = config.get('kernel_width', None)
        self.num_features = config.get('num_features', 10)
        self.min_pert = config.get('min_pert', 50)
        self.max_pert = config.get('max_pert', 100)
        # LXR baseline uses highest_weights in evaluation notebooks.
        self.feature_selection_mode = 'highest_weights'
        self.random_state = check_random_state(config.get('random_state', None))
        self.verbose = config.get('verbose', False)
        self.to(device)

    def _kernel_fn(self, distances):
        sum_distances = np.sum(distances)
        if sum_distances == 0:
            return np.ones_like(distances)
        return np.array([1 - distances[i] / sum_distances for i in range(len(distances))])

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector, weighted_labels, max_iter=15, eps=2.220446049250313e-7, method='lasso', verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]], labels, sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(zip(range(data.shape[1]), weighted_data), key=lambda x: np.abs(x[1]), reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weights = np.asarray(weights)
            weighted_data = ((data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights)) * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights, num_features, n_method)

    def _get_perturbations(self, user_vector, n_samples, seed=None):
        neighborhood_data = [user_vector.copy()]
        distances = [0]

        if seed is None:
            rng = self.random_state
        else:
            rng = np.random.RandomState(seed)

        user_indices = np.where(user_vector > 0)[0]
        non_user_indices = np.where(user_vector == 0)[0]

        for _ in range(n_samples):
            neighbor = user_vector.copy()
            dist = rng.randint(self.min_pert, high=self.max_pert)

            pos = min(rng.randint(0, high=dist), int(np.sum(user_vector)))
            neg = dist - pos

            neg_locations = rng.choice(non_user_indices, int(neg))
            for l in neg_locations:
                neighbor[l] = 1

            pos_locations = rng.choice(user_indices, int(pos))
            for l in pos_locations:
                neighbor[l] = 0

            neighborhood_data.append(neighbor)
            distances.append(dist)

        neighborhood_data = np.array(neighborhood_data)
        return neighborhood_data, distances

    def _compute_lime_scores(self, user_id, user_vector, target_items, level='list'):
        user_vector_perturb = user_vector.copy()
        for item_id in target_items:
            user_vector_perturb[item_id] = 0

        perturb_seed = int(target_items[0]) if len(target_items) == 1 else None

        neighborhood_data, distances = self._get_perturbations(user_vector_perturb, self.n_samples, seed=perturb_seed)

        pert_tensor = torch.tensor(neighborhood_data, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.device, dtype=torch.long).expand(len(pert_tensor))
            user_history = self.ui_mat[user_id].unsqueeze(0).expand_as(pert_tensor)
            corrected = user_history * pert_tensor + (1 - user_history) * (1 - pert_tensor)
            neighborhood_labels = self.rec_model.predict(
                users=user_tensor,
                mask=[corrected[i] for i in range(len(corrected))]
            )
            if neighborhood_labels.ndim == 2:
                neighborhood_labels = neighborhood_labels.cpu().numpy()
            else:
                neighborhood_labels = neighborhood_labels.cpu().numpy()

        weights = self._kernel_fn(np.array(distances))

        num_features = int(np.sum(user_vector > 0))

        if level == 'list':
            labels_column = np.zeros(len(neighborhood_labels))
            for item_id in target_items:
                labels_column += neighborhood_labels[:, item_id]

            used_features = self.feature_selection(neighborhood_data, labels_column, weights, num_features, self.feature_selection_mode)

            model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
            model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

            return used_features, model.coef_
        else:
            n_targets = len(target_items)
            all_coefs = np.zeros((n_targets, self.n_items))

            for i in range(n_targets):
                item_id = target_items[i]
                labels_column = neighborhood_labels[:, item_id]

                used_features = self.feature_selection(neighborhood_data, labels_column, weights, num_features, self.feature_selection_mode)

                model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
                model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

                full_coef = np.zeros(self.n_items)
                full_coef[used_features] = model.coef_
                all_coefs[i] = full_coef

            return None, all_coefs

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)

        user_interactions_np = user_interaction.cpu().numpy().reshape(1, -1)
        user_vector = user_interactions_np[0].copy()
        history_mask = user_vector > 0
        history_indices = np.where(history_mask)[0]

        target_items = item_ids
        if not isinstance(target_items, (list, np.ndarray)):
            target_items = [target_items]

        if len(history_indices) == 0:
            return torch.zeros((len(target_items), self.n_items), device=self.device).squeeze(0)

        if len(target_items) > 1:
            select_indices, coefficients = self._compute_lime_scores(user_id, user_vector, target_items, level='list')
            full_coef = np.zeros(self.n_items)
            full_coef[select_indices] = coefficients
            scores_tensor = torch.tensor(full_coef, dtype=torch.float32, device=self.device)
        else:
            select_indices, coefficients = self._compute_lime_scores(user_id, user_vector, target_items, level='item')
            scores_tensor = torch.tensor(coefficients, dtype=torch.float32, device=self.device).squeeze(0)

        mask = torch.zeros_like(scores_tensor)
        mask[..., history_indices] = 1.0
        scores_tensor = scores_tensor * mask
        scores_tensor += user_interaction * 1e-12
        return scores_tensor