import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import pandas as pd
import numpy as np
import pickle as pkl

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, base_pad=-10, side_pad=200):
            self.set_thetagrids(np.degrees(theta), labels)

            for angle, label in zip(theta, self.get_xticklabels()):
                angle_deg = np.degrees(angle) % 360
                pad = base_pad
                if 45 < angle_deg < 135 or 225 < angle_deg < 315:
                    pad = side_pad
                elif angle_deg % 90 != 0:
                    pad = 100
                label.set_fontsize(14)
                label.set_verticalalignment('center')
                label.set_horizontalalignment('center')
                label.set_y(label.get_position()[1] - pad / 1000)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                            path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

class UserVectorVisualizer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df_item, self.df_list = self.load_data()

    def load_data(self):
        exp_model_list = ['LIME-RS', 'SHAP', 'ACCENT', 'LXR']
        data_item = {
            "model": exp_model_list,
            "POS_P@3-MF": [0] * len(exp_model_list),
            "POS_P@5-MF": [0] * len(exp_model_list),
            "POS_P@3-VAE": [0] * len(exp_model_list),
            "POS_P@5-VAE": [0] * len(exp_model_list),
            "POS_P@3-DiffRec": [0] * len(exp_model_list),
            "POS_P@5-DiffRec": [0] * len(exp_model_list),
        }
        data_list = {
            "model": exp_model_list,
            "POS_P@3-MF": [0] * len(exp_model_list),
            "POS_P@5-MF": [0] * len(exp_model_list),
            "POS_P@3-VAE": [0] * len(exp_model_list),
            "POS_P@5-VAE": [0] * len(exp_model_list),
            "POS_P@3-DiffRec": [0] * len(exp_model_list),
            "POS_P@5-DiffRec": [0] * len(exp_model_list),
        }
        for rec_model in ['MF', 'VAE', 'DiffRec']:
            for i, exp_model in enumerate(exp_model_list):
                for top_k in [3, 5]:
                    with open(f"logs/{rec_model}_{exp_model}_{self.dataset_name}_top{top_k}_item_imp_agg.pkl", "rb") as f:
                        tmp_data = pkl.load(f)["overall"]
                        data_item[f"POS_P@{top_k}-{rec_model}"][i] = tmp_data['pos_p']
                    with open(f"logs/{rec_model}_{exp_model}_{self.dataset_name}_top{top_k}_list_imp_agg.pkl", "rb") as f:
                        tmp_data = pkl.load(f)["overall"]
                        data_list[f"POS_P@{top_k}-{rec_model}"][i] = tmp_data['pos_p']
                        
        df_item = pd.DataFrame(data_item)
        df_list = pd.DataFrame(data_list)
        return df_item, df_list

    def visualize(self):
        """Plot spider chart for one dataset with item-level and list-level"""
        dataset_name = self.dataset_name
        df_item = self.df_item
        df_list = self.df_list
        
        df_item_filtered = df_item[df_item['dataset'] == dataset_name].copy()
        df_list_filtered = df_list[df_list['dataset'] == dataset_name].copy()
        
        selected_cols = [
            'POS_P@3-MF', 'POS_P@3-VAE', 'POS_P@3-DiffRec',
            'POS_P@5-MF', 'POS_P@5-VAE', 'POS_P@5-DiffRec'
        ]
        pos_gini_cols = [col for col in selected_cols if col in df_item_filtered.columns]
        
        df_item_ranked = df_item_filtered.copy()
        for col in pos_gini_cols:
            df_item_ranked[col] = 1 - df_item_filtered[col]
        
        df_list_ranked = df_list_filtered.copy()
        for col in pos_gini_cols:
            df_list_ranked[col] = 1 - df_list_filtered[col]
        
        item_data = []
        for idx, row in df_item_ranked.iterrows():
            values = [float(row[col]) if pd.notna(row[col]) else 0 for col in pos_gini_cols]
            item_data.append((row['model'], values))
        
        list_data = []
        for idx, row in df_list_ranked.iterrows():
            values = [float(row[col]) if pd.notna(row[col]) else 0 for col in pos_gini_cols]
            list_data.append((row['model'], values))
        
        def shorten_label(label):
            label = label.replace('POS_P@', 'P@')
            return label
        
        labels = [shorten_label(col) for col in pos_gini_cols]
        
        colors = ['#FF0000','#FF9900' , '#00CC00', '#0000FF']
        markers = ['o', 's', '^', 'D']
        
        fig = plt.figure(figsize=(11, 6))
        
        N = len(pos_gini_cols)
        theta = radar_factory(N, frame='polygon')
        
        ax1 = fig.add_subplot(121, projection='radar')
        ax1.set_title('Item-level', weight='bold', size=16, pad=40, position=(0.5, 1.25))

        rgrid_values = [0.7, 0.75, 0.8, 0.85, 0.9]
        rgrid_labels = ['0.3', '0.25', '0.2', '0.15', '0.1']
        ax1.set_rgrids(rgrid_values, labels=rgrid_labels)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.set_ylim(0.7, 0.9)
        
        legend_handles = []
        legend_labels = []
        
        for idx, ((model, values), color) in enumerate(zip(item_data, colors)):
            marker = markers[idx % len(markers)]
            line, = ax1.plot(theta, values, marker=marker, linestyle='-', linewidth=2.5, 
                    markersize=4, color=color, label=model)
            ax1.fill(theta, values, facecolor=color, alpha=0.25)
            legend_handles.append(line)
            legend_labels.append(model)
        
        ax1.set_varlabels(labels)
        
        ax2 = fig.add_subplot(122, projection='radar')
        ax2.set_title('List-level', weight='bold', size=16, pad=40, position=(0.5, 1.25))

        rgrid_values_list = [0.7, 0.75, 0.8, 0.85, 0.9]
        rgrid_labels_list = ['0.3', '0.25', '0.2', '0.15', '0.1']
        ax2.set_rgrids(rgrid_values_list, labels=rgrid_labels_list)
        ax2.set_ylim(0.7, 0.9)
        ax2.tick_params(axis='y', labelsize=12)
        
        for idx, ((model, values), color) in enumerate(zip(list_data, colors)):
            marker = markers[idx % len(markers)]
            ax2.plot(theta, values, marker=marker, linestyle='-', linewidth=2.5,
                    markersize=4, color=color, label=model)
            ax2.fill(theta, values, facecolor=color, alpha=0.25)
        
        ax2.set_varlabels(labels)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.75, bottom=0.12)
        
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                bbox_to_anchor=(0.5, -0.08), ncol=min(len(legend_labels), 4), 
                fontsize=14, frameon=True)
        
        plt.show()

class GraphVisualizer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df_item, self.df_list = self.load_data()

    def load_data(self):
        exp_model_list = ['CF2', 'CF-GNNExplainer', 'C2Explainer', 'C2Explainer (-add)', 'UNRExplainer', 'GREASE', 'CLEAR']
        data_item = {
            "model": exp_model_list,
            "PN_S@3-LightGCN": [0] * len(exp_model_list),
            "PN_S@5-LightGCN": [0] * len(exp_model_list),
            "PN_S@3-GFormer": [0] * len(exp_model_list),
            "PN_S@5-GFormer": [0] * len(exp_model_list),
            "PN_S@3-SimGCL": [0] * len(exp_model_list),
            "PN_S@5-SimGCL": [0] * len(exp_model_list),
        }
        data_list = {
            "model": exp_model_list,
            "PN_R@3-LightGCN": [0] * len(exp_model_list),
            "PN_R@5-LightGCN": [0] * len(exp_model_list),
            "PN_R@3-GFormer": [0] * len(exp_model_list),
            "PN_R@5-GFormer": [0] * len(exp_model_list),
            "PN_R@3-SimGCL": [0] * len(exp_model_list),
            "PN_R@5-SimGCL": [0] * len(exp_model_list),
        }
        for rec_model in ['LightGCN', 'GFormer', 'SimGCL']:
            for i, exp_model in enumerate(exp_model_list):
                for top_k in [3, 5]:
                    with open(f"logs/{rec_model}_{exp_model}_{self.dataset_name}_top{top_k}_item_imp_agg.pkl", "rb") as f:
                        tmp_data = pkl.load(f)["overall"]
                        data_item[f"PN_S@{top_k}-{rec_model}"][i] = tmp_data['pn_s']
                    with open(f"logs/{rec_model}_{exp_model}_{self.dataset_name}_top{top_k}_list_imp_agg.pkl", "rb") as f:
                        tmp_data = pkl.load(f)["overall"]
                        data_list[f"PN_R@{top_k}-{rec_model}"][i] = tmp_data['pn_r']
                        
        df_item = pd.DataFrame(data_item)
        df_list = pd.DataFrame(data_list)
        return df_item, df_list

    def shorten_label(self, label):
        label = label.replace('GFormer', 'GF')
        label = label.replace('LightGCN', 'LGCN')
        label = label.replace('SimGCL', 'SGCL')
        label = label.replace('#Perturb', '#P')
        
        if label in ['PN-S@3-LGCN', 'PN-S@5-GF', 'PN-R@3-LGCN', 'PN-R@5-GF']:
            pass
        elif '-LGCN' in label:
            label = label.replace('-LGCN', '\nLGCN')
        elif '-GF' in label:
            label = label.replace('-GF', '\nGF')
        elif '-SGCL' in label:
            label = label.replace('-SGCL', '\nSGCL')
        return label
    
    def visualize(self):

        df_item_ranked = self.df_item
        df_list_ranked = self.df_list
        pn_cols = [col for col in df_item_ranked.columns if 'PN' in col]
        pn_cols_list = [col for col in df_list_ranked.columns if 'PN-R' in col]

        pn_labels = [self.shorten_label(col) for col in pn_cols]
        pn_labels_list = [self.shorten_label(col) for col in pn_cols_list]

        pn_data = []
        for idx, row in df_item_ranked.iterrows():
            values = []
            for col in pn_cols:
                val = row[col]
                if pd.isna(val) or val == '':
                    values.append(0)
                else:
                    values.append(float(val))
            pn_data.append((row['Model'], values))

        pn_data_list = []
        for idx, row in df_list_ranked.iterrows():
            values = []
            for col in pn_cols_list:
                val = row[col]
                if pd.isna(val) or val == '':
                    values.append(0)
                else:
                    values.append(float(val))
            pn_data_list.append((row['Model'], values))


        all_colors = [
            '#8c564b', 
            '#bcbd22',  
            '#17becf',  
            '#1f77b4',  
            '#ff7f0e',  
            '#2ca02c',  
            '#9467bd',  
        ]

        all_models = list(dict.fromkeys([model for model, _ in pn_data] + [model for model, _ in pn_data_list]))
        color_map = {model: all_colors[i % len(all_colors)] for i, model in enumerate(all_models)}
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X']
        marker_map = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

        fig = plt.figure(figsize=(11, 6))

        N_pn = len(pn_cols)
        theta_pn = radar_factory(N_pn, frame='polygon')
        ax1 = fig.add_subplot(121, projection='radar')
        ax1.set_title('Item-level', weight='bold', size=16, pad=40, position=(0.5, 1.25), ha='center', va='center')
        ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='y', labelsize=12)

        legend_handles = {}
        legend_labels = []

        for model, values in pn_data:
            color = color_map[model]
            marker = marker_map[model]
            line, = ax1.plot(theta_pn, values, marker=marker, linestyle='-', linewidth=2.5, 
                    markersize=4, color=color, label=model)
            ax1.fill(theta_pn, values, facecolor=color, alpha=0.25)
            if model not in legend_handles:
                legend_handles[model] = line

        ax1.set_varlabels(pn_labels)

        N_pn_list = len(pn_cols_list)
        theta_pn_list = radar_factory(N_pn_list, frame='polygon')
        ax2 = fig.add_subplot(122, projection='radar')
        ax2.set_title('List-level', weight='bold', size=16, pad=40, position=(0.5, 1.25), ha='center', va='center')
        ax2.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='y', labelsize=14)

        for model, values in pn_data_list:
            color = color_map[model]
            marker = marker_map[model]
            line, = ax2.plot(theta_pn_list, values, marker=marker, linestyle='-', linewidth=2.5,
                    markersize=4, color=color, label=model)
            ax2.fill(theta_pn_list, values, facecolor=color, alpha=0.25)
            if model not in legend_handles:
                legend_handles[model] = line

        ax2.set_varlabels(pn_labels_list)

        plt.tight_layout()
        plt.subplots_adjust(top=0.75, bottom=0.12)

        legend_items = [(model, legend_handles[model]) for model in all_models]
        fig.legend([handle for _, handle in legend_items], 
                [model for model, _ in legend_items], 
                loc='lower center', bbox_to_anchor=(0.5, -0.1), 
                ncol=min(len(legend_items), 4), fontsize=14, frameon=True)

        plt.show()

class UserVectorPositionVisualizer:
    def __init__(self):
        self.df = self.load_data()

    def load_data(self):
        output_data = {
            "dataset": [],
            "rec_model": [],
            "exp_model": [],
            "top_k": [],
            "POS-P@5": []
        }
        for dataset in ['Amazon', 'ML1M', 'Yahoo']:
            for rec_model in ['MF', 'VAE']:
                for exp_model in ['LIME-RS', 'SHAP', 'ACCENT', 'LXR']:
                    with open(f"logs/{rec_model}_{exp_model}_{dataset}_top5_item_imp_agg.pkl", "rb") as f:
                        tmp_data = pkl.load(f)["by_item"]
                        pos_p = tmp_data['pos_p']
                        for i in range(5):
                            output_data["dataset"].append(dataset)
                            output_data["rec_model"].append(rec_model)
                            output_data["exp_model"].append(exp_model)
                            output_data["top_k"].append(i + 1)
                            output_data["POS-P@5"].append(pos_p[i])

        return pd.DataFrame(output_data)

    def visualize(self):

        fig, axes = plt.subplots(2, 3, figsize=(7.5, 4))

        datasets = self.df['dataset'].unique()
        under_models = self.df['rec_model'].unique()
        models = self.df['exp_model'].unique()

        distinct_colors = [
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3', 
            '#ff7f00', 
            '#ffff33',  
            '#a65628',  
            '#f781bf', 
            '#999999',  
            '#00CED1',  
        ]
        colors = distinct_colors[:len(models)]
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X']

        for row_idx, under_model in enumerate(under_models):
            for col_idx, dataset in enumerate(datasets):
                ax = axes[row_idx, col_idx]
                
                df_subset = self.df[
                    (self.df['dataset'] == dataset) & 
                    (self.df['rec_model'] == under_model)
                ]
                
                if not df_subset.empty:
                    for idx, model in enumerate(models):
                        model_data = df_subset[df_subset['exp_model'] == model]
                        if not model_data.empty:
                            marker = markers[idx % len(markers)]
                            ax.plot(model_data['top_k'], model_data['POS-P@5'], 
                                    marker=marker, linestyle='-', linewidth=1.5, markersize=5,
                                    color=colors[idx], label=model)
                
                ax.set_title(f'{dataset} - {under_model}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Position', fontsize=11)
                ax.set_ylabel('POS-P@5', fontsize=11, fontweight='bold')
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_ylim(0, 0.8)
                ax.tick_params(axis='both', labelsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.17), 
                ncol=len(models), fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)

        plt.show()

class GraphPositionVisualizer:
    def __init__(self):
        self.df = self.load_data()

    def load_data(self):
        output_data = {
            "exp_model": [],
            "top_k": [],
            "PN-S@3": [],
            "PN-S@5": []
        }

        for exp_model in ['CF2', 'CF-GNNExplainer', 'C2Explainer', 'C2Explainer (-add)', 'UNRExplainer', 'GREASE', 'CLEAR']:
            with open(f"logs/LightGCN_{exp_model}_Amazon_top3_item_exp_agg.pkl", "rb") as f:
                tmp_data = pkl.load(f)["overall"]
                pn_s_3 = tmp_data['pn_s']
            with open(f"logs/LightGCN_{exp_model}_Amazon_top5_item_exp_agg.pkl", "rb") as f:
                tmp_data = pkl.load(f)["overall"]
                pn_s_5 = tmp_data['pn_s']
            for i in range(5):
                output_data["exp_model"].append(exp_model)
                output_data["top_k"].append(i + 1)
                output_data["PN-S@3"].append(pn_s_3[i] if i < 3 else np.nan)
                output_data["PN-S@5"].append(pn_s_5[i])
        
        return pd.DataFrame(output_data)
    
    def visualize(self):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.75))

        models = self.df['exp_model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X']

        df_plot_3 = self.df[self.df['PN-S@3'].notna()].copy()
        for idx, model in enumerate(models):
            model_data = df_plot_3[df_plot_3['exp_model'] == model]
            if not model_data.empty:
                marker = markers[idx % len(markers)]
                axes[0].plot(model_data['top_k'], model_data['PN-S@3'], 
                        marker=marker, linestyle='-', linewidth=1.5, markersize=3,
                        color=colors[idx], label=model)

        axes[0].set_xlabel('Position', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('PN-S@3', fontsize=14, fontweight='bold')
        axes[0].set_xticks([1, 2, 3])
        axes[0].tick_params(axis='both', labelsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        df_plot_5 = self.df[self.df['PN-S@5'].notna()].copy()
        for idx, model in enumerate(models):
            model_data = df_plot_5[df_plot_5['exp_model'] == model]
            if not model_data.empty:
                marker = markers[idx % len(markers)]
                axes[1].plot(model_data['top_k'], model_data['PN-S@5'], 
                        marker=marker, linestyle='-', linewidth=1.5, markersize=3,
                        color=colors[idx], label=model)

        axes[1].set_xlabel('Position', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('PN-S@5', fontsize=14, fontweight='bold')
        axes[1].set_xticks([1, 2, 3, 4, 5])
        axes[1].tick_params(axis='both', labelsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.show()

class DifferentGraphVisualizer:
    def __init__(self, dataset, level):
        self.level = level
        if self.level not in ['item', 'list']:
            raise ValueError("Unknown evaluation level: %s" % self.level)
        self.df = self.load_data(dataset)

        
    def load_data(self, dataset):
        output_data = {
            "model_name": [],
            "type": [],
            "PN-S@3": [],
            "PN-R@3": [],
            "#Perturb@3": [],
            "PN-S@5": [],
            "PN-R@5": [],
            "#Perturb@5": [],
        }
        for exp_model in ['CF2', 'CF-GNNExplainer', 'C2Explainer', 'C2Explainer (-add)', 'UNRExplainer', 'GREASE', 'CLEAR']:
            for type_graph in ['full', "khop", "indirect", "user"]:
                if type_graph == "khop": type_graph = ""
                with open(f"logs/LightGCN_{exp_model}_{dataset}_top3_{self.level}_exp_{type_graph}_agg.pkl", "rb") as f:
                    tmp_data = pkl.load(f)["overall"]
                    pn_s_3 = tmp_data['pn_s']
                    pn_r_3 = tmp_data['pn_r']
                    perturb_3 = tmp_data['#perturb']
                with open(f"logs/LightGCN_{exp_model}_{dataset}_top5_{self.level}_exp_{type_graph}_agg.pkl", "rb") as f:
                    tmp_data = pkl.load(f)["overall"]
                    pn_s_5 = tmp_data['pn_s']
                    pn_r_5 = tmp_data['pn_r']
                    perturb_5 = tmp_data['#perturb']

                output_data["model_name"].append(exp_model)
                output_data["type"].append(type_graph)
                output_data["PN-S@3"].append(pn_s_3)
                output_data["PN-R@3"].append(pn_r_3)
                output_data["#Perturb@3"].append(perturb_3)
                output_data["PN-S@5"].append(pn_s_5)
                output_data["PN-R@5"].append(pn_r_5)
                output_data["#Perturb@5"].append(perturb_5)

        return pd.DataFrame(output_data)
        

    def visualize(self):
        list_level = True if self.level == "list" else False

        types = ["Full", "K-hop", "Indirect", "User-only"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

        metric_col = 'PN-R@5' if list_level else 'PN-S@5'
        metric_label = "PN-R@5" if list_level else "PN-S@5"


        models = self.df['model_name'].unique()

        n_models = len(models)
        ncols = min(3, n_models)
        nrows = int(np.ceil(n_models / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2*nrows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes

        width = 0.1
        x_positions = np.arange(4) * width

        for idx, model_name in enumerate(models):
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            

            model_data = self.df[self.df['model_name'] == model_name]
            pn_metric = model_data[metric_col].values
            perturb = model_data['#Perturb@5'].values
            

            bars = ax1.bar(x_positions, pn_metric, width, color=colors, alpha=0.7)

            ax2.plot(x_positions, perturb, 'o-', color='#d62728', linewidth=2.5, markersize=8)
            
            ax2.set_ylim(0, max(perturb) * 1.15)
            

            ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            
            # Labels
            ax1.set_ylabel(f"{metric_label}", fontsize=11, fontweight='bold')
            ax2.set_ylabel("#Perturb@5", color='#d62728', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=11)
            ax1.tick_params(axis='both', labelsize=11)
            
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([])  
            ax1.set_xlabel(model_name, fontsize=11, fontweight='bold')
            
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=types[i]) for i in range(4)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='#d62728', linewidth=2.5, 
                                        markersize=8, label='#Perturb@5'))

        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                fontsize=11, bbox_to_anchor=(0.5, -0.05), frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()