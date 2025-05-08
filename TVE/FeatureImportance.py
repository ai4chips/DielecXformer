import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cmasher as cmr
import json
from contextlib import contextmanager
import sys
import os
import warnings
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import shap
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PermutationImportancePlotter:
    def __init__(self, model_dir, data_path, results_dir, config_file='PERMConfig.json', seed_value=1):
        self.model_dir = model_dir
        self.data_path = data_path
        self.results_dir = results_dir
        self.config_file = config_file
        self.seed_value = seed_value

        # Load config
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Load data
        self.df = pd.read_csv(data_path).iloc[:, :16]

        # Standardize data
        scaler = StandardScaler()
        self.df_scaled = scaler.fit_transform(self.df)

        # Separate features and labels
        self.data = pd.DataFrame(self.df_scaled[0:3000, 0:15])
        self.labels = np.ravel(self.df_scaled[0:3000, 15])
        self.X = self.data
        self.y = self.labels
        self.feature_name = [f'Modified FA{i + 1}' for i in range(15)]

        # StratifiedKFold setup
        self.num_bins = 10
        bins = np.linspace(self.labels.min(), self.labels.max(), self.num_bins)
        self.e_total_binned = np.digitize(self.labels, bins)
        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed_value)

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Define custom colormap for gradient background
        colors = ['white', 'white', 'white']
        Cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        base_cmap = Cmap
        colors = base_cmap(np.linspace(0, 1, 15))
        colors[:, -1] = 0.05  # Adjust alpha value to lighten the colors
        colors[:2, -1] = 0.05
        colors[8:, -1] = 1
        self.cm = LinearSegmentedColormap.from_list('custom_cmap', colors, N=15)

    def plot_permutation_importance(self, skip_to_fold=0):
        for i, (train_index, test_index) in enumerate(self.skf.split(self.labels, self.e_total_binned)):
            if i != skip_to_fold:
                continue
            else:
                X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]

                # Load the model for the current fold
                model_path = os.path.join(self.model_dir, f'model_fold_{i}.keras')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"File not found: {model_path}. Please ensure the file is an accessible `.keras` file.")

                model = load_model(model_path)

                @contextmanager
                def suppress_output():
                    with open(os.devnull, 'w') as devnull:
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = devnull
                        sys.stderr = devnull
                        try:
                            yield
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr

                # Calculate permutation importances using sklearn
                with suppress_output():
                    result = permutation_importance(model, X_test, y_test, n_repeats=50, random_state=self.seed_value,
                                                    scoring='neg_mean_squared_error')

                # Plotting setup
                fig, ax = plt.subplots(figsize=(4, 8))  # Adjust width and height to be vertical
                for spine in ax.spines.values():
                    spine.set_linewidth(2.5)
                perm_sorted_idx = result.importances_mean.argsort()[::-1][:5]  # Top 5
                max_y = np.max(result.importances_mean[perm_sorted_idx] + result.importances_std[perm_sorted_idx])

                # Plot the importance values
                for k, idx in enumerate(perm_sorted_idx):
                    x_position = k + 1  # Position along x-axis

                    # Create box plot
                    ax.boxplot(result.importances[idx],
                               vert=True,  # Vertical boxes
                               positions=[x_position],
                               widths=0.7,
                               patch_artist=True,
                               boxprops=dict(facecolor=self.cm(idx / 15), edgecolor='black', linewidth=1.2, alpha=0.7),
                               medianprops=dict(color='black', linewidth=2))

                    # Add scatter points for individual importance values
                    x_pos = x_position + np.linspace(-0.1, 0.1, num=result.importances[idx].shape[0])
                    ax.scatter(x_pos, result.importances[idx], color="white", alpha=1, zorder=1,
                               edgecolor="#6100FF", s=38, linewidths=2.2)

                # Set feature names and axis labels
                feature_names = [f'E.FA{i + 1}' for i in range(15)]
                sorted_feature_names = np.array(feature_names)[perm_sorted_idx]
                ax.set_xticks(range(1, len(sorted_feature_names) + 1))
                ax.set_xticklabels(sorted_feature_names, rotation=45, ha='right', size=20)

                ax.set_ylabel("Importance", labelpad=15, size=25)
                ax.set_title(f"Fold {i + 1}", pad=20, fontweight='bold')

                # Get y-limits from config
                ylim = self.config['folds'][str(i)]['ylim']
                ax.set_ylim(ylim)  # Set dynamic y-limit based on config

                # Add horizontal grid lines
                grid_y_positions = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                for ypos in grid_y_positions:
                    ax.axhline(y=ypos, color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)

                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f'perm_importances_plot_{i}.png'), dpi=600)
                # plt.show()

                # Optional: Save permutation importances as CSV
                perm_importances_df = pd.DataFrame(
                    {'feature': X_test.columns, 'importance_mean': result.importances_mean,
                     'importance_std': result.importances_std})
                perm_importances_file = os.path.join(self.results_dir, f'perm_importances_{i}_fold.csv')
                perm_importances_df.to_csv(perm_importances_file, index=False)



class SHAPFeatureImportance:
    def __init__(self, model_dir, data_file, shap_values_dir='./outputs/fig/SHAP', seed_value=1):
        """
        Initialize the SHAP feature importance calculation class.

        :param model_dir: Directory where the models are stored.
        :param data_file: Path to the input data file (CSV).
        :param shap_values_dir: Directory to save SHAP values.
        :param seed_value: Seed value for reproducibility.
        """
        self.model_dir = model_dir
        self.data_file = data_file
        self.shap_values_dir = shap_values_dir
        self.seed_value = seed_value
        self.df = None
        self.data = None
        self.labels = None
        self.X = None
        self.y = None
        self.model = None

        # Set the random seeds to ensure reproducibility
        np.random.seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        shap.initjs()

        # Disable SHAP and TensorFlow progress bars
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow progress bar

    def load_data(self):
        """Load and standardize the data."""
        # Load the dataset
        self.df = pd.read_csv(self.data_file).iloc[:, :16]
        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(self.df)
        self.data = pd.DataFrame(df_scaled[0:3000, 0:15])
        self.labels = np.ravel(df_scaled[0:3000, 15])
        self.X = self.data
        self.y = self.labels

    def stratified_k_fold(self, num_bins=10):
        """Perform stratified K-fold cross-validation and return train/test indices."""
        bins = np.linspace(self.labels.min(), self.labels.max(), num_bins)
        e_total_binned = np.digitize(self.labels, bins)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed_value)

        train_indices = []
        test_indices = []

        for train_index, test_index in skf.split(self.labels, e_total_binned):
            train_indices.append(train_index)
            test_indices.append(test_index)

        return train_indices, test_indices

    def load_model(self, fold_index):
        """Load the Keras model for a specified fold."""
        model_path = os.path.join(self.model_dir, f'model_fold_{fold_index}.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"File not found: {model_path}. Please ensure the file is an accessible `.keras` file.")
        self.model = load_model(model_path)

    def compute_shap_values(self, fold_index):
        """Compute SHAP values for a specified fold and save them."""
        train_indices, test_indices = self.stratified_k_fold()

        train_index = train_indices[fold_index]
        test_index = test_indices[fold_index]
        X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
        y_train, y_test = self.labels[train_index], self.labels[test_index]

        @contextmanager
        def suppress_output():
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        # Calculate permutation importances using sklearn
        with suppress_output():
            # Create SHAP explainer with a fixed random seed for reproducibility
            explainer = shap.KernelExplainer(self.model.predict,
                                             shap.sample(X_train, nsamples=100, random_state=self.seed_value), )

            # Compute SHAP values with a fixed seed for reproducibility
            shap_values = explainer.shap_values(X_test, progress_bar=False).squeeze()

        # Save SHAP values to CSV
        shap_values_file = os.path.join(self.shap_values_dir, f'shap_values_{fold_index + 1}_fold.csv')
        os.makedirs(self.shap_values_dir, exist_ok=True)
        shap_features_file = os.path.join(self.shap_values_dir, f'shap_features_{fold_index + 1}_fold.csv')
        os.makedirs(self.shap_values_dir, exist_ok=True)
        X_test = pd.DataFrame(X_test).to_csv(shap_features_file, index=False)
        shap_values_df = pd.DataFrame(shap_values)
        shap_values_df.to_csv(shap_values_file, index=False)

        print(f"    ✔ SHAP values saved to {shap_values_file}")

    def run(self, num_folds=10):
        """Run SHAP value computation for all folds."""
        for i in range(num_folds):
            print(f"{i + 1}. Processing fold {i + 1}...")
            self.load_data()
            self.load_model(i)
            self.compute_shap_values(i)

    def plot_shap_summary(self, fold_index, num_top_features=5, shap_values_dir=None):
        """
        Plot SHAP summary for the top features.

        :param fold_index: The fold index for which the SHAP values should be plotted.
        :param num_top_features: The number of top features to display based on SHAP values (default is 5).
        :param shap_values_dir: Directory where SHAP values are saved. If None, it defaults to the class's shap_values_dir.
        """
        # Default shap_values_dir if not provided
        shap_values_dir = shap_values_dir or self.shap_values_dir

        # Load the precomputed SHAP values for the specified fold
        shap_values_file = os.path.join(shap_values_dir, f'shap_values_{fold_index + 1}_fold.csv')
        if not os.path.exists(shap_values_file):
            raise FileNotFoundError(f"SHAP values file not found: {shap_values_file}")

        # Load the precomputed SHAP values for the specified fold
        shap_features_file = os.path.join(shap_values_dir, f'shap_features_{fold_index + 1}_fold.csv')
        if not os.path.exists(shap_features_file):
            raise FileNotFoundError(f"SHAP features file not found: {shap_features_file}")

        df_shap = pd.read_csv(shap_values_file)
        df_feature = pd.read_csv(shap_features_file)
        # Get the feature names from the data (assuming the data is already loaded and standardized)
        feature_names = df_shap.columns.astype(float)

        # Calculate the mean absolute SHAP values for each feature
        shap_values = df_shap.values
        importance = np.abs(shap_values).mean(axis=0)

        # Get the indices of the top `num_top_features` most important features
        top_indices = np.argsort(importance)[-num_top_features:]

        # Select the SHAP values for the top features
        top_shap_values = shap_values[:, top_indices]
        top_feature_names = feature_names[top_indices]

        # Modify feature names to add "E.FA" suffix, as in the second code
        modified_feature_names = [f"E.FA{int(feature) + 1}" for feature in top_feature_names]

        # Plot the SHAP summary for the top features
        shap.summary_plot(
            top_shap_values,
            df_feature.iloc[:, top_indices],  # Select the relevant columns of the data
            feature_names=modified_feature_names,  # Use modified feature names
            plot_type='dot',
            color_bar=False,
            show=False
        )

        # Rotate x and y labels for better visibility
        plt.xticks(rotation=270)
        plt.yticks(rotation=-30)

        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        # Get the current y-ticks (which correspond to the feature labels)
        current_yticks = ax.get_yticks()

        # The total number of labels
        n_labels = len(current_yticks)

        # The positions of the top, middle, and bottom labels
        # Move the top label upwards and keep the bottom one fixed, then distribute the middle ones evenly
        # We'll adjust by moving the top one upwards and the rest stay evenly spaced
        adjusted_yticks = np.linspace(current_yticks[0], current_yticks[-1] + 1, num=n_labels)


        # Set the new y-tick positions and labels
        ax.set_yticks(adjusted_yticks)
        ax.set_yticklabels(modified_feature_names)
        # Add black border around all edges of the plot
        for spine_name, spine in ax.spines.items():
            spine.set_edgecolor('black')  # Set the color to black
            spine.set_linewidth(2)  # Set the line width to 2

        bbox = ax.viewLim

        xmin, ymin = bbox.xmin, bbox.ymin
        xmax, ymax = bbox.xmax, bbox.ymax

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=6, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        plt.title(f'SHAP Values', fontsize=20, pad=10)
        plt.xlabel(' ', fontsize=0)

        # Add fold number on the right side, rotated 90 degrees
        plt.text(1.06, 0.5, f'Fold {fold_index + 1}', ha='left', va='center', fontsize=25, rotation=-90,
                 transform=plt.gca().transAxes, fontweight='bold')

        # Save the plot
        plot_filename = os.path.join(shap_values_dir, f'shap_summary_fold_{fold_index + 1}.png')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=600)
        # plt.show()
        print(f"SHAP summary plot saved to {plot_filename}")
