import json
import os
import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


class DimensionalityReducer:
    """
    A utility class for applying PCA and Factor Analysis (FA) to a dataset after preprocessing.
    """

    def __init__(self, file_path, columns_to_drop, n_components=15):
        """
        Initialize the reducer with file path, columns to exclude, and number of components.
        """
        self.file_path = file_path
        self.columns_to_drop = columns_to_drop
        self.n_components = n_components

        self.raw_data = None
        self.filtered_data = None
        self.standardized_data = None

        self.pca_results = None
        self.pca_loadings = None
        self.fa_results = None
        self.fa_loadings = None

    def load_and_prepare_data(self):
        """
        Load the dataset and perform preprocessing.
        """
        print("[1/5] Loading and preprocessing data...")
        self.raw_data = pd.read_csv(self.file_path, encoding='gbk')
        data = self.raw_data.drop(columns=self.columns_to_drop)

        data = data.loc[:, data.std() != 0]
        data = data.dropna(axis=1).dropna(axis=0)

        self.filtered_data = data
        print("✓ Data loaded and preprocessed.")
        print("\n")

    def standardize(self):
        """
        Standardize the filtered data.
        """
        print("[2/5] Standardizing data...")
        scaled_array = scale(self.filtered_data)
        self.standardized_data = pd.DataFrame(scaled_array, columns=self.filtered_data.columns)
        print("✓ Data standardized.")
        print("\n")

    def apply_pca(self):
        """
        Apply Principal Component Analysis.
        """
        print(f"[3/5] Applying PCA (extracting {self.n_components} components)...")
        pca = PCA(n_components=self.n_components)
        results = pca.fit_transform(self.standardized_data)

        self.pca_results = pd.DataFrame(results, columns=[f"PC{i}" for i in range(1, self.n_components + 1)])
        self.pca_loadings = pd.DataFrame(
            pca.components_.T,  # Transpose the components matrix
            index=self.standardized_data.columns,
            columns=[f"PC{i}" for i in range(1, self.n_components + 1)]
        )

        print("✓ PCA completed.")
        print("\n")

    def apply_factor_analysis(self):
        """
        Apply Factor Analysis.
        """
        print(f"[4/5] Applying Factor Analysis (extracting {self.n_components} common factors)...")
        fa = FactorAnalyzer(n_factors=self.n_components, rotation="varimax", method="principal", use_smc=True)
        fa.fit(self.standardized_data)

        self.fa_results = pd.DataFrame(fa.transform(self.standardized_data),
                                       columns=[f"F{i}" for i in range(1, self.n_components + 1)])
        self.fa_loadings = pd.DataFrame(
            fa.loadings_,
            index=self.standardized_data.columns,
            columns=[f"F{i}" for i in range(1, self.n_components + 1)]
        )
        print("✓ Factor Analysis completed.")
        print("\n")

    def save_results(self):
        """
        Save the results to CSV files inside the `reduction_outputs` folder,
        including 'formula_pretty' and 'Material ID' columns in pca_results and fa_results.
        """
        # Create the folder if it does not exist
        output_dir = "reduction_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("[5/5] Saving results to CSV files...")

        # Extract the 'formula_pretty' and 'Material ID' columns
        formula_pretty = self.raw_data['formula_pretty']
        material_id = self.raw_data['material_id']
        e_total = self.raw_data['e_total']
        e_electronic = self.raw_data['e_electronic']
        e_ionic = self.raw_data['e_ionic']
        # Add 'formula_pretty' and 'Material ID' to the PCA results
        pca_results_with_metadata = pd.concat(
            [self.pca_results, e_total, e_ionic, e_electronic, formula_pretty, material_id], axis=1)
        pca_results_with_metadata.columns = [*self.pca_results.columns, 'e_total', 'e_ionic', 'e_electronic',
                                             'formula_pretty', 'Material ID']

        # Add 'formula_pretty' and 'Material ID' to the Factor Analysis results
        fa_results_with_metadata = pd.concat(
            [self.fa_results, e_total, e_ionic, e_electronic, formula_pretty, material_id], axis=1)
        fa_results_with_metadata.columns = [*self.fa_results.columns, 'e_total', 'e_ionic', 'e_electronic',
                                            'formula_pretty', 'Material ID']

        # Save the results
        pca_results_with_metadata.to_csv(os.path.join(output_dir, "pca_results.csv"), encoding='utf_8_sig',
                                         na_rep='None')
        self.pca_loadings.to_csv(os.path.join(output_dir, "pca_loadings.csv"), encoding='utf_8_sig', na_rep='None')
        fa_results_with_metadata.to_csv(os.path.join(output_dir, "factor_results.csv"), encoding='utf_8_sig',
                                        na_rep='None')
        self.fa_loadings.to_csv(os.path.join(output_dir, "factor_loadings.csv"), encoding='utf_8_sig', na_rep='None')

        print("✓ Results successfully saved to:")
        print(f"  - PCA    results   : /{output_dir}/pca_results.csv")
        print(f"  - PCA    loadings  : /{output_dir}/pca_loadings.csv")
        print(f"  - Factor results   : /{output_dir}/factor_results.csv")
        print(f"  - Factor loadings  : /{output_dir}/factor_loadings.csv")


class PCA_FA_LoadingMatrixHeatmap:
    """
    A class for plotting the heatmap of PCA/FA loading matrices from CSV files.
    """

    def __init__(self, csv_filepath, color_map=None, output_dir=None, n_components=15):
        """
        Initialize the class with a CSV filepath, color map, and output directory.

        Args:
        - csv_filepath (str): Path to the CSV file containing the data.
        - color_map (str or list): If a string (name of a colormap from plt), it will use that colormap.
                                  If a list with 3 colors, it will create a custom color map.
                                  (default is None).
        - output_dir (str): Directory to save the plot (default is None).
        - n_components (int): Number of principal components or factors to extract (default is 15).
        - label_format (str): Format for y-axis labels (default is "*{i}").
        """
        self.csv_filepath = csv_filepath
        self.color_map = color_map
        self.output_dir = output_dir
        self.n_components = n_components

        # Data will be loaded and preprocessed inside the class
        self.loading_matrix = None

    def load_data(self):
        """
        Load the dataset and extract the PCA/FA loading matrix.
        """
        print("[1/2] Loading data...")
        # Load the dataset
        data = pd.read_csv(self.csv_filepath, encoding='utf-8', index_col=0)

        # Transpose the data to match PCA/FA loading matrix format
        self.loading_matrix = np.abs(data.T)
        print("✓ Data loaded.")
        print("\n")
        return self.loading_matrix

    def plot_heatmap(self, method):
        """
        Plot the heatmap of the loading matrix.
        """
        if self.loading_matrix is None:
            raise ValueError("Data has not been loaded. Please run `load_data()` first.")

        print("[2/2] Generating the heatmap of the loading matrix...")
        print("\n")

        # Create a colormap from the color map input (or default to 'viridis' if None)
        if isinstance(self.color_map, str):
            colormap = plt.get_cmap(self.color_map)  # Corrected line to use plt.get_cmap()
        else:
            colormap = self.color_map  # If it's a list of colors, use it directly

        # Set plot parameters
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 14  # Global font size
        plt.rcParams['axes.titlesize'] = 18  # Subtitle font size
        plt.rcParams['axes.labelsize'] = 12  # Axis label font size
        plt.rcParams['xtick.labelsize'] = 19  # x-axis tick font size
        plt.rcParams['ytick.labelsize'] = 17  # y-axis tick font size
        plt.rcParams['legend.fontsize'] = 10  # Legend font size
        plt.rcParams['figure.titlesize'] = 20  # Figure title font size

        # Create figure and axis for the heatmap
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot heatmap
        sns.heatmap(self.loading_matrix, cmap=colormap, ax=ax, linewidths=0.6)

        # Set ticks for better visibility
        y_ticks = np.arange(0, len([f'{method.format(i=i)}' for i in range(1, self.loading_matrix.shape[0] + 1)]),
                            step=3)
        x_ticks = np.arange(0, len(self.loading_matrix.columns), step=10)

        # Format the labels based on the label_format
        y_labels = [method.format(i=i) for i in range(1, self.loading_matrix.shape[0] + 1)][::3]

        plt.yticks(ticks=y_ticks, labels=y_labels, fontweight='bold')
        plt.xticks(ticks=x_ticks, labels=self.loading_matrix.columns[::10], fontweight='bold')

        # Set font for the tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')

        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')

        plt.tight_layout()

        # Show the plot
        plt.show()
        print("\n")
        print("✓ Heatmap successfully generated.")
        # Optionally save the figure
        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            save_path = os.path.join(self.output_dir, f'{method}_LoadingMatrix_Heatmap.svg')
            fig.savefig(save_path, format='svg', dpi=600)
            print(f"✓ Figure saved at: {save_path}")


class PearsonCorrelationCalculator:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data_matrix = None
        self.correlation_matrix = None

    def load_data(self):
        data = pd.read_csv(self.input_file)
        self.data_matrix = data.T.iloc[1:-2, 0:7200]
        self.data_matrix = self.data_matrix.apply(pd.to_numeric, errors='coerce')
        print("✓ Data Matrix has Loaded.")

    def calculate_correlation(self):
        print("[1/3] Calculating Pearson correlation matrix...")

        # Calculate the Pearson correlation matrix
        self.correlation_matrix = np.corrcoef(self.data_matrix)
        self.correlation_matrix = pd.DataFrame(self.correlation_matrix)

        print("[2/3] Correlation matrix calculation completed.")

        # Return the correlation matrix for Jupyter display
        return self.correlation_matrix

    def save_correlation_matrix(self):
        self.correlation_matrix.to_csv(self.output_file, encoding='utf_8_sig')
        print(f"[3/3] Correlation matrix saved to {self.output_file}")

    def run(self):
        self.load_data()
        correlation_matrix = self.calculate_correlation()
        self.save_correlation_matrix()
        return correlation_matrix


def load_config(config_file):
    """Load the JSON configuration file."""
    with open(config_file, 'r') as file:
        return json.load(file)


class InterCorrelationPlotter:
    def __init__(self, config, method):
        self.config = config
        self.font_family = config['font']['family']
        self.font_size = config['font']['size']
        self.serif = config['font']['serif']
        self.title_size = config['title']['size']
        self.title_location = config['title']['location']
        self.title_pad = config['title']['pad']
        self.title_fontweight = config['title']['fontweight']
        self.cmap_name = config['colors']['cmap']
        self.colors = config['colors']['colors']
        self.color_bar_position = config['color_bar']['position']
        self.color_bar_size = config['color_bar']['size']
        self.color_bar_pad = config['color_bar']['pad']
        self.circle_size_factor = config['circle_size_factor']
        self.circle_color_scale = config['circle_color_scale']
        self.method = method

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 20
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['mathtext.default'] = 'regular'

        # Create colormap
        self.Cmap = getattr(cmr, self.cmap_name)
        self.Cmap1 = LinearSegmentedColormap.from_list('custom_cmap', self.colors)

    def plot(self, data_path, output_path):
        data_matrix = pd.read_csv(data_path).iloc[0:15, 1:16]
        data_matrix.index = data_matrix.columns
        data = np.array(data_matrix)
        size_data = np.shape(data)[0]

        # Get max value excluding diagonal elements
        max_value = np.max(np.triu(data, k=1))

        fig, ax = plt.subplots(figsize=(7, 7))

        # Draw circles in upper triangle of the matrix
        for i in range(len(data)):
            for j in range(i + 1, len(data[i])):
                value = data[i, j]
                size = self.circle_size_factor + 0.5 * value / max_value
                color = self.Cmap1(value / (max_value / self.circle_color_scale))
                square = Rectangle((j - size / 2, i - size / 2), size, size, fill=True, color=color, alpha=0.7,
                                   linewidth=2, edgecolor='white', zorder=2)
                ax.add_patch(square)

        # Set tick positions to be at the edges of the rectangles
        ax.set_xticks(np.arange(len(data[0])))
        ax.set_yticks(np.arange(len(data)))

        ax.set_xticklabels([fr'${self.method}_{{{i}}}$' for i in range(1, len(data[0]) + 1)],
                           size=20,
                           rotation=90,
                           ha='center',
                           va='bottom',
                           position=(0, -0.06))

        ax.set_yticklabels([fr'${self.method}_{{{i}}}$' for i in range(1, len(data) + 1)],
                           size=20,
                           ha='left',
                           va='center',
                           position=(-0.07, 0),
                           )

        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=8)

        # Draw gridlines
        for i in range(len(data)):
            for j in range(i + 1, len(data[i])):
                size = 1
                ax.plot([j - size / 2, j + size / 2], [i - size / 2, i - size / 2], color='silver', linestyle='-',
                        linewidth=1.5, zorder=4)
                ax.plot([j - size / 2, j + size / 2], [i + size / 2, i + size / 2], color='silver', linestyle='-',
                        linewidth=1.5, zorder=4)
                ax.plot([j - size / 2, j - size / 2], [i - size / 2, i + size / 2], color='silver', linestyle='-',
                        linewidth=1.5, zorder=4)
                ax.plot([j + size / 2, j + size / 2], [i - size / 2, i + size / 2], color='silver', linestyle='-',
                        linewidth=1.5, zorder=4)

        # Hide gridlines in the upper triangle
        for i in range(len(data)):
            for j in range(i, len(data[i])):
                ax.plot([j - size / 2, j + size / 2], [i + size / 2, i + size / 2], color='white', linestyle='-',
                        linewidth=1, zorder=3)
                ax.plot([j - size / 2, j + size / 2], [i - size / 2, i - size / 2], color='white', linestyle='-',
                        linewidth=1, zorder=3)
                ax.plot([j - size / 2, j - size / 2], [i - size / 2, i + size / 2], color='white', linestyle='-',
                        linewidth=1, zorder=3)
                ax.plot([j + size / 2, j + size / 2], [i - size / 2, i + size / 2], color='white', linestyle='-',
                        linewidth=1, zorder=3)

        ax.set_xlim(-0.5, len(data[0]) - 0.5)
        ax.set_ylim(-0.5, len(data) - 0.5)
        ax.set_aspect('equal', 'box')

        # Scatter plot of labels
        x = [(i + 0.3) for i in range(size_data - 1)]
        y = [(i + 0.8) for i in range(size_data - 1)]
        c = [f'C{i}' for i in range(1, len(data) + 1)]
        # Set title based on method (P or F)
        if self.method == "P":
            plt.title('Inter-PC Correlation Analysis', size=25, loc='left', pad=30, fontweight='bold', x=0.075)
        elif self.method == "F":
            plt.title('Inter-Factor Correlation Analysis', size=25, loc='left', pad=30, fontweight='bold', x=0.02)

        plt.scatter(x, y, zorder=7, color="dimgray", s=60)

        # Plot circles connecting the labels (Example for 'k_ioc', 'k_ele', and 'k_tol')
        self._plot_label_connections(x, y, size_data, data, max_value, 'ioc', 7, 0.5, 7, c, data_path)
        self._plot_label_connections(x, y, size_data, data, max_value, 'ele', 13.5, 7, 13.5, c, data_path)
        self._plot_label_connections(x, y, size_data, data, max_value, 'tol', 12, 2, 12, c, data_path)
        if self.method == "F":
            self._add_marker(0.5, 7, r'${\it ε}_{\mathrm{ioc}}$', '#F15800', 0.45, 7.5)
            self._add_marker(7, 13.5, r'${\it ε}_{\mathrm{ele}}$', '#F15800', 5.9, 13.3)
            self._add_marker(2, 12, r'${\it ε}_{\mathrm{tol}}$', '#F15800', 1.2, 12.3)
        else:
            self._add_marker(0.5, 7, r'${\it ε}_{\mathrm{ioc}}$', 'navy', 0.45, 7.5)
            self._add_marker(7, 13.5, r'${\it ε}_{\mathrm{ele}}$', 'navy', 5.9, 13.3)
            self._add_marker(2, 12, r'${\it ε}_{\mathrm{tol}}$', 'navy', 1.2, 12.3)

        # Create and add colorbar
        # Create a ScalarMappable for the first colormap
        norm = Normalize(vmin=-1, vmax=1)
        sm = ScalarMappable(cmap=self.Cmap1, norm=norm)
        sm.set_array([])

        # Add the first colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_ticklabels([])

        # Create a ScalarMappable for the RdGy_r colormap
        sm2 = ScalarMappable(cmap=self.Cmap, norm=norm)
        sm2.set_array([])

        # Add a small separation between the colorbars
        ax.set_xlim(-0.5, len(data[0]) - 0.5)
        ax.set_ylim(-0.5, len(data) - 0.5)

        # Add the second colorbar with hidden tick labels
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(sm2, cax=cax2)
        cbar2.ax.tick_params(labelsize=15)

        # Add a separate axis for the shared label between the two colorbars
        label_ax = divider.append_axes("right", size="2%", pad=0.1)
        label_ax.axis('off')  # Turn off the axis
        label_ax.text(-2.5, -0.06, 'Colorbar', ha='center', va='center', fontsize=18)
        # Show and save the plot
        fig.savefig(output_path, format='png', dpi=600)
        # plt.show()

    def _plot_label_connections(self, x, y, size_data, data, max_value, label, y_pos, x_start, y_start, c, data_path):
        label_data = pd.read_csv(data_path).iloc[15:16, 1:16]
        label_data = np.array(label_data)
        x_label, y_label = x_start, y_start
        for i in range(0, len(x)):
            value = label_data[0, i]
            size = 0.1 + 2.5 * value / max_value
            color = self.Cmap(value / (max_value / self.circle_color_scale))
            plt.plot([x_label, x[i]], [y_label, y[i]], color=color, linewidth=size, zorder=6)

    def _add_marker(self, x, y, label, color, text_position_x, text_position_y):
        """ Helper function to add scatter markers and labels """
        plt.scatter(x, y, zorder=7, color=color, s=150, marker="*")
        plt.text(text_position_x, text_position_y, label, ha='center', va='bottom', color='black', fontsize=25,
                 zorder=8, fontstyle='italic', fontweight='bold')
