import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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

        # Add 'formula_pretty' and 'Material ID' to the PCA results
        pca_results_with_metadata = pd.concat([self.pca_results, formula_pretty, material_id], axis=1)
        pca_results_with_metadata.columns = [*self.pca_results.columns, 'formula_pretty', 'Material ID']

        # Add 'formula_pretty' and 'Material ID' to the Factor Analysis results
        fa_results_with_metadata = pd.concat([self.fa_results, formula_pretty, material_id], axis=1)
        fa_results_with_metadata.columns = [*self.fa_results.columns, 'formula_pretty', 'Material ID']

        # Save the results
        self.standardized_data.to_csv(os.path.join(output_dir, "standardized_data.csv"), encoding='utf_8_sig',
                                      na_rep='None')
        pca_results_with_metadata.to_csv(os.path.join(output_dir, "pca_results.csv"), encoding='utf_8_sig',
                                         na_rep='None')
        self.pca_loadings.to_csv(os.path.join(output_dir, "pca_loadings.csv"), encoding='utf_8_sig', na_rep='None')
        fa_results_with_metadata.to_csv(os.path.join(output_dir, "factor_results.csv"), encoding='utf_8_sig',
                                        na_rep='None')
        self.fa_loadings.to_csv(os.path.join(output_dir, "factor_loadings.csv"), encoding='utf_8_sig', na_rep='None')

        print("✓ Results successfully saved to:")
        print(f"  - Standardized data: /{output_dir}/standardized_data.csv")
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
        - color_map (str or list): If a string (name of a colormap from plt.cm), it will use that colormap.
                                  If a list with 3 colors, it will create a custom color map.
                                  (default is None).
        - output_dir (str): Directory to save the plot (default is None).
        - n_components (int): Number of principal components or factors to extract (default is 15).
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
        print("[1/5] Loading data...")
        # Load the dataset
        data = pd.read_csv(self.csv_filepath, encoding='utf-8')

        # Transpose the data to match PCA/FA loading matrix format
        self.loading_matrix = np.abs(data.T)

        print("✓ Data loaded.")
        print("\n")

    def create_colormap(self):
        """
        Create a colormap from plt.cm or custom colors.
        """
        if isinstance(self.color_map, str):
            # Use colormap from plt.cm
            colormap = plt.cm.get_cmap(self.color_map)
        elif isinstance(self.color_map, list) and len(self.color_map) == 3:
            # Create custom colormap using the 3 colors
            colormap = LinearSegmentedColormap.from_list('custom_cmap', self.color_map)
        else:
            raise ValueError("color_map must be either a string (name of a colormap) or a list of 3 colors.")
        return colormap

    def plot_heatmap(self):
        """
        Plot the heatmap of the loading matrix.
        """
        if self.loading_matrix is None:
            raise ValueError("Data has not been loaded. Please run `load_data()` first.")

        # Create a custom or predefined colormap
        colormap = self.create_colormap()

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
        y_ticks = np.arange(0, len([f'PC{i}' for i in range(1, self.loading_matrix.shape[1] + 1)]), step=3)
        x_ticks = np.arange(0, len(self.loading_matrix.columns), step=10)

        plt.yticks(ticks=y_ticks, labels=[f'PC{i}' for i in range(1, self.loading_matrix.shape[1] + 1)][::3], fontweight='bold')
        plt.xticks(ticks=x_ticks, labels=self.loading_matrix.columns[::10], fontweight='bold')

        # Set font for the tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')

        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')

        plt.tight_layout()

        # Show the plot
        plt.show()

        # Optionally save the figure
        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            save_path = os.path.join(self.output_dir, 'PCA_FA_LoadingMatrix_Heatmap.svg')
            fig.savefig(save_path, format='svg', dpi=600)
            print(f"Figure saved at: {save_path}")


