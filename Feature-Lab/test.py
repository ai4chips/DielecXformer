from FeatureProcessor import PearsonCorrelationCalculator

pca_input_file  = './reduction_outputs/pca_results.csv'
pca_output_file = './reduction_outputs/pca_InterCorrelation.csv'

pca_correlation_calculator = PearsonCorrelationCalculator(pca_input_file, pca_output_file)
pca_correlation_calculator.run()

fa_input_file  = './reduction_outputs/factor_results.csv'
fa_output_file = './reduction_outputs/factor_InterCorrelation.csv'

fa_correlation_calculator = PearsonCorrelationCalculator(fa_input_file, fa_output_file)
fa_correlation_calculator.run()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap
import json
from FeatureProcessor import InterCorrelationPlotter
# Example usage:
with open('analysis_visual_config.json') as config_file:
    config = json.load(config_file)

# Select the analysis type
analysis_type = 'factor_analysis'  # or 'pca_analysis'
plotter = InterCorrelationPlotter(config[analysis_type], "F")
plotter.plot("./reduction_outputs/factor_InterCorrelation.csv", "./figures/factor_InterCorrelation.svg")


# Select the analysis type
analysis_type = 'pca_analysis'
plotter = InterCorrelationPlotter(config[analysis_type], "P")
plotter.plot("./reduction_outputs/pca_InterCorrelation.csv", "./figures/pca_InterCorrelation.svg")

