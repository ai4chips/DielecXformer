# from TransformerBasedVariantEncoder import TransformerBasedVariantEncoderRunner, setSeed
# import pandas as pd
#
# if __name__ == "__main__":
#     # Example execution
#     setSeed(1)
#     dataPath = 'CrystalFeatureMatrix.csv'
#     modelDir = 'checkpoints/transformer_models'
#     outputDir = 'outputs/encoded_features'
#
#     df = pd.read_csv(dataPath, dtype=str)
#     dimModel = df.loc[:, df.columns[0]:'FA15'].shape[1]
#
#     runner = TransformerBasedVariantEncoderRunner(
#         dataPath=dataPath,
#         modelDir=modelDir,
#         outputDir=outputDir,
#         dimModel=dimModel,
#         numHeads=5
#     )
#     runner.run(maxLayers=8)

# import warnings
# import re
#
# # 全局忽略匹配特定内容的 UserWarning（例如 Sklearn Split 警告）
# warnings.simplefilter("ignore", UserWarning)
#
# from BayesianCV import BayesianCrossValidatedModel, set_seed
#
# set_seed(42)
#
# folder_path = 'outputs/encoded_features'
#
#
#
# model_svm = BayesianCrossValidatedModel(model_type="knn")
# model_svm.run_on_folder(folder_path, './checkpoints/BayesianResult/bayesian_knn.csv')
#
#
# from BayesianCV import BayesianModelOptimizerCSVProcessor
# import pandas as pd
#
# # Example usage
# if __name__ == "__main__":
#     # Assuming the file path is 'bayesian_knn.csv'
#     file_path = './checkpoints/BayesianResult/bayesian_knn.csv'
#     processor = BayesianModelOptimizerCSVProcessor(file_path)
#
#     # Process the file and compute the complexity
#     processed_df = processor.process()
#
#     # Set Pandas display options to show more rows and columns
#     pd.set_option('display.max_rows', None)  # None means no limit on rows
#     pd.set_option('display.max_columns', None)  # None means no limit on columns
#     pd.set_option('display.width', None)  # Auto-detect the display width
#     pd.set_option('display.max_colwidth', None)  # No limit on column width
#
#     # Output the processed DataFrame
#     print(processed_df)


#
# import os
# import csv
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from BayesianCV import DielectricPredictor
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# import tensorflow as tf
# import random
#
# def set_tf_seed(seed: int):
#     tf.random.set_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#
# if __name__ == '__main__':
#     # Example usage
#     set_tf_seed(1)
#     predictor = DielectricPredictor(
#         data_filepath='./outputs/encoded_features/encodedOutputLayer8.csv',
#         standardized_folder='./checkpoints/BayesianResult/ResultResNet/',
#         hyperopt_results_csv='./hyperopt_results.csv',
#         seed=1,
#         n_splits=10,
#         num_bins=10,
#         max_evals=100,
#     )
#     predictor.load_and_preprocess()
#     # Initialize hyperopt CSV
#     with open(predictor.hyperopt_results_csv, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['num_units', 'num_layers', 'dropout_rate', 'avg_r2', 'avg_mse'])
#     space = {
#         'num_units': hp.quniform('num_units', 16, 256, 1),
#         'num_layers': hp.quniform('num_layers', 3, 10, 1),
#         'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
#     }
#     predictor.optimize(space)


# from BayesianCV import ModelPlotter
# # Example usage
# file_path_pattern = './checkpoints/BayesianResult/ResultResNet/iteration_1/predictions_file/predictions_fold_{}.csv'
# cv_file_path = './checkpoints/BayesianResult/ResultResNet/iteration_1/cross_validation_results/cv_results.csv'
# output_folder = './outputs/fig/scatter_ResNet/'
#
# plotter = ModelPlotter(file_path_pattern, cv_file_path, output_folder)
# plotter.plot_all_models()




# from FeatureImportance import PermutationImportancePlotter
# import matplotlib.pyplot as plt
# plt.rcParams['font.family']      = 'serif'
# plt.rcParams['font.serif']       = ['Times New Roman']
# plt.rcParams['font.size']        = 14
# plt.rcParams['axes.titlesize']   = 25
# plt.rcParams['axes.labelsize']   = 12
# plt.rcParams['xtick.labelsize']  = 15
# plt.rcParams['ytick.labelsize']  = 20
# plt.rcParams['legend.fontsize']  = 20
# plt.rcParams['figure.titlesize'] = 18
#
#
# model_dir = './checkpoints/BayesianResult/ResultResNet/iteration_example/save_models'
# data_path = './outputs/encoded_features/encodedOutputLayerExample.csv'
# results_dir = './outputs/fig/PERM'
#
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
#
# import sys
# from PIL import Image
#
# # Initialize your plotter
# plotter = PermutationImportancePlotter(model_dir, data_path, results_dir, config_file='PERMConfig.json')
#
# # Total number of folds
# total_folds = 10
# progress_bar_length = 50  # Adjust the length of the progress bar
#
# # Loop through folds 0 to 9 and plot each
# for fold in range(total_folds):
#     # Calculate the progress ratio
#     progress = (fold + 1) / total_folds
#     block_count = int(progress * progress_bar_length)
#
#     # Create a progress bar
#     progress_bar = f"[{'#' * block_count}{' ' * (progress_bar_length - block_count)}] {fold + 1}/{total_folds} completed"
#     sys.stdout.write(f"\r{progress_bar}")
#     sys.stdout.flush()
#
#     # Plot the permutation importance for this fold
#     plotter.plot_permutation_importance(skip_to_fold=fold)  # This already saves the image as a PNG
#
# from PIL import Image
#
# # Read the saved images
# images = [Image.open(f"{results_dir}/perm_importances_plot_{i}.png") for i in range(total_folds)]
#
# # Calculate the width and height of the combined image
# max_width = max(image.width for image in images)
# max_height = max(image.height for image in images)
#
# # Calculate combined image dimensions: 5 images per row, 2 rows
# combined_width = max_width * 5  # 5 images per row
# combined_height = max_height * 2  # 2 rows
#
# # Create a blank image to combine all subplots
# combined_image = Image.new('RGB', (combined_width, combined_height))
#
# # Paste each image into the final combined image
# x_offset = 0
# y_offset = 0
# for i, image in enumerate(images):
#     combined_image.paste(image, (x_offset, y_offset))
#     if (i + 1) % 5 == 0:  # After every 5 images, move to the next row
#         y_offset += image.height
#         x_offset = 0
#     else:
#         x_offset += image.width
#
# # Save the final combined image with high quality
# combined_image.save(f"{results_dir}/permutation_importance_0_to_9.png", quality=100)
#
# # Show the final combined image
# combined_image.show()
#
# # Final output after completion
# print("\nAll folds completed!")

# from FeatureImportance import SHAPFeatureImportance
#
# # Set paths
# model_dir = './checkpoints/BayesianResult/ResultResNet/iteration_example/save_models'
# data_file = "./outputs/encoded_features/encodedOutputLayerExample.csv"
#
# # Create class instance and run the process
# shap_calc = SHAPFeatureImportance(model_dir, data_file, seed_value=1)
# shap_calc.run()


from FeatureImportance import SHAPFeatureImportance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

def rotate_and_arrange_plots(shap_calc, shap_values_dir, num_top_features=5):
    # Create a list to store the file paths of individual SHAP plots
    plot_files = []

    # Generate SHAP plots for the first 10 features
    for i in range(10):
        shap_calc.plot_shap_summary(i, num_top_features=num_top_features, shap_values_dir=shap_values_dir)
        plot_files.append(f'{shap_values_dir}/shap_summary_fold_{i + 1}.png')
        plt.close()

    # Set up the figure for 2 rows and 5 columns of subplots
    fig, axes = plt.subplots(2, 5, figsize=(6, 5))

    for i, ax in enumerate(axes.flat):
        # Load the generated plot image
        img = mpimg.imread(plot_files[i])

        # Convert to PIL Image to trim the white space
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit format for trimming
        pil_img = pil_img.convert("RGB")

        # Trim white borders using the crop method
        bbox = pil_img.getbbox()  # Get bounding box of non-white regions
        trimmed_img = pil_img.crop(bbox)

        # Rotate the image 90 degrees counterclockwise
        rotated_img = np.array(trimmed_img.rotate(90, expand=True))

        # Display the rotated image
        ax.imshow(rotated_img)
        ax.axis('off')  # Turn off axis for better visual appearance

    # Adjust layout for minimal spacing and no overlap
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

# Set the paths for your model directory and data file
model_dir = './checkpoints/BayesianResult/ResultResNet/iteration_example/save_models'
data_file = "./outputs/encoded_features/encodedOutputLayerExample.csv"

# Create the SHAPFeatureImportance object
shap_calc = SHAPFeatureImportance(model_dir, data_file, seed_value=1)

# Call the function to rotate and arrange the plots
rotate_and_arrange_plots(shap_calc, shap_values_dir='./outputs/fig/SHAP')








