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
# model_svm = BayesianCrossValidatedModel(model_type="svm")
# model_svm.run_on_folder(folder_path, 'checkpoints/BayesianResult/bayesian_svm.csv')


from BayesianCV import BayesianModelOptimizerCSVProcessor
import pandas as pd

# Example usage
if __name__ == "__main__":
    # Assuming the file path is 'bayesian_knn.csv'
    file_path = './checkpoints/BayesianResult/bayesian_knn.csv'
    processor = BayesianModelOptimizerCSVProcessor(file_path)
    processed_df = processor.process()

    # Set Pandas display options to show more rows and columns
    pd.set_option('display.max_rows', None)  # None means no limit on rows
    pd.set_option('display.max_columns', None)  # None means no limit on columns
    pd.set_option('display.width', None)  # Auto-detect the display width
    pd.set_option('display.max_colwidth', None)  # No limit on column width

    # Output the processed DataFrame
    print(processed_df)
