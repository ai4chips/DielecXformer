import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, kendalltau
import dcor

class CorrelationAnalyzer:
    def __init__(self, descriptor_path_trans, descriptor_path_origin, loadings_path, target_column, output_path):
        self.descriptor_path_trans = descriptor_path_trans
        self.descriptor_path_origin = descriptor_path_origin
        self.loadings_path = loadings_path
        self.target_column = target_column
        self.output_path = output_path

    def load_data(self):
        self.df_trans = pd.read_csv(self.descriptor_path_trans, nrows=3000)
        self.df_origin = pd.read_csv(self.descriptor_path_origin, nrows=960)
        self.loadings = pd.read_csv(self.loadings_path, index_col=0)
        self.features = self.loadings.index

    def compute_transformed_scores(self):
        df_cols = self.df_trans.iloc[:, :15].T.values
        result = df_cols
        self.trans_scores = {}

        for feature in self.features:
            load_vector = self.loadings.loc[feature].values.reshape(1, -1)
            score = load_vector @ result
            self.trans_scores[feature] = score.flatten()

    def compute_correlation(self, x, y):
        return {
            'Pearson': pearsonr(x, y)[0],
            'Spearman': spearmanr(x, y)[0],
            'Kendall': kendalltau(x, y)[0],
            'DistanceCorr': dcor.distance_correlation(x, y)
        }

    def analyze_transformed(self):
        trans_results = []
        y = self.df_trans[self.target_column].values
        for feature in self.features:
            x = self.trans_scores[feature]
            corr = self.compute_correlation(x, y)
            trans_results.append({
                'Feature': feature,
                'trans_Pearson': corr['Pearson'],
                'trans_Spearman': corr['Spearman'],
                'trans_Kendall': corr['Kendall'],
                'trans_DistanceCorr': corr['DistanceCorr']
            })
        return pd.DataFrame(trans_results)

    def analyze_original(self):
        origin_results = []
        y = self.df_origin[self.target_column].values
        for feature in self.features:
            x = self.df_origin[feature].values
            corr = self.compute_correlation(x, y)
            origin_results.append({
                'Feature': feature,
                'origin_Pearson': corr['Pearson'],
                'origin_Spearman': corr['Spearman'],
                'origin_Kendall': corr['Kendall'],
                'origin_DistanceCorr': corr['DistanceCorr']
            })
        return pd.DataFrame(origin_results)

    def run_analysis(self):
        self.load_data()
        self.compute_transformed_scores()
        trans_df = self.analyze_transformed()
        origin_df = self.analyze_original()
        final_df = pd.merge(trans_df, origin_df, on='Feature')
        final_df.to_csv(self.output_path, index=False)
        print("Correlation analysis completed and saved.")
        return final_df

analyzer = CorrelationAnalyzer(
    descriptor_path_trans='./outputs/encoded_features/encodedOutputLayerExample.csv',
    descriptor_path_origin='./origin_data.csv',
    loadings_path='./../Feature-Lab/reduction_outputs/factor_loadings.csv',
    target_column='e_total',
    output_path='./outputs/correlation_results/correlation_results.csv'
)

final_results = analyzer.run_analysis()