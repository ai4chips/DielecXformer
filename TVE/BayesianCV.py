import os
import re
import random
import warnings
import json
import torch
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import os
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cmasher as cmr
import matplotlib.pyplot as plt
import os

# Configure math and text font settings
plt.rcParams['mathtext.fontset'] = 'stix'  # Math font style
plt.rcParams['font.family'] = 'serif'  # Text font family
plt.rcParams['font.serif'] = ['Times New Roman']  # Specific font
plt.rcParams['axes.titlesize'] = 30  # Subplot title font size
plt.rcParams['axes.labelsize'] = 28  # Axis label font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 25  # Legend font size


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BayesianCrossValidatedModel:
    def __init__(
            self,
            seed: int = 1,
            num_bins: int = 10,
            num_folds: int = 10,
            model_type: str = 'rf'
    ):
        self.seed = seed
        self.num_bins = num_bins
        self.num_folds = num_folds
        self.model_type = model_type  # 'rf', 'mlp', 'knn', or 'svr'
        self.results = []
        self._initialize_seed()

    def _initialize_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, **params) -> tuple[float, float]:
        bins = np.linspace(y.min(), y.max(), self.num_bins)
        y_binned = np.digitize(y, bins)
        skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed
        )
        r2_list, mse_list = [], []

        for train_idx, test_idx in skf.split(X, y_binned):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=int(params['n_estimators']),
                    max_depth=int(params['max_depth']),
                    min_samples_split=int(params['min_samples_split']),
                    min_samples_leaf=int(params['min_samples_leaf']),
                    random_state=self.seed
                )

            elif self.model_type == 'mlp':  # Now using sklearn's MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=(int(params['units']),) * int(params['layers']),
                    activation='relu' if params['activation'] == 'relu' else 'tanh',
                    max_iter=int(params['epochs']),
                    random_state=self.seed
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_list.append(r2_score(y_test, y_pred))
                mse_list.append(mean_squared_error(y_test, y_pred))
                continue

            elif self.model_type == 'knn':
                model = KNeighborsRegressor(
                    n_neighbors=int(params['n_neighbors']),
                    leaf_size=int(params['leaf_size']),
                    p=int(params['p'])
                )

            elif self.model_type == 'svm':
                model = SVR(
                    kernel='rbf',
                    C=float(params['C']),
                    gamma=float(params['gamma']),
                    epsilon=float(params.get('epsilon', 0.01))
                )

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_list.append(r2_score(y_test, y_pred))
            mse_list.append(mean_squared_error(y_test, y_pred))

        return float(np.mean(r2_list)), float(np.mean(mse_list))

    def optimize_model(self, X: pd.DataFrame, y: np.ndarray, file_path: str) -> None:
        if self.model_type == 'rf':
            param_bounds = {
                'n_estimators': (10, 100),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }

            def evaluate_rf(**params) -> float:
                r2, _ = self.cross_validate(X, y, **params)
                return r2

            optimizer = BayesianOptimization(
                f=evaluate_rf,
                pbounds=param_bounds,
                random_state=self.seed,
            )

        elif self.model_type == 'mlp':  # Adjust parameters for MLPRegressor
            param_bounds = {
                'layers': (1, 5),
                'units': (10, 100),
                'activation': (0, 1),  # 0: relu, 1: tanh
                'epochs': (30, 50)
            }

            def evaluate_mlp(layers, units, activation, epochs) -> float:
                # Map integer activation to the corresponding string value
                act_fn = 'relu' if int(activation) == 0 else 'tanh'
                r2, _ = self.cross_validate(
                    X,
                    y,
                    layers=layers,
                    units=units,
                    activation=act_fn,  # Use the mapped activation function string
                    epochs=epochs
                )
                return r2

            optimizer = BayesianOptimization(
                f=evaluate_mlp,
                pbounds=param_bounds,
                random_state=self.seed,
            )

        elif self.model_type == 'knn':
            param_bounds = {
                'n_neighbors': (1, 20),
                'leaf_size': (10, 50),
                'p': (1, 2)
            }

            def evaluate_knn(n_neighbors, leaf_size, p) -> float:
                r2, _ = self.cross_validate(
                    X,
                    y,
                    n_neighbors=n_neighbors,
                    leaf_size=leaf_size,
                    p=p
                )
                return r2

            optimizer = BayesianOptimization(
                f=evaluate_knn,
                pbounds=param_bounds,
                random_state=self.seed,
            )

        elif self.model_type == 'svm':
            param_bounds = {
                'C': (0.1, 5),
                'gamma': (0.2, 1.0)
            }

            def evaluate_svr(C, gamma) -> float:
                r2, _ = self.cross_validate(
                    X,
                    y,
                    C=C,
                    gamma=gamma,
                    epsilon=0.01
                )
                return r2

            optimizer = BayesianOptimization(
                f=evaluate_svr,
                pbounds=param_bounds,
                random_state=self.seed,
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.maximize(init_points=5, n_iter=25)

        for result in optimizer.res:
            params = result['params']
            if self.model_type == 'mlp':
                params['activation'] = 'relu' if int(params['activation']) == 0 else 'tanh'
            avg_r2, avg_mse = self.cross_validate(X, y, **params)
            self.results.append({
                'filename': os.path.basename(file_path),
                'params': params,
                'target': avg_r2,
                'mse': avg_mse
            })

    def process_csv_file(self, file_path: str) -> None:
        df = pd.read_csv(file_path).iloc[:3000, :16]
        scaled_data = StandardScaler().fit_transform(df)
        features = pd.DataFrame(scaled_data[:, :15])
        labels = np.ravel(scaled_data[:, 15])
        self.optimize_model(features, labels, file_path)

    @staticmethod
    def find_all_csv_files(directory: str) -> list[str]:
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if f.endswith('.csv')
        ]

    @staticmethod
    def extract_index_from_filename(file_path: str) -> int:
        base_name = os.path.basename(file_path)
        match = re.search(r'(\d+)(?=\.csv$)', base_name)
        if match:
            return int(match.group(1))
        raise ValueError(f"Filename {base_name} does not contain a valid index.")

    def run_on_folder(self, input_folder: str, output_file: str) -> None:
        file_list = self.find_all_csv_files(input_folder)
        file_list.sort(key=self.extract_index_from_filename)

        for file_path in file_list:
            self.process_csv_file(file_path)

            # Extract the filename from the file_path
            filename = os.path.basename(file_path)

            # Filter results for the current filename
            file_results = [result for result in self.results if result['filename'] == filename]
            if file_results:
                # Find the result with the maximum R² score for the current filename
                best_result = max(file_results, key=lambda x: x['target'])

                # Print the best result
                print(f"Filename: {best_result['filename']}")
                print("(1) Best parameter configuration obtained via Bayesian optimization:")
                for key, value in best_result['params'].items():
                    print(f"    • {key}: {value}")
                print(f"(2) Corresponding R² score: {best_result['target']:.3f}")
                print(f"(3) Corresponding MSE: {best_result['mse']:.3f}\n")
            else:
                print(f"No results found for {filename}\n")

        # Save all the results to the output file
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


class BayesianModelOptimizerCSVProcessor:
    def __init__(self, file_path, config_path='ComplexCalconfig.json'):
        # Initialize with the file path to process
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

        # Load the config file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Extract model name from the file name
        self.model_name = self.file_path.split('/')[-1].split('.')[0].split('_')[1].upper()

    def process(self):
        # Add the 'model' column with the extracted model name
        self.df['model'] = self.model_name

        # Extract the 'Layer' number from the filename and add it as 'TVE-layers' column
        self.df['TVE-layers'] = self.df['filename'].apply(lambda x: self.extract_layer_number(x))

        # Rename 'target' to '10-fold CV R²' and 'mse' to '10-fold CV MSE'
        self.df['10-fold CV R²'] = self.df['target']
        self.df['10-fold CV MSE'] = self.df['mse']

        # Drop the original 'target' and 'mse' columns
        self.df = self.df.drop(columns=['target', 'mse'])

        # Process the 'params' column, which contains dictionaries, and expand it into separate columns
        params_dict = self.df['params'].apply(lambda x: self.extract_params(x))
        params_df = pd.DataFrame(params_dict.tolist())

        # Round the numerical values to 4 significant digits using `apply`
        params_df = params_df.apply(lambda x: x.round(4) if x.dtype in ['float64', 'int64'] else x)

        # Concatenate the new parameter columns to the original DataFrame
        self.df = pd.concat([self.df, params_df], axis=1)

        # Calculate the model complexity
        self.df['complexity'] = self.df.apply(self.calculate_complexity, axis=1)

        # Move 'filename' and 'params' columns to the end of the DataFrame
        self.df = self.df[
            [col for col in self.df.columns if col not in ['filename', 'params']] + ['filename', 'params']]

        # Set the display format to center-align the DataFrame for visual output (useful for Jupyter Notebooks or IPython environments)
        pd.set_option('display.colheader_justify', 'center')

        return self.df

    def extract_layer_number(self, filename):
        # Use regex to extract the number after 'Layer' in the filename
        match = re.search(r'Layer(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def extract_params(self, params_str):
        # Replace single quotes with double quotes to ensure it's a valid JSON format
        params_str = params_str.replace("'", '"')

        try:
            params = json.loads(params_str)
        except Exception as e:
            print(f"Error parsing params: {e}")
            return {}

        return params

    def calculate_complexity(self, row):
        """
        Calculate the complexity of the model based on the model's parameters and boundaries
        specified in the config.json file.
        """
        model_params = self.config[self.model_name]

        # Get the parameter bounds and lists of relevant parameters
        param_bounds = model_params['boundaries']
        positive_params = model_params['positive_params']
        negative_params = model_params['negative_params']
        irrelevant_params = model_params['irrelevant_params']

        complexity_score = 0

        # For each parameter in the row, calculate its contribution to complexity
        for param, bounds in param_bounds.items():
            if param in irrelevant_params:
                continue

            param_value = row[param]

            if param in positive_params:
                # Complexity is proportional to how close the value is to the upper boundary
                upper_bound = bounds[1]
                complexity_score += (upper_bound - param_value) / (upper_bound - bounds[0])

            elif param in negative_params:
                # Complexity is proportional to how close the value is to the lower boundary
                lower_bound = bounds[0]
                complexity_score += (param_value - lower_bound) / (bounds[1] - lower_bound)

        # Normalize complexity to make sure the maximum complexity score is 1
        complexity_score = complexity_score / len(param_bounds)

        # Ensure complexity is capped at 1
        return min(complexity_score, 1)


class DielectricPredictor:
    def __init__(
            self,
            data_filepath: str,
            standardized_folder: str,
            hyperopt_results_csv: str,
            seed: int = 42,
            n_splits: int = 10,
            num_bins: int = 10,
            max_evals: int = 100,
    ):
        # Paths and configuration
        self.data_filepath = data_filepath
        self.standardized_folder = standardized_folder
        self.hyperopt_results_csv = hyperopt_results_csv
        os.makedirs(self.standardized_folder, exist_ok=True)

        # Random seeds
        self.seed = seed
        self.set_random_seeds(self.seed)

        # Cross-validation config
        self.n_splits = n_splits
        self.num_bins = num_bins

        # Hyperopt config
        self.max_evals = max_evals
        self.call_counter = 0

        # Placeholders
        self.data = None
        self.labels = None
        self.binned_labels = None
        self.skf = None

    @staticmethod
    def set_random_seeds(seed: int):
        np.random.seed(seed)  # Set NumPy random seed
        random.seed(seed)  # Set Python random seed
        tf.random.set_seed(seed)  # Set TensorFlow random seed

    @staticmethod
    def _residual_block(input_layer, units: int):
        x = Dense(units, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(units, activation=None)(x)
        x = Add()([x, input_layer])
        return Activation('relu')(x)

    def _build_model(self, input_shape: tuple, num_units: int, num_layers: int, dropout_rate: float) -> Model:
        inputs = Input(shape=input_shape)
        x = Dense(num_units, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        for _ in range(num_layers):
            x = self._residual_block(x, num_units)
            x = Dropout(dropout_rate)(x)

        x = Dense(num_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def load_and_preprocess(self, use_rows: slice = slice(0, 3000), feature_range: slice = slice(0, 15)):
        # Load
        df = pd.read_csv(self.data_filepath).iloc[:, :16]

        # Scale
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        # Separate
        self.data = pd.DataFrame(scaled[use_rows, feature_range])
        self.labels = scaled[use_rows, 15].ravel()

        # Stratify bins
        bins = np.linspace(self.labels.min(), self.labels.max(), self.num_bins)
        self.binned_labels = np.digitize(self.labels, bins)
        self.skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

    def _train_fold(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            params: dict,
            iteration_folder: str,
            fold_idx: int,
    ) -> dict:
        # Build and train
        model = self._build_model(
            input_shape=(X_train.shape[1],),
            num_units=int(params['num_units']),
            num_layers=int(params['num_layers']),
            dropout_rate=float(params['dropout_rate']),
        )
        model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)

        # Predict
        y_test_pred = model.predict(X_test, verbose=0).flatten()
        y_train_pred = model.predict(X_train, verbose=0).flatten()

        # Scores
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        # Save model
        save_dir = os.path.join(iteration_folder, 'save_models')
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f'model_fold_{fold_idx}.keras'))

        # Save predictions
        save_pred_dir = os.path.join(iteration_folder, 'predictions_file')
        os.makedirs(save_pred_dir, exist_ok=True)
        df_pred = pd.concat([
            pd.DataFrame({'y': y_train, 'y_pred': y_train_pred, 'Dataset': 'Train'}),
            pd.DataFrame({'y': y_test, 'y_pred': y_test_pred, 'Dataset': 'Test'}),
        ], ignore_index=True)
        df_pred.to_csv(os.path.join(save_pred_dir, f'predictions_fold_{fold_idx}.csv'), index=False)

        return {'r2_test': r2_test, 'mse_test': mse_test, 'r2_train': r2_train, 'mse_train': mse_train}

    def train_and_evaluate(self, params: dict) -> dict:
        # New iteration folder
        self.call_counter += 1
        iter_folder = os.path.join(self.standardized_folder, f'iteration_{self.call_counter}')
        os.makedirs(iter_folder, exist_ok=True)
        cross_val_results_folder = os.path.join(iter_folder, 'cross_validation_results')
        os.makedirs(cross_val_results_folder, exist_ok=True)

        # Print current hyperparameters
        print("=" * 50)
        print(f"Optimizing for ResNet model, trial {self.call_counter}")
        print("=" * 50)
        print(f"[1] Current hyperparameters: ")
        print(f"    ✦ {'num_units:':<20} {params['num_units']}")
        print(f"    ✦ {'num_layers:':<20} {params['num_layers']}")
        print(f"    ✦ {'dropout_rate:':<20} {params['dropout_rate']:.5f}")

        # Run CV
        r2_list, mse_list, results = [], [], []
        for fold_idx, (train_idx, test_idx) in enumerate(self.skf.split(self.data, self.binned_labels)):
            res = self._train_fold(
                self.data.iloc[train_idx],
                self.labels[train_idx],
                self.data.iloc[test_idx],
                self.labels[test_idx],
                params,
                iter_folder,
                fold_idx,
            )
            r2_list.append(res['r2_test'])
            mse_list.append(res['mse_test'])
            results.append({**res, 'fold': fold_idx})

        # Save cross-val results
        cv_df = pd.DataFrame(results)
        cv_df.to_csv(
            os.path.join(cross_val_results_folder, 'cv_results.csv'),
            index=False,
        )

        # Calculate averages
        avg_r2 = np.mean(r2_list)
        avg_mse = np.mean(mse_list)

        # Print performance
        print(f"[2] Performance for this trial:")
        print(f"    ✦ {'avg_r2:':<20} {avg_r2:.4f}")
        print(f"    ✦ {'avg_mse:':<20} {avg_mse:.4f}")
        print("\n")

        # Append Hyperopt results per iteration
        with open(os.path.join(cross_val_results_folder, 'hyperopt_results.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                int(params['num_units']),
                int(params['num_layers']),
                params['dropout_rate'],
                avg_r2,
                avg_mse,
            ])

        return {'loss': -avg_r2, 'status': STATUS_OK}

    def optimize(self, space: dict):
        # Set the random seed for Hyperopt
        self.set_random_seeds(self.seed)
        trials = Trials()
        best = fmin(
            fn=lambda p: self.train_and_evaluate(p),
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            verbose=False,
            rstate=np.random.default_rng(10)
        )
        return best


class ModelPlotter:
    def __init__(self, file_path_pattern, cv_file_path, output_folder):
        """
        :param file_path_pattern: Pattern to access prediction files (e.g., predictions_fold_{}.csv)
        :param cv_file_path: Path to the cross-validation results CSV file
        :param output_folder: Folder to save the generated plots
        """
        self.file_path_pattern = file_path_pattern
        self.cv_file_path = cv_file_path
        self.output_folder = output_folder
        self.cv_data = pd.read_csv(cv_file_path)

    def plot_model(self, file_number, ax):
        """
        Plot for a single model on the provided axis with scatter point colors based on distance to x=y line.
        """
        # Read the prediction data for each fold
        file_path = self.file_path_pattern.format(file_number)
        data = pd.read_csv(file_path)

        # Assuming `data` is your DataFrame
        train_true = []
        train_pred = []
        test_true = []
        test_pred = []

        for idx, row in data.iterrows():
            if row.iloc[2] == "Train":
                train_true.append(row.iloc[0])  # First column is train_true
                train_pred.append(row.iloc[1])  # Second column is train_pred
            elif row.iloc[2] == "Test":
                test_true.append(row.iloc[0])  # First column is test_true
                test_pred.append(row.iloc[1])  # Second column is test_pred

        # Retrieve corresponding cross-validation results
        cv_row = self.cv_data.iloc[file_number, :]

        # Extract values from the corresponding row
        test_r2 = cv_row["r2_test"]  # Test Set $R^2$
        train_r2 = cv_row["r2_train"]  # Train Set $R^2$
        test_mse = cv_row["mse_test"]  # Test Set MSE
        train_mse = cv_row["mse_train"]  # Train Set MSE

        # Calculate the distance from each point to the line x = y
        train_distances = np.abs(np.array(train_true) - np.array(train_pred))
        test_distances = np.abs(np.array(test_true) - np.array(test_pred))

        # Normalize the distances for colormap scaling
        all_distances = np.concatenate([train_distances, test_distances])
        norm = plt.Normalize(vmin=np.min(all_distances), vmax=np.max(all_distances) - 0.5)

        # Create the scatter plot
        scatter_train = ax.scatter(train_true, train_pred, color='#FF7800', label='Train data', s=90, marker='o',
                                   alpha=1,
                                   edgecolor="white", linewidth=2)
        scatter_test = ax.scatter(test_true, test_pred, c=test_distances, cmap=cmr.cosmic_r,
                                  s=90, marker='o', alpha=1, edgecolor="white", linewidth=2, norm=norm)

        # Add a colorbar to show the distance scale
        cbar = plt.colorbar(scatter_test, ax=ax, orientation='vertical')
        cbar.set_label(r'Distance', fontsize=28)

        # Dynamically generate title
        suffix = {1: r'$^{\text{st}}$', 2: r'$^{\text{nd}}$', 3: r'$^{\text{rd}}$'}.get(file_number + 1,
                                                                                        r'$^{\text{th}}$')

        ax.set_title(f'{file_number + 1}{suffix} ResNet Model', pad=30, fontweight='bold', loc='left')
        ax.set_xlabel(r'Cal.epsilon (Scaler}', fontweight='bold', labelpad=10)
        ax.set_ylabel(r'Pre.epsilon (Scaler)$', fontweight='bold', labelpad=10)

        # Plot the identity line x = y
        x = np.linspace(-4, 2, 400)
        ax.plot(x, x, color='black', linestyle=':', linewidth=2)

        # Set limits
        ax.set_xlim((-3.5, 2.1))
        ax.set_ylim((-3.5, 2.1))

        # Adjust tick parameters
        ax.tick_params(axis='both', labelsize=25)

        # Add legend
        ax.legend(prop={'weight': 'bold'})

        # Add text for R² values
        ax.text(0.04, 0.84, f'Test $R^2$  :{test_r2:.2f}', fontsize=28, color='black', ha='left', va='top',
                transform=ax.transAxes)
        ax.text(0.04, 0.95, f'Train $R^2$:{train_r2:.2f}', fontsize=28, color='black', ha='left', va='top',
                transform=ax.transAxes)

        # Layout adjustment
        plt.tight_layout()
        plt.close()

    def plot_all_models(self, pad=0):
        """
        Plot all models in a grid: 3x3 grid for the first 9 models,
        with the 10th plot in the center of the 4th row (middle of the last row).
        `pad` controls the spacing between subplots.
        """
        # Create individual figures and axes first
        models = []
        for i in range(10):
            fig, ax = plt.subplots(figsize=(8, 7))  # Create an individual figure for each model
            self.plot_model(i, ax)  # Plot model to the individual axis
            # Save the figure to a temporary file
            temp_filename = os.path.join(self.output_folder, f'fold-{i + 1}_scatter.png')
            fig.savefig(temp_filename, format='png', dpi=600)
            models.append(temp_filename)  # Save the temporary filename for later

        # Now, create the combined 4x3 grid for displaying the plots
        fig, axs = plt.subplots(4, 3, figsize=(21, 28))
        axs = axs.flatten()

        # Add the first 9 models to the 3x3 grid
        for i in range(9):
            img = plt.imread(models[i])  # Read the saved image file
            axs[i].imshow(img)  # Display the image
            axs[i].axis('off')  # Hide the axis for cleaner display

        # Hide the 4th row, 1st and 3rd columns (no image)
        axs[9].axis('off')  # Hide empty subplot in the 4th row, 1st column
        axs[11].axis('off')  # Hide empty subplot in the 4th row, 3rd column

        # Place the 10th model in the center of the 4th row (2nd column)
        img = plt.imread(models[9])  # Read the saved image file
        axs[10].imshow(img)  # Display the image in the center of the last row
        axs[10].axis('off')  # Hide axis for cleaner display

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=pad, hspace=pad)  # You can adjust these values for smaller spacing

        # Show the combined grid plot
        plt.tight_layout()
        tenmodels_filename = os.path.join(self.output_folder, f'all_scatter.png')
        plt.savefig(tenmodels_filename, format='png', dpi=1000)
        # plt.show()
