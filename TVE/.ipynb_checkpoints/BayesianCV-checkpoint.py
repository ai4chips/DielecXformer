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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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




import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import csv

# Set global random seed for reproducibility
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

import random
import numpy as np
import os
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tensorflow.keras.models import save_model

class DielectricPredictor:
    def __init__(
        self,
        data_filepath: str,
        standardized_folder: str,
        hyperopt_results_csv: str,
        seed: int = 1,
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
        y_pred = model.predict(X_test, verbose=0).flatten()
        y_train_pred = model.predict(X_train, verbose=0).flatten()

        # Scores
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Save model
        save_dir = os.path.join(iteration_folder, 'save_models')
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f'model_fold_{fold_idx}.keras'))

        # Save predictions
        save_pred_dir = os.path.join(iteration_folder, 'predictions_file')
        os.makedirs(save_pred_dir, exist_ok=True)
        df_pred = pd.concat([
            pd.DataFrame({'y': y_train, 'y_pred': y_train_pred, 'Dataset': 'Train'}),
            pd.DataFrame({'y': y_test, 'y_pred': y_pred, 'Dataset': 'Test'}),
        ], ignore_index=True)
        df_pred.to_csv(os.path.join(save_pred_dir, f'predictions_fold_{fold_idx}.csv'), index=False)

        return {'r2': r2, 'mse': mse}

    def train_and_evaluate(self, params: dict) -> dict:
        # New iteration folder
        self.call_counter += 1
        iter_folder = os.path.join(self.standardized_folder, f'iteration_{self.call_counter}')
        os.makedirs(iter_folder, exist_ok=True)
        os.makedirs(os.path.join(iter_folder, 'cross_validation_results'), exist_ok=True)

        # Print current hyperparameters
        print("=" * 50)  # 打印分隔符，创建醒目的标题区分
        print(f"Optimizing for ResNet model, trial {self.call_counter} ")
        print("=" * 50)  # 再次打印分隔符
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
            r2_list.append(res['r2'])
            mse_list.append(res['mse'])
            results.append({**res, 'fold': fold_idx})

        # Save cross-val results
        cv_df = pd.DataFrame(results)
        cv_df.to_csv(
            os.path.join(iter_folder, 'cross_validation_results', 'cv_results.csv'),
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
        with open(self.hyperopt_results_csv, 'a', newline='') as f:
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
            rstate=np.random.default_rng(1)
        )
        return best




if __name__ == '__main__':
    # Example usage
    predictor = DielectricPredictor(
        data_filepath='/Users/guozikang/.../layer_8_output.csv',
        standardized_folder='./standardized_results',
        hyperopt_results_csv='./hyperopt_results.csv',
        seed=1,
        n_splits=10,
        num_bins=10,
        max_evals=100,
    )
    predictor.load_and_preprocess()
    # Initialize hyperopt CSV
    with open(predictor.hyperopt_results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_units', 'num_layers', 'dropout_rate', 'avg_r2', 'avg_mse'])
    space = {
        'num_units': hp.quniform('num_units', 16, 256, 1),
        'num_layers': hp.quniform('num_layers', 3, 10, 1),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
    }
    predictor.optimize(space)






