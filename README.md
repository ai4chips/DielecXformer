![DielecXformer Framework](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/DielecXformerFramework.png)

# **DielecXformer**

---

This package provides a general framework for crystal property prediction. **DielecXformer** (Transformer for dielectric) demonstrates excellent predictive performance for dielectric constant targets in crystals. It is a Transformer variant specifically designed for small- to medium-sized tabular datasets, aiming to overcome the performance bottlenecks of traditional deep learning architectures on such data. This is particularly significant given that 76% of datasets on the popular benchmark site [openml.org](https://www.openml.org) contain fewer than 10,000 rows.

DielecXformer leverages the Tree-structured Parzen Estimator (TPE) algorithm and integrates multiple prediction heads (MLP, KNN, ResNet, SVM, and RF) to perform hyperparameter blending optimization for TVE and its downstream models. Through both linear/nonlinear correlation analysis and feature importance evaluation, it provides deep insights into the mechanisms and optimization effects of TVE.
## **Getting Started**

---

### Installation

Set up the conda environment and clone the GitHub repository. You can find the corresponding `installation.md` and `requirements.md` files at:

- [installation.md](https://github.com/ai4chips/DielecXformer/blob/main/installation.md)
- [requirements.md](https://github.com/ai4chips/DielecXformer/blob/main/requirements.md)

```bash
# create a new environment
$ conda create --name DieleXformer python=3.11
$ conda activate DieleXformer

# install requirements
$ pip install IPython==8.30.0
$ pip install cmasher==1.7.2
$ pip install dcor==0.6
$ pip install factor_analyzer==0.5.0
$ pip install hyperopt==0.2.7
$ pip install imageio==2.37.0
$ pip install ipywidgets==8.1.5
$ pip install keras==3.6.0
$ pip install matminer==0.9.2
$ pip install matplotlib==3.8.0
$ pip install moviepy==2.1.2
$ pip install numpy==1.26.4
$ pip install pandas==2.2.3
$ pip install pymatgen==2024.4.13
$ pip install scipy==1.11.4
$ pip install seaborn==0.13.2
$ pip install shap==0.45.1
$ pip install tensorflow==2.18.0
$ pip install torch==2.1.1

# clone the source code of DieleXformer
$ git clone https://github.com/ai4chips/DielecXformer.git
$ cd DieleXformer
```


### Dataset

The dataset used for Transformer-based Variant Encode feature learning is derived from the `Materials Project` database. Using its provided API key, both structural files (e.g., `POSCAR`) and material-related property calculation results (such as `Formation Energy`, `Density`, `Volume`, etc.) can be obtained. The method for calculating the `dielectric constant` is detailed in the following references:

1. [I. Petousis, et al. High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials. *Sci. Data*. 4, 1-12 (2017).](https://doi.org/10.1038/sdata.2016.134)  
2. [I. Petousis, et al. Benchmarking density functional perturbation theory to enable high-throughput screening of materials for dielectric constant and refractive index. *Physical Review B*. 93, 115151 (2016).](https://doi.org/10.1103/PhysRevB.93.115151)

Using the crystal structure `POSCAR` file (which includes information about the `elemental composition` of the material), 283-dimensional descriptors related to material structure and composition were computed using `Magpie`, for further calculations.

**NOTE:** Although **Magpie** has been integrated into the `matminer` library, inspection of the `matminer.utils.data.MagpieData` class reveals that it only provides two functions:

- `get_elemental_property(elem, property_name)`
- `get_oxidation_states(elem)`

Clearly, it does not contain any information related to **material structure**.

In the original Magpie Java project’s API documentation ([https://wolverton.bitbucket.io/javadoc/index.html](https://wolverton.bitbucket.io/javadoc/index.html)), the following two packages are provided:

- `magpie.attributes.generators.composition`  
  *Attributes based on the composition of a material.*
- `magpie.attributes.generators.crystal`  
  *Tools to generate attributes based on crystal structures.*

Please note that the original Java version of Magpie is **no longer maintained**. Only the **composition-based descriptor** functionalities have been incorporated into the `matminer` library.


### Tokenization

You can use `./PI-Encoding/MagpiePIEncoding.py` to encode materials. For detailed procedures, refer to the `EncodingExample.ipynb` file in the same directory.

---

## **Run the Model**

### 1. Physical information embedding
The table `PIEmb.csv`, generated using the original Java version of Magpie, can be found in the `./Feature-Lab` folder. Alternatively, you can regenerate similar structured data using the `matminer` library.

- **SCFeatureExtractor**
```python
import pandas as pd
from MgapiePIEncoding import SCFeatureExtractor

if __name__ == "__main__":
    extractor = SCFeatureExtractor(
        folder="POSCAR_batch",                # Folder containing .vasp or POSCAR files
        output_file="tabular_data.csv"        # Output CSV file
    )
    extractor.extract_features()
```
- **MPPropertiesExtractor**
```python
from MgapiePIEncoding import MPPropertiesExtractor

if __name__ == "__main__":
    # Initialize the MPPropertiesExtractor with the desired parameters
    extractor = MPPropertiesExtractor(
        folder="POSCAR_batch",                      # Folder where structure files (POSCAR, .vasp) are located
        output_file="mp_properties.csv",            # Output CSV file to save the retrieved properties
        api_key="5vDJu0MvvXZtFQ4T3a0d8AU7ZQzh4aFD"  # Materials Project API key
    )

    # Extract the properties from the structure files
    extractor.extract_properties()
```


### 2. Feature Engineering

```python
from FeatureProcessor import DimensionalityReducer
import os

# Columns to exclude from analysis
columns_to_exclude = [
    # Metadata and identifiers
    "formula_pretty", "Class", "material_id", "n",
    
    # Density of states
    "dos_energy_up", "dos_energy_down",
    
    # Elastic moduli
    "k_voigt", "k_reuss", "k_vrh",
    "g_voigt", "g_reuss", "g_vrh",
    
    # Electronic structure
    "cbm", "vbm", "efermi", "e_total", "energy_above_hull",
    
    # Magnetic and ionic properties
    "min_NfUnfilled", "min_GSmagmom",
    "e_ionic", "e_electronic",
    
    # Mechanical max stress component
    "e_ij_max"
]

reducer = DimensionalityReducer(
    file_path="PIEmb.csv",
    columns_to_drop=columns_to_exclude,
    n_components=15
)

output_directory = os.getcwd()
reducer.load_and_prepare_data()
reducer.standardize()
reducer.apply_pca()
reducer.apply_factor_analysis()
reducer.save_results()
```

### 3. TVE Encoding of the Reduced Data
```python
from TransformerBasedVariantEncoder import TransformerBasedVariantEncoderRunner, setSeed
import pandas as pd

if __name__ == "__main__":
    # Example execution
    setSeed(1)
    dataPath = 'CrystalFeatureMatrix.csv'
    modelDir = 'checkpoints/transformer_models'
    outputDir = 'outputs/encoded_features'

    df = pd.read_csv(dataPath, dtype=str)
    dimModel = df.loc[:, df.columns[0]:'FA15'].shape[1]

    runner = TransformerBasedVariantEncoderRunner(
        dataPath=dataPath,
        modelDir=modelDir,
        outputDir=outputDir,
        dimModel=dimModel,
        numHeads=5
    )
    runner.run(maxLayers=8)
```

### 4. Perform hyperparameter optimization of TVE and downstream models

```python
from BayesianCV import DielectricPredictor
if __name__ == '__main__':
    # Example usage
    predictor = DielectricPredictor(
        data_filepath='./outputs/encoded_features/encodedOutputLayer8.csv',
        standardized_folder='./checkpoints/BayesianResult/ResultResNet/',
        hyperopt_results_csv='./hyperopt_results.csv',
        seed=1,
        n_splits=10,
        num_bins=10,
        max_evals=10,
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
```

### 5. Model performance validation
```python
from BayesianCV import ModelPlotter
# Example usage
file_path_pattern = './checkpoints/BayesianResult/ResultResNet/iteration_example/predictions_file/predictions_fold_{}.csv'
cv_file_path = './checkpoints/BayesianResult/ResultResNet/iteration_example/cross_validation_results/cv_results.csv'
output_folder = './outputs/fig/scatter_ResNet/'

plotter = ModelPlotter(file_path_pattern, cv_file_path, output_folder)
plotter.plot_all_models()
```

### 6. Feature Importance Analysis: A Case Study Using SHAP Values
```python
from FeatureImportance import SHAPFeatureImportance

# Set paths
model_dir = './checkpoints/BayesianResult/ResultResNet/iteration_example/save_models'
data_file = "./outputs/encoded_features/encodedOutputLayerExample.csv"

# Create class instance and run the process
shap_calc = SHAPFeatureImportance(model_dir, data_file, seed_value=1)
shap_calc.run()
```

## **Architecture**

---

### Overall Framework
Workflow of `DielecXformer` for predicting dielectric properties of perovskite, spinel, and binary metal oxides. 
- a. Embedding process: Elemental features are derived from `Magpie` and `Materials Project` data, while structural features include atomic arrangements, bonding motifs, and crystallographic parameters.  
- b. Feature engineering: Structure-agnostic and structure-aware descriptors are reduced via factor analysis to 15 latent variables. The final descriptors are fed into a Transformer-based Variational Encoder with optimized hyperparameters for model training.  
- c. TVE feature learning, TVE structure, and downstream hyperparameter optimization.

![DielecXformer Framework](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/Overall%20framework.png)

### Feature Importance Analysis Framework

Using SHAP (SHapley Additive exPlanations) and Permutation Importance (PERM), we systematically decompose feature contributions in ten independently trained residual neural networks. Due to differences in parameter spaces across models, local feature importance rankings show model-specific variability. To address this, we average results across models to compute global feature importance for each method.

![Feature Importance](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/FeatureImportance.png)

## **Visualization**

---

You can easily visualize Bayesian hyperparameter optimization results, model performance evaluation results, feature importance analysis, and comparisons of linear/non-linear correlations between original/synthetic descriptors and dielectric constant. The related Jupyter notebook files can be found at:

1. Bayesian hyperparameter optimization results, model performance evaluation, and feature importance analysis: `./TVE/visualization.ipynb`  
2. Comparison of original/synthetic descriptors and dielectric constant (linear/non-linear): `./TVE/ComparisonforODandSD.ipynb`  

### 1. Bayesian Hyperparameter Optimization Results  
![BayesianResult](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/BayesianResult.gif)

### 2. Model Performance Evaluation Results  
![ModelPerformance](https://github.com/ai4chips/DielecXformer/raw/main/TVE/outputs/fig/scatter_ResNet/all_scatter.png)

### 3. Feature Importance Analysis (e.g., SHAP Values)  
![SHAP](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/SHAP.png)

### 4. Comparison of Original/Synthetic Descriptors and Dielectric Constant (Linear/Non-linear)  
![ODandSD](https://github.com/ai4chips/DielecXformer/raw/main/docs/figs/ODandSD.png)

## **Citation**  

---  
If you use this repository, please cite:  
1. Zikang Guo, Shenghong Ju. *Project Title*. GitHub repository: [https://github.com/ai4chips/DielecXformer.git](https://github.com/ai4chips/DielecXformer.git), 2025. Manuscript in preparation.

## **License**  

---  
This project is distributed under the terms of the MIT License. For full license details, please refer to the LICENSE.md file provided in this repository.
