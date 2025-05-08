import os
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.conversions import StructureToComposition
from matminer.featurizers import composition as cf
from matminer.featurizers import structure as sf
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.ext.matproj import MPRester


class SCFeatureExtractor:
    def __init__(self, folder="POSCAR_batch", output_file="1.csv"):
        """
        Initializes the feature extractor class
        :param folder: Path to the folder containing structure files (POSCAR, .vasp)
        :param output_file: Path to save the resulting CSV file
        """
        self.folder = folder
        self.output_file = output_file

        # Define composition feature extractor
        self.composition_featurizer = MultipleFeaturizer([
            cf.Stoichiometry(),
            cf.ElementProperty.from_preset("magpie"),
            cf.ValenceOrbital(props=["avg"]),
            cf.IonProperty(fast=True),
            cf.BandCenter(),
            cf.AtomicOrbitals()
        ])

        # Define structure feature extractor
        self.structure_featurizer = MultipleFeaturizer([
            sf.DensityFeatures(),
            sf.RadialDistributionFunction(cutoff=10.0),
            sf.GlobalSymmetryFeatures()
        ])

    @staticmethod
    def parse_filename(filename):
        """
        Parse the filename to extract MP-ID and formula
        :param filename: Filename
        :return: (mp_id, formula)
        """
        base = os.path.splitext(filename)[0]
        if '-' in base:
            mp_id, formula = base.rsplit('-', 1)
        else:
            mp_id, formula = "unknown_id", base
        return mp_id, formula

    def extract_features(self):
        """
        Extract features and save them as CSV
        """
        # Set pandas options for displaying full DataFrame
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        dataframes = []

        # Iterate through all files in the folder
        for file in os.listdir(self.folder):
            if file.endswith(".vasp") or file.startswith("POSCAR"):
                file_path = os.path.join(self.folder, file)

                try:
                    structure = Structure.from_file(file_path)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                # Parse MP-ID and formula from filename
                mp_id, formula = self.parse_filename(file)
                print("=" * 60)
                print(f"[Material] Formula: {formula:<20} MP-ID: {mp_id}")
                print("-" * 60)

                df = pd.DataFrame({'structure': [structure]})

                # Step 1: Convert structure to composition
                print("[Step 1] Converting Structure To Composition...")
                df = StructureToComposition().featurize_dataframe(df, 'structure', pbar=False)
                print("[Step 1] Completed.")
                print("-" * 60)

                # Step 2: Extract composition features
                print("[Step 2] Extracting composition features...")
                df = self.composition_featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True, pbar=False)
                print("[Step 2] Completed.")
                print("-" * 60)

                # Step 3: Extract structure features
                print("[Step 3] Extracting structure features...")
                df = self.structure_featurizer.featurize_dataframe(df, col_id='structure', ignore_errors=True, pbar=False)
                print("[Step 3] Completed.")

                print("[Done] Feature extraction finished for this material.")
                print("=" * 60 + "\n")
                print("\n")


                # Insert mp-id and Formula at the beginning of the DataFrame
                df.insert(0, "mp-id", mp_id)
                df.insert(1, "Formula", formula)

                # Drop unnecessary columns
                df = df.drop(columns=["structure", "composition"], errors='ignore')

                dataframes.append(df)

        # If data was collected, concatenate and save to CSV
        if dataframes:
            result = pd.concat(dataframes, ignore_index=True)
            result.to_csv(self.output_file, index=False)
            print(f"Features saved to {self.output_file}. Total structures processed: {len(dataframes)}")
        else:
            print("No structure files were successfully processed.")


class MPPropertiesExtractor:
    def __init__(self, folder="POSCAR_batch", output_file="mp_properties.csv", api_key=None):
        """
        Initializes the MPPropertiesExtractor class
        :param folder: Path to the folder containing structure files (POSCAR, .vasp)
        :param output_file: Path to save the resulting CSV file
        :param api_key: API key for the Materials Project (if None, should be set in the environment)
        """
        self.folder = folder
        self.output_file = output_file
        self.api_key = api_key or os.getenv("MP_API_KEY")  # API key can be set in environment variables

        # Initialize MPRester object for querying the Materials Project
        self.mpr = MPRester(self.api_key)

    @staticmethod
    def parse_filename(filename):
        """
        Parse the filename to extract MP-ID and formula
        :param filename: Filename
        :return: (mp_id, formula)
        """
        base = os.path.splitext(filename)[0]
        if '-' in base:
            mp_id, formula = base.rsplit('-', 1)
        else:
            mp_id, formula = "unknown_id", base
        return mp_id, formula

    def extract_properties(self):
        """
        Extract properties for materials in the specified folder and save to a CSV file
        """
        results = []

        # Iterate through all files in the folder
        for file in os.listdir(self.folder):
            if file.endswith(".vasp") or file.startswith("POSCAR"):
                mp_id, formula = self.parse_filename(file)
                print("=" * 60)
                print(f"[Query] MP-ID: {mp_id:<20} Formula: {formula}")
                print("-" * 60)

                if not mp_id.startswith("mp-"):
                    print(f"Skipped: Invalid MP-ID in filename: {file}")
                    continue

                try:
                    summary = self.mpr.summary.search(material_ids=[mp_id])
                    if summary:
                        s = summary[0]
                        result = {
                            "mp-id": mp_id,
                            "Formula": formula,
                            "Band gap (eV)": s.band_gap,
                            "Formation energy per atom (eV)": s.formation_energy_per_atom,
                            "Energy above hull (eV)": s.energy_above_hull,
                            "Density (g/cm³)": s.density,
                            "Volume (Å³)": s.volume,
                            "Sites": s.nsites,
                            "Crystal system": s.symmetry.crystal_system,
                        }
                        results.append(result)
                        print("[Success] Properties retrieved.")
                    else:
                        print("[Warning] No data found in MP database for this MP-ID.")
                except Exception as e:
                    print(f"Error: Failed to retrieve data for {mp_id}: {e}")

                print("=" * 60 + "\n")

        # Save results to CSV file
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_file, index=False)
            print(f"Saved {len(results)} entries to '{self.output_file}'")
        else:
            print("No valid entries were retrieved.")
