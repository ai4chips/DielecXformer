# Notes on Magpie Integration in Matminer

Although **Magpie** has been integrated into the `matminer` library, inspection of the `matminer.utils.data.MagpieData` class reveals that it only provides two functions:

- `get_elemental_property(elem, property_name)`
- `get_oxidation_states(elem)`

Clearly, it does not contain any information related to **material structure**.

In the original Magpie Java project’s API documentation ([https://wolverton.bitbucket.io/javadoc/index.html](https://wolverton.bitbucket.io/javadoc/index.html)), the following two packages are provided:

- `magpie.attributes.generators.composition`  
  *Attributes based on the composition of a material.*
- `magpie.attributes.generators.crystal`  
  *Tools to generate attributes based on crystal structures.*

Please note that the original Java version of Magpie is **no longer maintained**. Only the **composition-based descriptor** functionalities have been incorporated into the `matminer` library.

---

## Structure Descriptors in Matminer

This code generates several structure-related features. The descriptions of the individual descriptors are as follows:

### `DensityFeatures`
This class calculates density and density-like features.

**Features**:
- Density
- Volume per atom ("vpa")
- Packing fraction

### `RadialDistributionFunction`
This class calculates the radial distribution function (RDF) of a crystal structure.

**Features**:
- Radial distribution function. Each feature is the "density" of the distribution at a certain radius.

**Args**:
- `cutoff`: (float) Angstrom distance up to which to calculate the RDF.
- `bin_size`: (float) Size in Angstrom of each bin of the (discrete) RDF.

**Attributes**:
- `bin_distances` (np.Ndarray): The distances each bin represents. Can be used for graphing the RDF.

### `GlobalSymmetryFeatures`
This class determines symmetry features such as spacegroup number and crystal system.

**Features**:
- Spacegroup number
- Crystal system (1 of 7)
- Centrosymmetry (has inversion symmetry)
- Number of symmetry operations, obtained from the spacegroup

**Crystal System Mapping**:
```python
crystal_idx = {
    "triclinic": 7,
    "monoclinic": 6,
    "orthorhombic": 5,
    "tetragonal": 4,
    "trigonal": 3,
    "hexagonal": 2,
    "cubic": 1,
}
```
---
## References

- 📦 **Installation Guide**: [https://wolverton.bitbucket.io/installation.html](https://wolverton.bitbucket.io/installation.html)  
- 📘 **Tutorial**: [https://wolverton.bitbucket.io/tutorial.html](https://wolverton.bitbucket.io/tutorial.html)  

PDF versions of these guides are also included in this folder for offline reference.
