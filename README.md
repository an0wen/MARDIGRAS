# MARDIGRAS
**Mass-Radius DIaGRAm with Sliders (MARDIGRAS)** is a visualization tool that allows simple and intuitive manipulation of mass-radius relationships (also known as iso-composition curves) using interactive sliders.

While *mardigras* screen captures can be implemented in your scientific work (talks, posters, communications), if you are looking for paper-quality mass-radius diagrams we recommend the use of [*mr-plotter*](https://github.com/castro-gzlz/mr-plotter).

## Run the tool
### Download and run
To run the program, download the repository and execute it with Python:
```
git clone https://github.com/an0wen/MARDIGRAS
cd MARDIGRAS
python3 mardigras.py
```
<img width="400" alt="Capture d’écran 2024-12-14 à 23 49 00" src="https://github.com/user-attachments/assets/a86b6f8c-7938-4849-8e9f-893a795ada97" />
<br/>
<br/>

### Available options
*mardigras* can be run with the following options:
1. Updating the NEA catalog:
   ```
   python3 mardigras.py --update-nea-catalog
   ```
   This will print the following statement in the terminal:
   ```
   Catalog updated successfully and saved to ./data/catalog_exoplanets.dat
   ```

### Contents of v1
Three curves are controlled by sliders:
1. Aguichine et al. 2021 (https://ui.adsabs.harvard.edu/abs/2021ApJ...914...84A/abstract): Represents an envelope of pure water in a supercritical state, with a pure steam atmosphere on top.
2. Lopez & Fortney 2014 (https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract): Represents an H-He gas envelope with metallicity ranging from 1xSolar to 50xSolar.
3. Zeng et al. 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z/abstract): For terrestrial planets with variable (iron) core mass fractions.

Additionally, five static profiles from Zeng et al. 2016 are shown, from bottom to top: pure iron core, Earth-like composition, pure mantle, 50% liquid water, and 100% liquid water.

Finally, three planet populations are shown:
1. Exoplanets from the NASA Exoplanet Archive (NEA) in the background.
2. A smaller sample of highlighted targets.
3. Planets of the Solar System.

Highlighted targets are intended for dedicated studies, discovery, or parameter updates of a few planets, a system, or a group of planets.

## Catalog update
### Automatic update
To update the NEA catalog, rnu the tool with the flag `--update-nea-catalog`:
```
python3 mardigras.py --update-nea-catalog
```
This will automatically fetch an updated version of the NEA and update the catalog file. Access to internet is necessary to perform this operation.

### Explanation
The NEA catalog is updated using NEA's Table Access Protocol (TAP). The TAP allows to query the content of the catalog in machine readable format using a single link. Custom filters can be included in the link. The full link used in *mardigras* is:

https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv

The following arguments have been added to the query:
- The default flag is 1
- The controversial flag is 0
- The planet radius is not null
- The planet mass is not null
- The planet mass represents the actual mass, i.e., not Msini or Msini/sini
- *hardcoded in the script*: the error on the planet mass is lower than 50%

Please refer to the [NEA website](https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html) for the full list and description of all catalog options.

### Manual update
Since *mardigras* is a tool to infer composition based on mass and radius (and other parameters), it is critical to use actual measurements of mass and radius and avoid values that are upper/lower limits, derived from empirical mass-radius relations, or are somewhat controversial. That being said, users are free to add or remove constraints depending on their goal. To do so, the user can manually enter their customized link in a web browser, and copy-paste the content in the `./data/catalog_exoplanets.dat` file. The header can be of any length, as long as each line begins with `#`. The file must contain at least 7 columns separated by tabs. Extra columns will be ignored. The program can handle empty entries for mass, radius, and their error bars.

The catalog of targets (planets shown as stars on the figure) has the same formatting as the full catalog. It is recommended to have at most 7 highlighted targets to avoid overloading the figure.

## Credits

If you use *mardigras*, please give credit to the initial release:
```
@article{Aguichine_2024,
doi = {10.3847/2515-5172/ad7506},
url = {https://dx.doi.org/10.3847/2515-5172/ad7506},
year = {2024},
month = {aug},
publisher = {The American Astronomical Society},
volume = {8},
number = {8},
pages = {216},
author = {Artyom Aguichine},
title = {mardigras: A Visualization Tool of Theoretical Mass–Radius Relations in the Context of Planetary Science},
journal = {Research Notes of the AAS},
abstract = {Over the past two decades, mass–radius relations have become a crucial tool for inferring the bulk composition of exoplanets using only their measured masses and radii. These relations, often referred to as isocomposition curves, are derived from interior structure models by calculating the theoretical radius as a function of mass for a given fixed planetary composition. Each mass–radius curve can be influenced by a variety of parameters, such as planetary composition, age, and equilibrium temperature. Navigating this parameter space can be cumbersome, particularly when models or their results are not open-source. To address this challenge, I have developed MAss–Radius DIaGRAm with Sliders, a visualization tool that enables simple, fast, and interactive exploration of the parameter space that governs mass–radius relations for any given model.}
}
```
