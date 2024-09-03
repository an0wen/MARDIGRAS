# MARDIGRAS
**Mass-Radius DIaGRAm with Sliders (MARDIGRAS)** is a visualization tool that allows simple and intuitive manipulation of mass-radius relationships (also known as iso-composition curves) using interactive sliders.

## Run the tool
To run the program, download the repository and execute it with Python:
```
git clone https://github.com/an0wen/MARDIGRAS
cd MARDIGRAS
python3 mardigras.py
```
<img width="400" alt="Capture d’écran 2024-08-21 à 18 25 24" src="https://github.com/user-attachments/assets/d64cfffe-9163-442d-9fb6-c0778821b8a9">
<br/>
<br/>

Three curves are controlled by sliders:
1. Aguichine et al. 2021 (https://ui.adsabs.harvard.edu/abs/2021ApJ...914...84A/abstract): Represents an envelope of pure water in a supercritical state, with a pure steam atmosphere on top.
2. Lopez & Fortney 2014 (https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract): Represents an H-He gas envelope with metallicity ranging from 1xSolar to 50xSolar.
3. Zeng et al. 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z/abstract): For terrestrial planets with variable (iron) core mass fractions.

Additionally, five static profiles from Zeng et al. 2016 are shown, from bottom to top: pure iron core, Earth-like composition, pure mantle, 50% liquid water, and 100% liquid water.

Finally, three planet populations are shown:
1. Full exoplanet catalog in the background.
2. A smaller sample of highlighted targets.
3. Planets of the Solar System.

Highlighted targets are intended for dedicated studies, discovery, or parameter updates of a few planets, a system, or a group of planets.

## Catalogs update
The exoplanet catalog can be easily updated using NASA's Exoplanet Archive's Table Access Protocol (TAP). Use the link below, and copy/paste the content generated in your browser into the data/catalog_exoplanets.dat file. It is recommended to keep the header and update it with the relevant information. For MARDIGRAS, we recommend the following query:

https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv

The following arguments have been added to the query:
- The default flag is 1, to avoid redundancy
- The controversial flag is 0
- The planet radius is not null
- The planet mass is not null
- The planet mass represents the actual mass, i.e., not Msini or Msini/sini
- *hardcoded in the script*: the error on the planet mass is lower than 50%

Since *mardigras* is a tool to infer composition based on mass and radius (and other parameters), it is critical to use actual measurements of mass and radius and avoid values that are upper/lower limits, derived from empirical mass-radius relations, or are somewhat controversial. That said, users are free to add or remove constraints depending on their goal.

The header can be removed, or additional lines can be added (comment lines start with #), but the file must contain at least 7 columns separated by tabs. Extra columns can be present and will be ignored. The program can handle empty entries for mass, radius, and their error bars.

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
