# MARDIGRAS
The MAss-Radius DIaGRAm with Sliders (MARDIGRAS) is a visualization tool that allows a simple and easy manipulation of mass-radius relationships (also known as iso-composition curves) with interactive sliders.

## Run the tool
To run the program, download the repository and run with python:
```
git clone https://github.com/an0wen/MARDIGRAS
cd MARDIGRAS
python3 mardigras.py
```
<img width="400" alt="Capture d’écran 2024-08-21 à 18 25 24" src="https://github.com/user-attachments/assets/d64cfffe-9163-442d-9fb6-c0778821b8a9">
<br/>
<br/>

Three curves are controlled by sliders:
1. Aguichine et al. 2021 (https://ui.adsabs.harvard.edu/abs/2021ApJ...914...84A/abstract), where the envelope is pure water in supercritical state, and a pure steam atmosphere ontop
2. Lopez & Fortney 2014 (https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract), where the envelope is made of H-He gas with 1xSolar to 50xSolar metallicity
3. Zeng et al. 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z/abstract), for terrestrial planets with variable (iron) core mass fraction

5 static profiles from Zeng et al. 2016 are also shown, from bottom to top: pure iron core, Earth-like, pure mantle, 50% liquid water, and 100% liquid water.

Finally, three planet populations are shown:
1. Full exoplanet catalog in the background
2. A smaller sample of highlighted targets
3. Planets of the Solar System

Highlighted targets are intended for dedicated studies, discovery, or parameter update of a few planets/a system/a group of planets.

## Catalogs update
The exoplanet catalog can now be easily updated using the NASA Exoplanet Archive's Table Access Protocol (TAP). Use the link below, and simply copy/paste all the content generated in your browser to the `data/catalog_exoplanets.dat` file. It is recommended to keep the header, and update the header with the relevant information. For mardigras, we recommend making the following query:

https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv

where the following arguments have been added to the query:
- the default flag is 1, to avoid redundancy
- the controversial flag is 0
- the planet radius is not null
- the planet mass is not null
- the planet mass represents the actual mass, i.e. not Msini or Msini/sini
- *hardcoded in the script*: the error on the planet mass is lower than 50%

Since mardigras is a tool to infer composition based on mass and radius (and other extra parameters), it is critical to use actual measurements of mass and radius, and avoid values that are upper/lower limits, derived from empirical mass-radius relations, or are somewhat controversial. This being said,  users are free to add or remove constraints depending on their goal.

The header can be removed, or additional lines can be added (comment lines start with `#`), but the file must contain at least 7 columns separated by tabulations. Extra columns can be present and will be ignored. The program can handle empty entries for mass, radius, and their error bars.

The catalog of targets (planets shown as stars on the figure) has the same formatting as the full catalog. It is recommended to have at most 7 highlighted targets to avoid overburden the figure.
