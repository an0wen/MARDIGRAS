# MARDIGRAS
The MAss-Radius DIaGRAm with Sliders (MARDIGRAS) is a visualization tool that allows a simple and easy manipulation of mass-radius relationships (also known as iso-composition curves) with interactive sliders.

## Run the tool
To run the program, download the repository and run with python:
```
git clone https://github.com/an0wen/MARDIGRAS
cd MARDIGRAS
python3 mardigras.py
```

## Description
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
The exoplanet catalog can now be easily updated using the NASA Exoplanet Archive's Table Access Protocol (TAP). Use the link below, and simply copy/paste all the content generated in your browser to the .dat file. Make sure to keep the header, and update the header with the relevant information. It is possible to use either actual planet mass (pl_masse), or the best available estimate among M, M*sin(i), or M*sin(i)/sin(i) (pl_bmasse). The two corresponding links are (choose one):

1. https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+default_flag=1&format=tsv
2. https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_eqt+from+ps+where+default_flag=1&format=tsv

It is recommended to use the first option, since an inaccurate mass results in the wrong position in the mass-radius plane, which can lead to a misinterpretation of planet composition.

The catalog of targets must have the same formatting as the full catalog of exoplanets. Targets can either be a subset of the full catalog, or completely different planets with values of parameters that are user-defined.
