##############################################
#
#   Mass-Radius DIaGRAm with Sliders (MARDIGRAS)
#
##############################################

# Math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Plot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import patheffects

# I/O
import os
from pathlib import Path
from datetime import datetime
import tomllib
import argparse
import pandas as pd

# Internet
import requests
import xml.etree.ElementTree as ET


# Load configuration
def _deep_update(base, override):
    '''
    Takes base config, and overrides default parameters from additional config file.
    '''
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(default_path="./data/default_config.toml", override_paths=None):
    '''
    Load default and optional config override.
    '''
    with open(default_path, "rb") as f:
        config = tomllib.load(f)

    if override_paths:
        for path in override_paths:
            with open(path, "rb") as f:
                override = tomllib.load(f)
            _deep_update(config, override)

    return config

# Optional: reading path to user-defined config file.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to override config file (TOML, argument can be used multiple times)",
    )
    return parser.parse_args()

# Fetch potential custom config files
args = get_args()

# Load configuration (default config) and overrides with custom files, if provided
mardigras_config = load_config(override_paths=args.config)


# Paths to interior structure model grids
path_models = mardigras_config["models"]["path"]


##############################################
#
#   MAKE SWEET INTERPOLATOR (Aguichine+2025)
#
##############################################

path_sweet = path_models + "Aguichine2025_SWEET_all.dat"

# top of the atmosphere pressure (index)
sweet_top = np.array([0,1])
sweet_labels_top = ["20 mbar", "1 µbar"]
# host star type
sweet_labels_host = ["Type M", "Type G"]
# equilibrium temperature at zero albedo (K)
sweet_teqs = np.array([400, 500, 700, 900, 1100, 1300, 1500])
# bulk water mass fraction (%)
sweet_wmfs = np.array([0.1,1,10,20,30,40,50,60,70,80,90,100])
# total planet mass (Me)
sweet_masses = np.array([0.2       ,  0.254855  ,  0.32475535,  0.41382762,  0.52733018,
        0.67196366,  0.85626648,  1.09111896,  1.39038559,  1.77173358,
        2.25767578,  2.87689978,  3.66596142,  4.67144294,  5.95270288,
        7.58538038,  9.66586048, 12.31696422, 15.69519941, 20.        ])
# planet age (Gyr)
sweet_ages = np.array([0.001,0.0015,0.002,0.003,0.005,0.01,
                        0.02,0.03,0.05,
                        0.1,0.2,0.5,
                        1.0,2.0,5.0,
                        10,20])

# make dimension axes for interpolator
sweet_dim_wmf = sweet_wmfs.copy()/100
sweet_dim_teq = sweet_teqs.copy()
sweet_dim_mass = sweet_masses.copy()
sweet_dim_age = sweet_ages.copy()
sweet_dim_star = np.array([0,1])
sweet_dim_top = np.array([0,1])

# read radius tracks data and reshape
listrpfull1 = np.loadtxt(path_sweet,skiprows=36,unpack=True,usecols=(6))
listrpfull2 = np.loadtxt(path_sweet,skiprows=36,unpack=True,usecols=(7))
listrpfull = np.array((listrpfull1,listrpfull2))
listrpfull = np.delete(listrpfull, np.arange(17, listrpfull.size, 18))

sweet_data_radius = np.reshape(listrpfull,(2,2,12,7,20,17))
mask = np.isnan(sweet_data_radius)
sweet_data_radius[mask] = -1

# fill_value: value to return when extrapolating beyond grid validity range
fill_value = np.nan
interp_sweet = RegularGridInterpolator((sweet_dim_top,
                                      sweet_dim_star,
                                      sweet_dim_wmf,
                                      sweet_dim_teq,
                                      sweet_dim_mass,
                                      sweet_dim_age), 
                                     sweet_data_radius, method='slinear', bounds_error=False, fill_value=fill_value)

##############################################
#
#   MAKE ZENG INTERPOLATOR (Zeng+2016)
#
##############################################

# Load curves from Zeng et al. 2016
path_zeng = path_models + "Zeng2016.dat"

# table containing mass-radius curves for different compositions, used for the Zeng water curve
zeng_mass,zeng_purefe,zeng_rock,zeng_50wat,zeng_100wat,zeng_earth = np.loadtxt(path_zeng,delimiter="\t",skiprows=1,unpack=True)

# single-composition files, much larger mass range and many more points, used for rocky curves and interpolation
array_zeng_fe_m,array_zeng_fe_r = np.loadtxt(path_models+"zeng2016-iron.dat",unpack=True,usecols=(0,1))
array_zeng_ea_m,array_zeng_ea_r = np.loadtxt(path_models+"zeng2016-earth.dat",unpack=True,usecols=(0,1))
array_zeng_mg_m,array_zeng_mg_r = np.loadtxt(path_models+"zeng2016-rock.dat",unpack=True,usecols=(0,1))

array_zeng_masses = np.logspace(np.log10(0.01), np.log10(100.0), 30)

# interpolate so that the three rocky curves share the same mass array
array_zeng_fe_radii = np.interp(array_zeng_masses, array_zeng_fe_m, array_zeng_fe_r)
array_zeng_ea_radii = np.interp(array_zeng_masses, array_zeng_ea_m, array_zeng_ea_r)
array_zeng_mg_radii = np.interp(array_zeng_masses, array_zeng_mg_m, array_zeng_mg_r)

# create interpolator for CMF on rocky curves
dimcmf_zeng = np.array([0.0,0.325,1.0])
data_zeng = np.vstack((array_zeng_mg_radii, array_zeng_ea_radii,array_zeng_fe_radii)).T
interp_zeng = RegularGridInterpolator((array_zeng_masses,dimcmf_zeng), data_zeng, method='slinear', bounds_error=False, fill_value=None)


##############################################
#
#   MAKE GAS DWARF INTERPOLATOR (Tang+2025)
#
##############################################

# dimensions of the T25 grid
dim_met_t25 = np.array([1.0,50.0])
dim_age_t25 = np.array([0.01,0.1,1.0,10.0])
dim_logage_t25 = np.log10(dim_age_t25)
dim_finc_t25= np.array([1.0,10.0,100.0,1000.0])
dim_teq_t25 = 278.0*(dim_finc_t25)**(0.25)
dim_mass_t25= np.array([1,2,3,4,5,6,8,10,13,16,20])
dim_fenv_t25= np.array([0.10,0.20,0.50,1,2,5,10,20])
dim_top_t25 = np.array([0,1,2])
t25_labels = ["RCB", "20 mbar", "1 nbar"]

# load data of T25 model
t25_data_radius = np.zeros((3,2,4,4,11,8))

data0 = np.genfromtxt(path_models+"Tang2025.dat",filling_values=fill_value,comments='#',skip_header=1,usecols=(5,6,7,8))
t25_data_radius[0,:,:,:,:,:] = data0[:,0].reshape(2,4,4,11,8)
t25_data_radius[1,:,:,:,:,:] = data0[:,1].reshape(2,4,4,11,8)
t25_data_radius[2,:,:,:,:,:] = data0[:,2].reshape(2,4,4,11,8)

# construct interpolator on T25 grid
interp_t25 = RegularGridInterpolator((dim_top_t25,
                                      dim_met_t25, 
                                      dim_logage_t25, 
                                      dim_teq_t25, 
                                      dim_mass_t25, 
                                      dim_fenv_t25), 
                                      t25_data_radius, method='slinear', bounds_error=False, fill_value=fill_value)

# boil-off limit: for a given mass, teq, and metallicity, 
# what is the maximum f_env the planet can start with
t25_bolim_fenv = np.zeros((2,8,11))
t25_bolim_radius = np.zeros((3,2,8,11))
data3 = np.genfromtxt(path_models+"Tang25_boil-off.csv",delimiter=",",filling_values=20.0,comments='#',skip_header=0)
data3 = data3[:,:12]
t25_bolim_fenv = data3[:,1:].reshape(2,8,11)
t25_bolim_fenv = np.clip(t25_bolim_fenv,a_min=0.0,a_max=20)

dim_finc_t25_bolim= np.array([1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0])
dim_teq_t25_bolim = 278.0*(dim_finc_t25_bolim)**(0.25)

interp_t25_bolim_maxf = RegularGridInterpolator((dim_met_t25, dim_teq_t25_bolim, dim_mass_t25), t25_bolim_fenv, method='slinear', bounds_error=False, fill_value=np.inf)

##############################################
#
#   LOAD DATA
#
##############################################

# NASA Exoplanet catalog

def check_internet_connection():
    """Check if there is internet access by pinging a known URL."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def update_nea_exoplanet_catalog(catalog_url, output_file):
    """
    Updates the NEA exoplanet catalog from the NASA Exoplanet Archive TAP.
    Parameters:
        catalog_url (str): The TAP URL with the SQL query for the catalog.
        output_file (str): Path to save the downloaded catalog.
    """
    if not check_internet_connection():
        print("No internet connection. Unable to update the exoplanet catalog.")
        return

    try:
        response = requests.get(catalog_url, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
        
        # Prepare the header
        header = [
            "# NASA Exoplanet Catalog",
            f"# Source: {catalog_url}",
            f"# Catalog last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        # Write the header and data to the file
        with open(output_file, "w") as f:
            f.write("\n".join(header) + "\n")
            # Add '#' to the parameter names
            data_lines = response.text.splitlines()
            f.write("# " + data_lines[0] + "\n")  # Add # to parameter names
            f.write("\n".join(data_lines[1:]) + "\n")
        
        print(f"Catalog updated successfully and saved to {output_file}")
    except requests.RequestException as e:
        print(f"Error fetching the catalog: {e}")

def read_nea_last_update(output_file):
    """
    Reads the date of the last update from the catalog file and prints it.
    Parameters:
        output_file (str): Path to the catalog file.
    """
    if not os.path.exists(output_file):
        print("Catalog file does not exist.")
        return

    try:
        with open(output_file, "r") as f:
            for line in f:
                text_lookup = "# Catalog last updated:"
                if line.startswith(text_lookup):
                    print("# NEA catalog last updated: ",line.replace(text_lookup,'').replace('\n',''))
                    break
    except Exception as e:
        print(f"Error reading the catalog: {e}")

# File paths and URL of the NEA Catalog
nea_catalog_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
               "query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,"
               "pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+"
               "default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+"
               "and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv")
nea_output_file = mardigras_config["catalog"]["paths"]["nea"]

# Update NEA catalog
if mardigras_config["catalog"]["update_nea"]:
    # Update the catalog
    update_nea_exoplanet_catalog(nea_catalog_url, nea_output_file)
else:
    # Check for existing catalog and print the last update date
    read_nea_last_update(nea_output_file)


# PlanetS catalog

def update_planets_exoplanet_catalog(catalog_url, output_file):
    """
    Updates the PlanetS exoplanet catalog from the DACE website.
    Warning: Work in progress, currently not working
    """
    # if True:
    #     print("PlanetS auto update currently not implemented, skipping updating attempt.")
    #     return

    if not check_internet_connection():
        print("No internet connection. Unable to update the exoplanet catalog.")
        return

    try:
        # Fetching the PlanetS catalog from internet. Code provided by Léna Parc + adapted by Artem Aguichine
        response = requests.get(catalog_url, verify=False, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Parse XML content
        root = ET.fromstring(response.content)

        # VOTable namespace
        # This line defines a dictionary that maps the prefix 'v' so that the XML parser knows it belongs to the VOTable namespace defined by the IVOA standard
        ns = {'v': 'http://www.ivoa.net/xml/VOTable/v1.3'} 

        # Get field names (column headers)
        fields = root.findall(".//v:FIELD", ns)
        column_names = [f.attrib['name'] for f in fields]
        # print("Columns:", column_names)

        # Get data rows (in TABLEDATA format)
        rows = root.findall(".//v:TR", ns)

        # Extract data row-by-row
        data = []
        for tr in rows:
            values = [td.text for td in tr.findall("v:TD", ns)]
            data.append(values)

        # Convert to pandas dataframe
        df_planets_catalog_full = pd.DataFrame(data, columns=column_names)
        
        MJUP_TO_MEARTH = 317.828
        RJUP_TO_REARTH = 11.209
        df_planets_catalog_keep = pd.DataFrame({
            "pl_name": df_planets_catalog_full["Planet Name"],
            "pl_rade": pd.to_numeric(df_planets_catalog_full["Planet Radius [Rjup]"]) * RJUP_TO_REARTH,
            "pl_radeerr1": pd.to_numeric(df_planets_catalog_full["Planet Radius - Upper Unc [Rjup]"]) * RJUP_TO_REARTH,
            "pl_radeerr2": pd.to_numeric(df_planets_catalog_full["Planet Radius - Lower Unc [Rjup]"]) * RJUP_TO_REARTH,
            "pl_masse": pd.to_numeric(df_planets_catalog_full["Planet Mass [Mjup]"]) * MJUP_TO_MEARTH,
            "pl_masseerr1": pd.to_numeric(df_planets_catalog_full["Planet Mass - Upper Unc [Mjup]"]) * MJUP_TO_MEARTH,
            "pl_masseerr2": pd.to_numeric(df_planets_catalog_full["Planet Mass - Lower Unc [Mjup]"]) * MJUP_TO_MEARTH,
            "pl_eqt": pd.to_numeric(df_planets_catalog_full["Equilibrium Temperature [K]"]),
        })

        # Writing the catalog to local file
        # Prepare the header
        header = [
            "# DACE PlanetS Catalog",
            f"# Source: {catalog_url}",
            f"# Catalog last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        # Write the header and data to the file
        with open(output_file, "w") as f:
            f.write("\n".join(header) + "\n")

            f.write("# " + " ".join(df_planets_catalog_keep.columns) + "\n")

            df_planets_catalog_keep.to_csv(f, sep="\t", index=False, header=False)

        print(f"Catalog updated successfully and saved to {output_file}")
    except requests.RequestException as e:
        print(f"Error fetching the catalog: {e}")

def read_planets_last_update(output_file):
    """
    Reads the date of the last update from the catalog file and prints it.
    Parameters:
        output_file (str): Path to the catalog file.
    """
    if not os.path.exists(output_file):
        print("Catalog file does not exist.")
        return

    try:
        with open(output_file, "r") as f:
            for line in f:
                text_lookup = "# Catalog last updated:"
                if line.startswith(text_lookup):
                    print("# PlanetS catalog last updated: ",line.replace(text_lookup,'').replace('\n',''))
                    break
    except Exception as e:
        print(f"Error reading the catalog: {e}")

# File paths and URL of the PlanetS Catalog
planets_catalog_url = ("https://dace.unige.ch/downloads/exoplanets/PLANETS.vot")
planets_output_file = mardigras_config["catalog"]["paths"]["planets"]

# Update PlanetS catalog
if mardigras_config["catalog"]["update_planets"]:
    # Update the catalog
    update_planets_exoplanet_catalog(planets_catalog_url, planets_output_file)
else:
    # Check for existing catalog and print the last update date
    read_planets_last_update(planets_output_file)

# Load exoplanet catalog
def load_exoplanet_catalog(catalog_name="NEA"):
    if catalog_name == "NEA":
        # Load NEA data
        # find the line containing column names, for header of variable length
        with open(nea_output_file) as nea_f:
            nea_lines = nea_f.readlines()
        nea_header_idx = None
        for i, line in enumerate(nea_lines):
            if line.startswith("#"):
                nea_header_idx = i  # keeps updating → last # line
        nea_colnames = nea_lines[nea_header_idx].lstrip("#").strip().split()

        data_nea = pd.read_csv(nea_output_file,
                            sep="\t",
                            skiprows=nea_header_idx + 1,
                            names=nea_colnames,
                            comment="#")

        # Remove lines with empty Mp, Rp, or mass precision less than 50%
        exo_mass_prec = 0.5
        filter_nea = (data_nea["pl_rade"]>0) & (data_nea["pl_masse"]>0) \
        & ( abs(data_nea["pl_masseerr1"])/data_nea["pl_masse"] < exo_mass_prec) \
        & ( abs(data_nea["pl_masseerr2"])/data_nea["pl_masse"] < exo_mass_prec)

        catalog_exoplanets = data_nea[filter_nea]

        return nea_colnames,catalog_exoplanets


    elif catalog_name == "PlanetS":
        # Load PlanetS data
        # find the line containing column names, for header of variable length
        with open(planets_output_file) as planets_f:
            planets_lines = planets_f.readlines()
        planets_header_idx = None
        for i, line in enumerate(planets_lines):
            if line.startswith("#"):
                planets_header_idx = i  # keeps updating → last # line
        planets_colnames = planets_lines[planets_header_idx].lstrip("#").strip().split()

        data_planets = pd.read_csv(planets_output_file,
                            sep="\t",
                            skiprows=planets_header_idx + 1,
                            names=planets_colnames,
                            comment="#")

        # Remove lines with empty Mp and Rp
        filter_planets = (data_planets["pl_rade"]>0) & (data_planets["pl_masse"]>0)

        catalog_exoplanets = data_planets[filter_planets]

        return planets_colnames,catalog_exoplanets

catalog_colnames,catalog_exoplanets = load_exoplanet_catalog(catalog_name = mardigras_config["catalog"]["active"])

# Targets catalog
# The intended use is to showcase a few targets (dedicated study, new discovery, update of parameters, etc.)
# The catalog of targets must have the same formatting as the exoplanet catalog.

# Determine catalog targets path
default_catalog_targets = Path("./data/catalog_targets.dat")
path_targets = Path(mardigras_config["catalog"]["paths"]["targets"])
if path_targets and path_targets.exists():
    catalog_targets_path = path_targets
elif path_targets:
    print(f"Warning: Provided file '{path_targets}' does not exist. Using default: '{default_catalog_targets}'")
    catalog_targets_path = default_catalog_targets
else:
    catalog_targets_path = default_catalog_targets

# find the line containing column names, for header of variable length
with open(catalog_targets_path) as targets_f:
    targets_lines = targets_f.readlines()
targets_header_idx = None
for i, line in enumerate(targets_lines):
    if line.startswith("#"):
        targets_header_idx = i  # keeps updating → last # line
targets_colnames = targets_lines[targets_header_idx].lstrip("#").strip().split()

data_targets = pd.read_csv(catalog_targets_path,
                        sep="\t",
                        skiprows=targets_header_idx + 1,
                        names=targets_colnames,
                        comment="#")

catalog_targets = data_targets

##############################################
#
#   SOLAR SYSTEM DATA
#
##############################################

# Data for masses and radii of planets (in Earth masses and Earth radii)
ss_planet_data = {
    "Mercury": (0.055, 0.383),
    "Venus": (0.815, 0.949),
    "Earth": (1.0, 1.0),
    "Mars": (0.107, 0.532),
    "Jupiter": (317.8, 11.21),
    "Saturn": (95.2, 9.45),
    "Uranus": (14.6, 4.01),
    "Neptune": (17.2, 3.88),
}

# Alchemy symbols for planets
ss_alchemy_symbols = {
    "Mercury": "☿",
    "Venus": "♀",
    "Earth": "⊕",
    "Mars": "♂",
    "Jupiter": "♃",
    "Saturn": "♄",
    "Uranus": "♅",
    "Neptune": "♆",
}

# Extracting data
ss_planets = list(ss_planet_data.keys())
ss_masses = [ss_planet_data[planet][0] for planet in ss_planets]
ss_radii = [ss_planet_data[planet][1] for planet in ss_planets]
ss_symbols = [ss_alchemy_symbols[planet] for planet in ss_planets]


##############################################
#
#   MAKE PLOT
#
##############################################

# The parametrized function to be plotted
def rad_sweet(x,top_sweet,star_sweet,wmf_sweet,teq_sweet,age_sweet):
    input0 = np.stack((np.full(len(x),top_sweet),
                       np.full(len(x),star_sweet),
                       np.full(len(x),wmf_sweet),
                       np.full(len(x),teq_sweet),
                       x,
                       np.full(len(x),age_sweet)), axis=-1)
    return interp_sweet(input0)

t25_tolerance_main = 1.0
t25_tolerance_sec = 0.0
def rad_t25(x,top,met,age,teq,fenv,tolerance=t25_tolerance_sec):
    logage = np.log10(age)
    input0 = np.stack((np.full(len(x),top),
                       np.full(len(x),met),
                       np.full(len(x),logage),
                       np.full(len(x),teq),
                       x,
                       np.full(len(x),fenv)), axis=-1)
    rp = interp_t25(input0)
    input1 = np.stack((np.full(len(x),met),np.full(len(x),teq),x), axis=-1)
    boiloff_maxf = interp_t25_bolim_maxf(input1)
    isbad = np.full(len(x),fenv)*tolerance > boiloff_maxf
    rp[isbad] = np.nan
    return rp

def rad_zeng(x,cmf):
    input0 = np.stack((x,np.full(len(x),cmf)), axis=-1)
    rp = interp_zeng(input0)
    return rp



## Plot parameters
xmin, xmax, nx = 0.5, 30.0, 200
ymin, ymax     = 0.5, 4.5

x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

# Define initial parameters
init_host_sweet = 0
init_top_sweet = 1
init_wmf_sweet = 0.5
init_teq_sweet = 600.0
init_age_sweet = 1.0


init_met_t25 = 1.0
init_age_t25 = 1.0
init_teq_t25 = 600.0
init_fenv_t25 = 1.0
init_top_t25 = 1.0

init_cmf_zeng = 0.325

# Create the figure
fig = plt.figure(figsize=(7,7))

# make axes for main figure
ax_main = fig.add_axes([0.11, 0.1, 0.83, 0.6])  # [left, bottom, width, height]

# Figure options
ax_main.set_xscale("log")
ax_main.set_xlim(xmin, xmax)
ax_main.set_ylim(ymin, ymax)
ax_main.set_xlabel('Mass [Me]')
ax_main.set_ylabel('Radius [Re]')
ax_main.grid(visible=True,which='major', axis='both')

# Sliding lines
current_sweet = rad_sweet(x,init_top_sweet,init_host_sweet,init_wmf_sweet,init_teq_sweet,init_age_sweet)
line_sweet, = ax_main.plot(x, current_sweet,lw=2,color='blue',ls='-',zorder=20)
current_t25 = rad_t25(x,init_top_t25,init_met_t25,init_age_t25,init_teq_t25,init_fenv_t25,tolerance=t25_tolerance_main)
line_t25, = ax_main.plot(x, current_t25,lw=2,color='red',zorder=30)
line_zeng, = ax_main.plot(x, rad_zeng(x,init_cmf_zeng),lw=2,color='brown',zorder=10)

# Minor lines
line_sweet_mbar, = ax_main.plot(x, rad_sweet(x,0,init_host_sweet,init_wmf_sweet,init_teq_sweet,init_age_sweet),lw=0.5,color='blue',ls='--',zorder=19)
line_sweet_mibar, = ax_main.plot(x, rad_sweet(x,1,init_host_sweet,init_wmf_sweet,init_teq_sweet,init_age_sweet),lw=0.5,color='blue',ls='--',zorder=19)
line_t25_rcb, = ax_main.plot(x, rad_t25(x,0,init_met_t25,init_age_t25,init_teq_t25,init_fenv_t25),lw=0.5,ls='--',color='red',zorder=29)
line_t25_mbar, = ax_main.plot(x, rad_t25(x,1,init_met_t25,init_age_t25,init_teq_t25,init_fenv_t25),lw=0.5,ls='--',color='red',zorder=29)
line_t25_nbar, = ax_main.plot(x, rad_t25(x,2,init_met_t25,init_age_t25,init_teq_t25,init_fenv_t25),lw=0.5,ls='--',color='red',zorder=29)

# Exoplanets data (catalog and targets)
array_exo_mp = catalog_exoplanets["pl_masse"].to_numpy()
array_exo_rp = catalog_exoplanets["pl_rade"].to_numpy()

array_targets_mp = catalog_targets["pl_masse"].to_numpy()
array_targets_rp = catalog_targets["pl_rade"].to_numpy()

array_exo_mpe = [abs(catalog_exoplanets["pl_masseerr2"].to_numpy()), catalog_exoplanets["pl_masseerr1"].to_numpy()]
array_exo_rpe = [abs(catalog_exoplanets["pl_radeerr2"].to_numpy()), catalog_exoplanets["pl_radeerr1"].to_numpy()]
array_targets_mpe = [abs(catalog_targets["pl_masseerr2"].to_numpy()), catalog_targets["pl_masseerr1"].to_numpy()]
array_targets_rpe = [abs(catalog_targets["pl_radeerr2"].to_numpy()), catalog_targets["pl_radeerr1"].to_numpy()]

# Exoplanets from the catalog file
catalog_points = ax_main.errorbar(array_exo_mp,array_exo_rp,
            yerr=array_exo_rpe,
            xerr=array_exo_mpe,
            fmt='o',zorder=-30,
            c="black",alpha=0.2)

# Add hover annotations
annot = ax_main.annotate(
    "",
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
    zorder = 100
)
annot.set_visible(False)

def update_annot(ind):
    """Update the annotation based on the index of the closest point."""
    x, y = catalog_points[0].get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = f"{catalog_exoplanets["pl_name"].iloc[
ind['ind'][0]]}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    """Event handler for mouse motion."""
    vis = annot.get_visible()
    if event.inaxes == ax_main:
        cont, ind = catalog_points[0].contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# Exoplanets to be highlighted
ax_main.errorbar(array_targets_mp,array_targets_rp,
            yerr=array_targets_rpe,
            xerr=array_targets_mpe,
            ls='',c='orange',elinewidth=3,
            marker='*',mfc='orange',mec='black', ms=15, mew=1,
            zorder=50)

# Annotate each point
texts = []
# Use these to expand annotations horizontally and/or vertically
text_expand_h = False
text_expand_v = True
for i, label in enumerate(catalog_targets["pl_name"]):
    x_text = 10
    y_text = -15
    if text_expand_h: x_text = ((array_targets_mp[i]-np.median(array_targets_mp))/(np.max(array_targets_mp)-np.min(array_targets_mp)))*15
    if text_expand_v: y_text = ((array_targets_rp[i]-np.median(array_targets_rp))/(np.max(array_targets_rp)-np.min(array_targets_rp)))*15
    texts.append(ax_main.annotate(
    label, 
    (array_targets_mp[i], array_targets_rp[i]), 
    textcoords="offset points", 
    xytext=(x_text,y_text), 
    ha='left',
    fontsize='x-small',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7),zorder=50+i
    ))
    #plt.annotate(label, (array_targets_mp[i], array_targets_rp[i]), textcoords="offset points", xytext=(5,-10),
    #ha='left',bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7),zorder=50+i)

# Solar system planets
for i in range(len(ss_planets)):
    text = ax_main.text(ss_masses[i], ss_radii[i], ss_alchemy_symbols[ss_planets[i]], \
            fontsize=15, ha='center', va='center', color='red')
    text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])  # Add outline to text

# Zeng fixed
zeng_width=1.0
ax_main.plot(array_zeng_masses,array_zeng_fe_radii,linewidth=zeng_width,color='black',zorder=5)
ax_main.plot(array_zeng_masses,array_zeng_ea_radii,linewidth=zeng_width,color='brown',zorder=6)
ax_main.plot(array_zeng_masses,array_zeng_mg_radii,linewidth=zeng_width,color='grey',zorder=7)
ax_main.plot(zeng_mass,zeng_50wat,linewidth=zeng_width,color='cyan',zorder=8)
ax_main.plot(zeng_mass,zeng_100wat,linewidth=zeng_width,color='cyan',zorder=9)

# Zeng fixed labels
ax_main.text(0.55, 0.60, '100% Core',fontsize=5,
        color='black',rotation=10)
        
ax_main.text(0.55, 0.76, 'Earth-like',fontsize=5,
        color='brown',rotation=10)

ax_main.text(0.55, 0.93, '100% Mantle',fontsize=5,
        color='grey',rotation=15)

ax_main.text(0.55, 1.10, '50% Liquid H2O',fontsize=5,
        color='blue',rotation=17)

ax_main.text(0.55, 1.23, '100% Liquid H2O',fontsize=5,
        color='blue',rotation=17)

##############################################
#
#   MAKE SLIDERS
#
##############################################

# Aguichine+2025 Slider

# Label
fig.text(0.05, 0.95, 'Aguichine et al. 2025',weight='bold',
        color='blue',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Host Star
ax_host_sweet = fig.add_axes([0.08, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
host_sweet_slider = Slider(
    ax=ax_host_sweet,
    label='Star  ',
    valmin=0,
    valmax=1,
    valinit=init_host_sweet,
    valstep=1
)
host_sweet_slider.label.set_ha('left')
host_sweet_slider.label.set_position((-0.4, 0.5))
host_sweet_slider.valtext.set_ha('right')
host_sweet_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider
host_sweet_slider.valtext.set_text(sweet_labels_host[init_host_sweet])

# Make a horizontal oriented slider to control the WMF
ax_wmf_sweet = fig.add_axes([0.08, 0.75, 0.15, 0.02])  # [left, bottom, width, height]
wmf_sweet_slider = Slider(
    ax=ax_wmf_sweet,
    label="WMF  ",
    valmin=np.log10(0.001),
    valmax=np.log10(1.0),
    valinit=np.log10(init_wmf_sweet),
    #valfmt=' %1.3f'
)
wmf_sweet_slider.label.set_ha('left')
wmf_sweet_slider.label.set_position((-0.4, 0.5))
wmf_sweet_slider.valtext.set_text(f'{init_wmf_sweet*100:#.3g} %')
wmf_sweet_slider.valtext.set_ha('right')
wmf_sweet_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Teq
ax_teq_sweet = fig.add_axes([0.08, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_sweet_slider = Slider(
    ax=ax_teq_sweet,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=400.0,
    valmax=1500.0,
    valinit=init_teq_sweet,
    valfmt=' %4.0f K'
)
teq_sweet_slider.label.set_ha('left')
teq_sweet_slider.label.set_position((-0.4, 0.5))
teq_sweet_slider.valtext.set_ha('right')
teq_sweet_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Teq
ax_age_sweet = fig.add_axes([0.08, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
age_sweet_slider = Slider(
    ax=ax_age_sweet,
    label="Age  ",
    valmin=np.log10(0.0010000001),
    valmax=np.log10(19.99999),
    valinit=np.log10(init_age_sweet),
    #valfmt=' %4.0f K'
)
age_sweet_slider.label.set_ha('left')
age_sweet_slider.label.set_position((-0.4, 0.5))
age_sweet_slider.valtext.set_text(f'{init_age_sweet:#.3g} Gyr')
age_sweet_slider.valtext.set_ha('right')
age_sweet_slider.valtext.set_position((1.63, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Top of the Atmosphere
ax_top_sweet = fig.add_axes([0.08, 0.71, 0.15, 0.02])  # [left, bottom, width, height]
top_sweet_slider = Slider(
    ax=ax_top_sweet,
    label="Atm top  ",
    valmin=0,
    valmax=1,
    valinit=init_top_sweet,
    valstep=1
)
top_sweet_slider.label.set_ha('left')
top_sweet_slider.label.set_position((-0.5, 0.5))
top_sweet_slider.label.set_size(9)
top_sweet_slider.valtext.set_ha('right')
top_sweet_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider
top_sweet_slider.valtext.set_text(sweet_labels_top[init_top_sweet])


# Tang et al. 2025 Slider

# Label
fig.text(0.40, 0.95, 'Tang et al. 2025',weight='bold',
        color='red',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Metallicity
ax_met_t25 = fig.add_axes([0.42, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
met_t25_slider = Slider(
    ax=ax_met_t25,
    label='Met  ',
    valmin=1.0,
    valmax=50.0,
    valinit=init_met_t25,
    valfmt=' x%2.0f Solar'
)
#met_t25_slider.label.set_size(9)
met_t25_slider.label.set_ha('left')
met_t25_slider.label.set_position((-0.4, 0.5))
met_t25_slider.valtext.set_size(9)
met_t25_slider.valtext.set_ha('right')
met_t25_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Age
ax_age_t25 = fig.add_axes([0.42, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
age_t25_slider = Slider(
    ax=ax_age_t25,
    label="Age  ",
    valmin=np.log10(0.01),
    valmax=np.log10(10.0),
    valinit=np.log10(init_age_t25),
    #valfmt=' %2.1f'
)
age_t25_slider.label.set_ha('left')
age_t25_slider.label.set_position((-0.4, 0.5))
age_t25_slider.valtext.set_text(f'{init_age_t25:#.3g} Gyr')
age_t25_slider.valtext.set_ha('right')
age_t25_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider


# Make a horizontal oriented slider to control the Teq
ax_teq_t25 = fig.add_axes([0.42, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_t25_slider = Slider(
    ax=ax_teq_t25,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=278.0,
    valmax=1500.0,
    valinit=init_teq_t25,
    valfmt=' %4.0f K'
)
teq_t25_slider.label.set_ha('left')
teq_t25_slider.label.set_position((-0.4, 0.5))
teq_t25_slider.valtext.set_ha('right')
teq_t25_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Envelope fraction
ax_fenv_t25 = fig.add_axes([0.42, 0.75, 0.15, 0.02])  # [left, bottom, width, height]
fenv_t25_slider = Slider(
    ax=ax_fenv_t25,
    label=r"f$_{\mathrm{env}}$  ",
    valmin=np.log10(0.1),
    valmax=np.log10(20.0),
    valinit=np.log10(init_fenv_t25),
    #valfmt=" %2.2f %%"
)
fenv_t25_slider.label.set_ha('left')
fenv_t25_slider.label.set_position((-0.4, 0.5))
fenv_t25_slider.valtext.set_text(f'{init_fenv_t25:#.3g} %')
fenv_t25_slider.valtext.set_ha('right')
fenv_t25_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Top of the Atmosphere
ax_top_t25 = fig.add_axes([0.42, 0.71, 0.15, 0.02])  # [left, bottom, width, height]
top_t25_slider = Slider(
    ax=ax_top_t25,
    label="Atm top  ",
    valmin=0,
    valmax=2,
    valinit=init_top_t25,
    valstep=1
)
top_t25_slider.label.set_ha('left')
top_t25_slider.label.set_position((-0.5, 0.5))
top_t25_slider.label.set_size(9)
top_t25_slider.valtext.set_ha('right')
top_t25_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider
top_t25_slider.valtext.set_text(t25_labels[int(init_top_t25)])

# Zeng+2016 Slider

# Label
fig.text(0.75, 0.95, 'Zeng et al. 2016',weight='bold',
        color='brown',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the CMF
ax_cmf_zeng = fig.add_axes([0.75, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
cmf_zeng_slider = Slider(
    ax=ax_cmf_zeng,
    label='CMF ',
    valmin=0.0,
    valmax=1.0,
    valinit=init_cmf_zeng,
    valfmt=' %1.3f'
)

# The function to be called anytime a slider's value changes
def update(val):
    # Update SWE
    wmf_sweet_linear = 10**wmf_sweet_slider.val
    wmf_sweet_slider.valtext.set_text(f'{wmf_sweet_linear*100:#.3g} %')  # Update displayed value

    age_linear_sweet = 10**age_sweet_slider.val
    age_sweet_slider.valtext.set_text(f'{age_linear_sweet:#.3g} Gyr')  # Update displayed value

    host_sweet_index = int(host_sweet_slider.val)
    host_sweet_slider.valtext.set_text(sweet_labels_host[host_sweet_index])
    top_sweet_index = int(top_sweet_slider.val)
    top_sweet_slider.valtext.set_text(sweet_labels_top[top_sweet_index])
    current_sweet = rad_sweet(x,top_sweet_index,host_sweet_index,wmf_sweet_linear,teq_sweet_slider.val,age_linear_sweet)
    line_sweet.set_ydata(current_sweet)

    line_sweet_mbar.set_ydata(rad_sweet(x,0,host_sweet_index,wmf_sweet_linear,teq_sweet_slider.val,age_linear_sweet))
    line_sweet_mibar.set_ydata(rad_sweet(x,1,host_sweet_index,wmf_sweet_linear,teq_sweet_slider.val,age_linear_sweet))

    # Update T25
    age_t25_linear = 10**age_t25_slider.val
    age_t25_slider.valtext.set_text(f'{age_t25_linear:#.3g} Gyr')  # Update displayed value

    fenv_t25_linear = 10**fenv_t25_slider.val
    fenv_t25_slider.valtext.set_text(f'{fenv_t25_linear:#.3g} %')  # Update displayed value

    func_t25_index = int(top_t25_slider.val)
    top_t25_slider.valtext.set_text(t25_labels[func_t25_index])
    current_t25 = rad_t25(x,func_t25_index,met_t25_slider.val,age_t25_linear,teq_t25_slider.val,fenv_t25_linear,tolerance=t25_tolerance_main)
    line_t25.set_ydata(current_t25)

    line_t25_rcb.set_ydata(rad_t25(x,0,met_t25_slider.val,age_t25_linear,teq_t25_slider.val,fenv_t25_linear))
    line_t25_mbar.set_ydata(rad_t25(x,1,met_t25_slider.val,age_t25_linear,teq_t25_slider.val,fenv_t25_linear))
    line_t25_nbar.set_ydata(rad_t25(x,2,met_t25_slider.val,age_t25_linear,teq_t25_slider.val,fenv_t25_linear))

    # Update Zeng
    line_zeng.set_ydata(rad_zeng(x, cmf_zeng_slider.val))


# register the update function with each slider
host_sweet_slider.on_changed(update)
wmf_sweet_slider.on_changed(update)
teq_sweet_slider.on_changed(update)
age_sweet_slider.on_changed(update)
top_sweet_slider.on_changed(update)

met_t25_slider.on_changed(update)
age_t25_slider.on_changed(update)
teq_t25_slider.on_changed(update)
fenv_t25_slider.on_changed(update)
top_t25_slider.on_changed(update)

cmf_zeng_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.84, 0.025, 0.1, 0.04])  # [left, bottom, width, height]
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    host_sweet_slider.reset()
    wmf_sweet_slider.reset()
    teq_sweet_slider.reset()
    age_sweet_slider.reset()
    top_sweet_slider.reset()

    met_t25_slider.reset()
    age_t25_slider.reset()
    teq_t25_slider.reset()
    fenv_t25_slider.reset()
    top_t25_slider.reset()

    cmf_zeng_slider.reset()

button.on_clicked(reset)

plt.show()
