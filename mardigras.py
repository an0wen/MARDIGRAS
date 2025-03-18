##############################################
#
#   Mass-Radius DIaGRAm with Sliders (MARDIGRAS)
#
##############################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import patheffects
import mplcursors

from scipy.interpolate import RegularGridInterpolator

import requests
import os
from datetime import datetime
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="Run the mardigras tool with optional features.")

# Flag to update the catalog
parser.add_argument(
    "--update-nea-catalog",
    action="store_true",
    help="Update the NASA Exoplanet Archive catalog before starting the tool.",
)

# Flag to choose another type of catalog
parser.add_argument(
    "--catalog",
    choices=["NEA", "PlanetS"],
    default="NEA",
    help="Choose the exoplanet catalog to use. Default is NEA."
)

args = parser.parse_args()

#Paths to models
path_models = "./models/"

path_aguichine = path_models + "Aguichine2021_fit_coefficients_2024.dat"
path_zeng = path_models + "Zeng2016.dat"


# Load curves from Zeng et al. 2016
zeng_mass,zeng_purefe,zeng_rock,zeng_50wat,zeng_100wat,zeng_earth = np.loadtxt(path_zeng,delimiter="\t",skiprows=1,unpack=True)


##############################################
#
#   MAKE SWE INTERPOLATOR (Aguichine+2024)
#
##############################################

path_swe = path_models + "A25_SWE_all.dat"

swe_top = np.array([0,1])
swe_labels_top = ["20 mbar", "1 µbar"]
swe_labels_host = ["Type M", "Type G"]
# swe_teqs = [400, 500, 700, 900, 1100, 1300, 1500]
swe_teqs = [500, 600, 700]
swe_wmfs = [0.1,1,10,20,30,40,50,60,70,80,90,100]  # 12 water mass fractions from 0.05 to 0.60
swe_masses = [0.2       ,  0.254855  ,  0.32475535,  0.41382762,  0.52733018,
        0.67196366,  0.85626648,  1.09111896,  1.39038559,  1.77173358,
        2.25767578,  2.87689978,  3.66596142,  4.67144294,  5.95270288,
        7.58538038,  9.66586048, 12.31696422, 15.69519941, 20.        ]  # 20 points in mass from 0.1 to 2.0

swe_ages = np.array([0.001,0.0015,0.002,0.003,0.005,0.01,
                        0.02,0.03,0.05,
                        0.1,0.2,0.5,
                        1.0,2.0,5.0,
                        10,20])

listrpfull1 = np.loadtxt(path_swe,skiprows=36,unpack=True,usecols=(6))
listrpfull2 = np.loadtxt(path_swe,skiprows=36,unpack=True,usecols=(7))
listrpfull = np.array((listrpfull1,listrpfull2))
listrpfull = np.delete(listrpfull, np.arange(17, listrpfull.size, 18))

# mask = listrpfull == -1.0
# listrpfull[mask] == np.inf

# listrpfull_m = listrpfull[0:int(len(listrpfull)/2)]
# listrpfull_g = listrpfull[int(len(listrpfull)/2):]

# Make SWE interpolator

swe_dim_wmf = np.array(swe_wmfs)/100
swe_dim_teq = np.array(swe_teqs)
swe_dim_mass = np.array(swe_masses)
swe_dim_age = swe_ages
swe_dim_star = np.array([0,1])
swe_dim_top = np.array([0,1])

swe_data_radius = np.reshape(listrpfull,(2,2,12,3,20,17))
# swe_data_radius_g = np.reshape(listrpfull_g,(12,7,20,17))

fill_value = np.nan
interp_swe = RegularGridInterpolator((swe_dim_top,
                                      swe_dim_star,
                                      swe_dim_wmf,
                                      swe_dim_teq,
                                      swe_dim_mass,
                                      swe_dim_age), 
                                     swe_data_radius, method='slinear', bounds_error=False, fill_value=fill_value)
# interp_swe_g = RegularGridInterpolator((swe_dim_wmf, swe_dim_teq, swe_dim_mass, swe_dim_age), swe_data_radius_g, method='slinear', bounds_error=False, fill_value=fill_value)

##############################################
#
#   MAKE ZENG INTERPOLATOR (Zeng+2016)
#
##############################################

# open files
list_zeng_fe_m,list_zeng_fe_r = np.loadtxt(path_models+"zeng2016-iron.dat",unpack=True,usecols=(0,1))
list_zeng_ea_m,list_zeng_ea_r = np.loadtxt(path_models+"zeng2016-earth.dat",unpack=True,usecols=(0,1))
list_zeng_mg_m,list_zeng_mg_r = np.loadtxt(path_models+"zeng2016-rock.dat",unpack=True,usecols=(0,1))

list_zeng_masses = np.logspace(np.log10(0.01), np.log10(100.0), 30)

# reshape radius data to the same vector
list_zeng_fe_radii = np.interp(list_zeng_masses, list_zeng_fe_m, list_zeng_fe_r)
list_zeng_ea_radii = np.interp(list_zeng_masses, list_zeng_ea_m, list_zeng_ea_r)
list_zeng_mg_radii = np.interp(list_zeng_masses, list_zeng_mg_m, list_zeng_mg_r)

# create interpolator
dimcmf_zeng = np.array([0.0,0.325,1.0])
data_zeng = np.vstack((list_zeng_mg_radii, list_zeng_ea_radii,list_zeng_fe_radii)).T
interp_zeng = RegularGridInterpolator((list_zeng_masses,dimcmf_zeng), data_zeng, method='slinear', bounds_error=False, fill_value=None)


##############################################
#
#   MAKE GAS DWARF INTERPOLATOR (Tang+2024)
#
##############################################

dim_met_t24 = np.array([1.0,50.0])
dim_age_t24 = np.log10(np.array([0.1,1.0,10.0]))
dim_finc_t24= np.array([1.0,10.0,100.0,1000.0])
dim_teq_t24 = 278.0*(dim_finc_t24)**(0.25)
dim_mass_t24= np.array([1,2,3,4,5,6,8,10,13,16,20])
dim_fenv_t24= np.array([0.10,0.20,0.50,1,2,5,10,20])
dim_top_t24 = np.array([0,1,2])
t24_labels = ["RCB", "20 mbar", "1 nbar"]

t24_data_radius = np.zeros((3,2,3,4,11,8))

data0 = np.genfromtxt(path_models+"Tang2024.dat",filling_values=fill_value,comments='#',skip_header=1,usecols=(5,6,7,8))
t24_data_radius[0,:,:,:,:,:] = data0[:,0].reshape(2,3,4,11,8)
t24_data_radius[1,:,:,:,:,:] = data0[:,1].reshape(2,3,4,11,8)
t24_data_radius[2,:,:,:,:,:] = data0[:,2].reshape(2,3,4,11,8)

interp_t24 = RegularGridInterpolator((dim_top_t24,dim_met_t24, dim_age_t24, dim_teq_t24, dim_mass_t24, dim_fenv_t24), t24_data_radius, method='linear', bounds_error=False, fill_value=fill_value)

# Make the boil-off limit
t24_bolim_fenv = np.zeros((2,8,11))
t24_bolim_radius = np.zeros((3,2,8,11))
data3 = np.genfromtxt(path_models+"T24_grids/boil-off.csv",delimiter=",",filling_values=20.0,comments='#',skip_header=0)
data3 = data3[:,:12]
t24_bolim_fenv = data3[:,1:].reshape(2,8,11)
t24_bolim_fenv = np.clip(t24_bolim_fenv,a_min=0.0,a_max=20)

dim_finc_t24_bolim= np.array([1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0])
dim_teq_t24_bolim = 278.0*(dim_finc_t24_bolim)**(0.25)

interp_t24_bolim_maxf = RegularGridInterpolator((dim_met_t24, dim_teq_t24_bolim, dim_mass_t24), t24_bolim_fenv, method='linear', bounds_error=False, fill_value=np.inf)

##############################################
#
#   LOAD DATA
#
##############################################

# Exoplanet catalog

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
                if line.startswith("# Catalog last updated:"):
                    print(line.strip())
                    break
    except Exception as e:
        print(f"Error reading the catalog: {e}")

# File paths and URL of the NEA Catalog
nea_catalog_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
               "query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,"
               "pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+"
               "default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+"
               "and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv")
nea_output_file = "./data/catalog_exoplanets.dat"

if args.update_nea_catalog:
    # Update the catalog
    update_nea_exoplanet_catalog(nea_catalog_url, nea_output_file)
else:
    # Check for existing catalog and print the last update date
    read_nea_last_update(nea_output_file)

list_catalog_rp,list_catalog_rpe1,list_catalog_rpe2,list_catalog_mp,list_catalog_mpe1,list_catalog_mpe2 \
    = np.genfromtxt(nea_output_file,delimiter="\t",unpack=True,usecols=(1,2,3,4,5,6),filling_values=0.0)

# Load the exoplanet names
list_catalog_names = np.genfromtxt(
    nea_output_file, delimiter="\t", dtype=str, usecols=0
)

# This procedure removes planets that don't have radius and/or mass measurements
# the goal is to have arrays of smaller size, so that rendering is faster when sliders are used
list_exo_rp = []
list_exo_rpe1 = []
list_exo_rpe2 = []
list_exo_mp = []
list_exo_mpe1 = []
list_exo_mpe2 = []
list_exo_names = []
for i in range(len(list_catalog_rp)):
    exo_mass_prec = 0.5 # minimum precision on exoplanet mass
    if list_catalog_rp[i]!=0.0 and list_catalog_mp[i]!=0.0 \
        and (abs(list_catalog_mpe1[i]))/list_catalog_mp[i] < exo_mass_prec \
        and (abs(list_catalog_mpe2[i]))/list_catalog_mp[i] < exo_mass_prec:
        list_exo_rp = np.append(list_exo_rp,[list_catalog_rp[i]])
        list_exo_rpe1 = np.append(list_exo_rpe1,[list_catalog_rpe1[i]])
        list_exo_rpe2 = np.append(list_exo_rpe2,[list_catalog_rpe2[i]])
        list_exo_mp = np.append(list_exo_mp,[list_catalog_mp[i]])
        list_exo_mpe1 = np.append(list_exo_mpe1,[list_catalog_mpe1[i]])
        list_exo_mpe2 = np.append(list_exo_mpe2,[list_catalog_mpe2[i]])
        list_exo_names = np.append(list_exo_names,[list_catalog_names[i]])

# Targets catalog
# The intended use is to showcase a few targets (dedicated study, new discovery, update of parameters, etc.)
# The catalog of targets must have the same formatting as the exoplanet catalog.
list_targets_rp,list_targets_rpe1,list_targets_rpe2,list_targets_mp,list_targets_mpe1,list_targets_mpe2\
    = np.genfromtxt("./data/catalog_targets.dat",delimiter="\t",unpack=True,usecols=(1,2,3,4,5,6),filling_values=0.0)

import csv
file_path = "./data/catalog_targets.dat"
list_targets_names = []
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        if row[0][0]!='#': list_targets_names.append(row[0])

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
def rad_swe(x,top_swe,star_swe,wmf_swe,teq_swe,age_swe):
    input0 = np.stack((np.full(len(x),top_swe),
                       np.full(len(x),star_swe),
                       np.full(len(x),wmf_swe),
                       np.full(len(x),teq_swe),
                       x,
                       np.full(len(x),age_swe)), axis=-1)
    return interp_swe(input0)

# def rad_swe_g(x,wmf_swe,teq_swe,age_swe):
#     input0 = np.stack((np.full(len(x),wmf_swe),np.full(len(x),teq_swe),x,np.full(len(x),age_swe)), axis=-1)
#     return interp_swe_g(input0)

# swe_func = [rad_swe_m, rad_swe_g]
# swe_labels = ["Type M", "Type G"]  # Optional labels for each 

t24_tolerance_main = 1.0
t24_tolerance_sec = 0.0
def rad_t24(x,top,met,age,teq,fenv,tolerance=t24_tolerance_sec):
    logage = np.log10(age)
    input0 = np.stack((np.full(len(x),top),
                       np.full(len(x),met),
                       np.full(len(x),logage),
                       np.full(len(x),teq),
                       x,
                       np.full(len(x),fenv)), axis=-1)
    rp = interp_t24(input0)
    input1 = np.stack((np.full(len(x),met),np.full(len(x),teq),x), axis=-1)
    boiloff_maxf = interp_t24_bolim_maxf(input1)
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
init_host_swe = 0
init_top_swe = 1
init_wmf_swe = 0.5
init_teq_swe = 600.0
init_age_swe = 1.0


init_met_t24 = 1.0
init_age_t24 = 1.0
init_teq_t24 = 600.0
init_fenv_t24 = 1.0
init_top_t24 = 1.0

init_cmf_zeng = 0.325

# Create the figure
fig = plt.figure(figsize=(7,7))

# make axes for main figure
ax = fig.add_axes([0.11, 0.1, 0.83, 0.6])  # [left, bottom, width, height]

# Figure options
ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('Mass [Me]')
ax.set_ylabel('Radius [Re]')
ax.grid(visible=True,which='major', axis='both')

# Sliding lines
current_swe = rad_swe(x,init_top_swe,init_host_swe,init_wmf_swe,init_teq_swe,init_age_swe)
line_swe, = ax.plot(x, current_swe,lw=2,color='blue',ls='-',zorder=20)
current_t24 = rad_t24(x,init_top_t24,init_met_t24,init_age_t24,init_teq_t24,init_fenv_t24,tolerance=t24_tolerance_main)
line_t24, = ax.plot(x, current_t24,lw=2,color='red',zorder=30)
line_zeng, = ax.plot(x, rad_zeng(x,init_cmf_zeng),lw=2,color='brown',zorder=10)

# Minor lines
line_swe_mbar, = ax.plot(x, rad_swe(x,0,init_host_swe,init_wmf_swe,init_teq_swe,init_age_swe),lw=0.5,color='blue',ls='--',zorder=19)
line_swe_mibar, = ax.plot(x, rad_swe(x,1,init_host_swe,init_wmf_swe,init_teq_swe,init_age_swe),lw=0.5,color='blue',ls='--',zorder=19)
line_t24_rcb, = ax.plot(x, rad_t24(x,0,init_met_t24,init_age_t24,init_teq_t24,init_fenv_t24),lw=0.5,ls='--',color='red',zorder=29)
line_t24_mbar, = ax.plot(x, rad_t24(x,1,init_met_t24,init_age_t24,init_teq_t24,init_fenv_t24),lw=0.5,ls='--',color='red',zorder=29)
line_t24_nbar, = ax.plot(x, rad_t24(x,2,init_met_t24,init_age_t24,init_teq_t24,init_fenv_t24),lw=0.5,ls='--',color='red',zorder=29)


# Planets
list_exo_mpe = [abs(list_exo_mpe2), list_exo_mpe1]
list_exo_rpe = [abs(list_exo_rpe2), list_exo_rpe1]
list_targets_mpe = [abs(list_targets_mpe2), list_targets_mpe1]
list_targets_rpe = [abs(list_targets_rpe2), list_targets_rpe1]

# Exoplanets from the catalog file
catalog_points = ax.errorbar(list_exo_mp,list_exo_rp,
            yerr=list_exo_rpe,
            xerr=list_exo_mpe,
            fmt='o',zorder=-30,
            c="black",alpha=0.2)

# Add hover annotations
annot = ax.annotate(
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
    text = f"{list_exo_names[ind['ind'][0]]}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    """Event handler for mouse motion."""
    vis = annot.get_visible()
    if event.inaxes == ax:
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
ax.errorbar(list_targets_mp,list_targets_rp,
            yerr=list_targets_rpe,
            xerr=list_targets_mpe,
            ls='',c='orange',elinewidth=3,
            marker='*',mfc='orange',mec='black', ms=15, mew=1,
            zorder=50)

# Annotate each point
texts = []
# Use these to expand annotations horizontally and/or vertically
text_expand_h = False
text_expand_v = True
for i, label in enumerate(list_targets_names):
    x_text = 10
    y_text = -15
    if text_expand_h: x_text = ((list_targets_mp[i]-np.median(list_targets_mp))/(max(list_targets_mp)-min(list_targets_mp)))*15
    if text_expand_v: y_text = ((list_targets_rp[i]-np.median(list_targets_rp))/(max(list_targets_rp)-min(list_targets_rp)))*15
    texts.append(ax.annotate(
    label, 
    (list_targets_mp[i], list_targets_rp[i]), 
    textcoords="offset points", 
    xytext=(x_text,y_text), 
    ha='left',
    fontsize='x-small',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7),zorder=50+i
    ))
    #plt.annotate(label, (list_targets_mp[i], list_targets_rp[i]), textcoords="offset points", xytext=(5,-10),
    #ha='left',bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7),zorder=50+i)

# Solar system planets
for i in range(len(ss_planets)):
    text = ax.text(ss_masses[i], ss_radii[i], ss_alchemy_symbols[ss_planets[i]], \
            fontsize=15, ha='center', va='center', color='red')
    text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])  # Add outline to text

# Zeng fixed
zeng_width=1.0
ax.plot(list_zeng_masses,list_zeng_fe_radii,linewidth=zeng_width,color='black',zorder=5)
ax.plot(list_zeng_masses,list_zeng_ea_radii,linewidth=zeng_width,color='brown',zorder=6)
ax.plot(list_zeng_masses,list_zeng_mg_radii,linewidth=zeng_width,color='grey',zorder=7)
ax.plot(zeng_mass,zeng_50wat,linewidth=zeng_width,color='cyan',zorder=8)
ax.plot(zeng_mass,zeng_100wat,linewidth=zeng_width,color='cyan',zorder=9)

# Zeng fixed labels
ax.text(0.55, 0.60, '100% Core',fontsize=5,
        color='black',rotation=10)
        
ax.text(0.55, 0.76, 'Earth-like',fontsize=5,
        color='brown',rotation=10)

ax.text(0.55, 0.93, '100% Mantle',fontsize=5,
        color='grey',rotation=15)

ax.text(0.55, 1.10, '50% Liquid H2O',fontsize=5,
        color='blue',rotation=17)

ax.text(0.55, 1.23, '100% Liquid H2O',fontsize=5,
        color='blue',rotation=17)

##############################################
#
#   MAKE SLIDERS
#
##############################################

# Aguichine+2024 Slider

# Label
fig.text(0.05, 0.95, 'Aguichine et al. 2025',weight='bold',
        color='blue',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Host Star
ax_host_swe = fig.add_axes([0.08, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
host_swe_slider = Slider(
    ax=ax_host_swe,
    label='Star  ',
    valmin=0,
    valmax=1,
    valinit=init_host_swe,
    valstep=1
)
host_swe_slider.label.set_ha('left')
host_swe_slider.label.set_position((-0.4, 0.5))
host_swe_slider.valtext.set_ha('right')
host_swe_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider
host_swe_slider.valtext.set_text(swe_labels_host[init_host_swe])

# Make a horizontal oriented slider to control the WMF
ax_wmf_swe = fig.add_axes([0.08, 0.75, 0.15, 0.02])  # [left, bottom, width, height]
wmf_swe_slider = Slider(
    ax=ax_wmf_swe,
    label="WMF  ",
    valmin=np.log10(0.001),
    valmax=np.log10(1.0),
    valinit=np.log10(init_wmf_swe),
    #valfmt=' %1.3f'
)
wmf_swe_slider.label.set_ha('left')
wmf_swe_slider.label.set_position((-0.4, 0.5))
wmf_swe_slider.valtext.set_text(f'{init_wmf_swe*100:#.3g} %')
wmf_swe_slider.valtext.set_ha('right')
wmf_swe_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Teq
ax_teq_swe = fig.add_axes([0.08, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_swe_slider = Slider(
    ax=ax_teq_swe,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=500.0,
    valmax=700.0,
    valinit=init_teq_swe,
    valfmt=' %4.0f K'
)
teq_swe_slider.label.set_ha('left')
teq_swe_slider.label.set_position((-0.4, 0.5))
teq_swe_slider.valtext.set_ha('right')
teq_swe_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Teq
ax_age_swe = fig.add_axes([0.08, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
age_swe_slider = Slider(
    ax=ax_age_swe,
    label="Age  ",
    valmin=np.log10(0.0010000001),
    valmax=np.log10(19.99999),
    valinit=np.log10(init_age_swe),
    #valfmt=' %4.0f K'
)
age_swe_slider.label.set_ha('left')
age_swe_slider.label.set_position((-0.4, 0.5))
age_swe_slider.valtext.set_text(f'{init_age_swe:#.3g} Gyr')
age_swe_slider.valtext.set_ha('right')
age_swe_slider.valtext.set_position((1.63, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Top of the Atmosphere
ax_top_swe = fig.add_axes([0.08, 0.71, 0.15, 0.02])  # [left, bottom, width, height]
top_swe_slider = Slider(
    ax=ax_top_swe,
    label="Atm top  ",
    valmin=0,
    valmax=1,
    valinit=init_top_swe,
    valstep=1
)
top_swe_slider.label.set_ha('left')
top_swe_slider.label.set_position((-0.5, 0.5))
top_swe_slider.label.set_size(9)
top_swe_slider.valtext.set_ha('right')
top_swe_slider.valtext.set_position((1.53, 0.5))  # Shift text to the right outside the slider
top_swe_slider.valtext.set_text(swe_labels_top[init_top_swe])


# Tang et al. 2024 Slider

# Label
fig.text(0.40, 0.95, 'Tang et al. 2024',weight='bold',
        color='red',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Metallicity
ax_met_t24 = fig.add_axes([0.42, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
met_t24_slider = Slider(
    ax=ax_met_t24,
    label='Met  ',
    valmin=1.0,
    valmax=50.0,
    valinit=init_met_t24,
    valfmt=' x%2.0f Solar'
)
#met_t24_slider.label.set_size(9)
met_t24_slider.label.set_ha('left')
met_t24_slider.label.set_position((-0.4, 0.5))
met_t24_slider.valtext.set_size(9)
met_t24_slider.valtext.set_ha('right')
met_t24_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Age
ax_age_t24 = fig.add_axes([0.42, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
age_t24_slider = Slider(
    ax=ax_age_t24,
    label="Age  ",
    valmin=np.log10(0.1),
    valmax=np.log10(10.0),
    valinit=np.log10(init_age_t24),
    #valfmt=' %2.1f'
)
age_t24_slider.label.set_ha('left')
age_t24_slider.label.set_position((-0.4, 0.5))
age_t24_slider.valtext.set_text(f'{init_age_t24:#.3g} Gyr')
age_t24_slider.valtext.set_ha('right')
age_t24_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider


# Make a horizontal oriented slider to control the Teq
ax_teq_t24 = fig.add_axes([0.42, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_t24_slider = Slider(
    ax=ax_teq_t24,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=278.0,
    valmax=1500.0,
    valinit=init_teq_t24,
    valfmt=' %4.0f K'
)
teq_t24_slider.label.set_ha('left')
teq_t24_slider.label.set_position((-0.4, 0.5))
teq_t24_slider.valtext.set_ha('right')
teq_t24_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Envelope fraction
ax_fenv_t24 = fig.add_axes([0.42, 0.75, 0.15, 0.02])  # [left, bottom, width, height]
fenv_t24_slider = Slider(
    ax=ax_fenv_t24,
    label=r"f$_{\mathrm{env}}$  ",
    valmin=np.log10(0.1),
    valmax=np.log10(20.0),
    valinit=np.log10(init_fenv_t24),
    #valfmt=" %2.2f %%"
)
fenv_t24_slider.label.set_ha('left')
fenv_t24_slider.label.set_position((-0.4, 0.5))
fenv_t24_slider.valtext.set_text(f'{init_fenv_t24:#.3g} %')
fenv_t24_slider.valtext.set_ha('right')
fenv_t24_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Top of the Atmosphere
ax_top_t24 = fig.add_axes([0.42, 0.71, 0.15, 0.02])  # [left, bottom, width, height]
top_t24_slider = Slider(
    ax=ax_top_t24,
    label="Atm top  ",
    valmin=0,
    valmax=2,
    valinit=init_top_t24,
    valstep=1
)
top_t24_slider.label.set_ha('left')
top_t24_slider.label.set_position((-0.5, 0.5))
top_t24_slider.label.set_size(9)
top_t24_slider.valtext.set_ha('right')
top_t24_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

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
    wmf_swe_linear = 10**wmf_swe_slider.val
    wmf_swe_slider.valtext.set_text(f'{wmf_swe_linear*100:#.3g} %')  # Update displayed value

    age_linear_swe = 10**age_swe_slider.val
    age_swe_slider.valtext.set_text(f'{age_linear_swe:#.3g} Gyr')  # Update displayed value

    host_swe_index = int(host_swe_slider.val)
    host_swe_slider.valtext.set_text(swe_labels_host[host_swe_index])
    top_swe_index = int(top_swe_slider.val)
    top_swe_slider.valtext.set_text(swe_labels_top[top_swe_index])
    current_swe = rad_swe(x,top_swe_index,host_swe_index,wmf_swe_linear,teq_swe_slider.val,age_linear_swe)
    line_swe.set_ydata(current_swe)

    line_swe_mbar.set_ydata(rad_swe(x,0,host_swe_index,wmf_swe_linear,teq_swe_slider.val,age_linear_swe))
    line_swe_mibar.set_ydata(rad_swe(x,1,host_swe_index,wmf_swe_linear,teq_swe_slider.val,age_linear_swe))

    # Update T24
    age_t24_linear = 10**age_t24_slider.val
    age_t24_slider.valtext.set_text(f'{age_t24_linear:#.3g} Gyr')  # Update displayed value

    fenv_t24_linear = 10**fenv_t24_slider.val
    fenv_t24_slider.valtext.set_text(f'{fenv_t24_linear:#.3g} %')  # Update displayed value

    func_t24_index = int(top_t24_slider.val)
    top_t24_slider.valtext.set_text(t24_labels[func_t24_index])
    current_t24 = rad_t24(x,func_t24_index,met_t24_slider.val,age_t24_linear,teq_t24_slider.val,fenv_t24_linear,tolerance=t24_tolerance_main)
    line_t24.set_ydata(current_t24)

    line_t24_rcb.set_ydata(rad_t24(x,0,met_t24_slider.val,age_t24_linear,teq_t24_slider.val,fenv_t24_linear))
    line_t24_mbar.set_ydata(rad_t24(x,1,met_t24_slider.val,age_t24_linear,teq_t24_slider.val,fenv_t24_linear))
    line_t24_nbar.set_ydata(rad_t24(x,2,met_t24_slider.val,age_t24_linear,teq_t24_slider.val,fenv_t24_linear))

    # Update Zeng
    line_zeng.set_ydata(rad_zeng(x, cmf_zeng_slider.val))


# register the update function with each slider
host_swe_slider.on_changed(update)
wmf_swe_slider.on_changed(update)
teq_swe_slider.on_changed(update)
age_swe_slider.on_changed(update)
top_swe_slider.on_changed(update)

met_t24_slider.on_changed(update)
age_t24_slider.on_changed(update)
teq_t24_slider.on_changed(update)
fenv_t24_slider.on_changed(update)
top_t24_slider.on_changed(update)

cmf_zeng_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.84, 0.025, 0.1, 0.04])  # [left, bottom, width, height]
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    host_swe_slider.reset()
    wmf_swe_slider.reset()
    teq_swe_slider.reset()
    age_swe_slider.reset()
    top_swe_slider.reset()

    met_t24_slider.reset()
    age_t24_slider.reset()
    teq_t24_slider.reset()
    fenv_t24_slider.reset()
    top_t24_slider.reset()

    cmf_zeng_slider.reset()

button.on_clicked(reset)

plt.show()
