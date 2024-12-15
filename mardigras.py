##############################################
#
#   READ FIT COEFFICIENTS
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


# Load fit coefficients from Aguichine et al. 2021
listcmf,listwmf,listteq,lista,listb,listd,listc,listmasslow,listmasshigh \
    = np.loadtxt(path_aguichine,skiprows=19,unpack=True,usecols=(0,1,2,3,4,5,6,11,12))
        
def radius(mass,a,b,c,d):
    return 10**(a*np.log10(mass) + np.exp(-d*(np.log10(mass)+c)) + b)



# Load curves from Zeng et al. 2016
zeng_mass,zeng_purefe,zeng_rock,zeng_50wat,zeng_100wat,zeng_earth = np.loadtxt(path_zeng,delimiter="\t",skiprows=1,unpack=True)


##############################################
#
#   MAKE IOP INTERPOLATOR
#
##############################################

dimcmf = np.linspace(0.0,0.9,10)
dimwmf = np.linspace(0.1,1.0,10)
dimteq = np.linspace(400.0,1300.0,10)

data_a = np.reshape(lista,(10,10,10))
data_b = np.reshape(listb,(10,10,10))
data_c = np.reshape(listc,(10,10,10))
data_d = np.reshape(listd,(10,10,10))

data_masslimlow = np.reshape(listmasslow,(10,10,10))
data_masslimhigh = np.reshape(listmasshigh,(10,10,10))

interp_a = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_a, method='cubic', bounds_error=False, fill_value=None)
interp_b = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_b, method='cubic', bounds_error=False, fill_value=None)
interp_c = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_c, method='cubic', bounds_error=False, fill_value=None)
interp_d = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_d, method='cubic', bounds_error=False, fill_value=None)

interp_masslimlow = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_masslimlow, method='cubic', bounds_error=False, fill_value=None)
interp_masslimhigh = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_masslimhigh, method='cubic', bounds_error=False, fill_value=None)


##############################################
#
#   MAKE ZENG INTERPOLATOR
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
interp_zeng = RegularGridInterpolator((list_zeng_masses,dimcmf_zeng), data_zeng, method='linear', bounds_error=False, fill_value=None)



##############################################
#
#   MAKE LOPEZ-FORTNEY INTERPOLATOR
#
##############################################

# open files
list_met_lf14,list_age_lf14,list_finc_lf14,list_m_lf14,list_fenv_lf14,list_r_lf14 = np.loadtxt(path_models+"LF2014.dat",skiprows=11,unpack=True,usecols=(0,1,2,3,4,5))



dim_met_lf14 = np.array([1.0,50.0])
dim_age_lf14 = np.array([0.1,1.0,10.0])
dim_finc_lf14= np.array([0.1,10.0,1000.0])
dim_teq_lf14 = 278.0*(dim_finc_lf14)**(0.25)
dim_mass_lf14= np.array([1,1.5,2.4,3.6,5.5,8.5,13,20])
dim_fenv_lf14= np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0])

data_r_lf14 = np.reshape(list_r_lf14,(2,3,3,8,11))

interp_lf14 = RegularGridInterpolator((dim_met_lf14, dim_age_lf14, dim_teq_lf14, dim_mass_lf14, dim_fenv_lf14), data_r_lf14, method='linear', bounds_error=False, fill_value=np.inf)


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
def rad_iop(x, cmf_iop,wmf_iop,teq_iop):
    ## Get parameters and validity range or IOP model
    a,b,c,d = interp_a([cmf_iop,teq_iop,wmf_iop]),interp_b([cmf_iop,teq_iop,wmf_iop]),interp_c([cmf_iop,teq_iop,wmf_iop]),interp_d([cmf_iop,teq_iop,wmf_iop])
    return 10**(a*np.log10(x) + np.exp(-d*(np.log10(x)+c)) + b)

def rad_iop_lim(x, cmf_iop,wmf_iop,teq_iop):
    ## Get parameters and validity range or IOP model
    a,b,c,d = interp_a([cmf_iop,teq_iop,wmf_iop]),interp_b([cmf_iop,teq_iop,wmf_iop]),interp_c([cmf_iop,teq_iop,wmf_iop]),interp_d([cmf_iop,teq_iop,wmf_iop])
    masslimlow,masslimhigh=interp_masslimlow([cmf_iop,teq_iop,wmf_iop]),interp_masslimhigh([cmf_iop,teq_iop,wmf_iop])
    if cmf_iop < 0.0 or cmf_iop > 0.9 or wmf_iop < 0.1 or wmf_iop > 1.0 or teq_iop < 400.0 or teq_iop > 1300.0:
        masslimlow,masslimhigh = 1.0,1.0

    # if a.all(x) < masslimlow or a.all(x) > masslimhigh:
    #     return np.inf
    o1=np.where(x<masslimlow,np.inf,10**(a*np.log10(x) + np.exp(-d*(np.log10(x)+c)) + b))
    o2=np.where(x>masslimhigh,np.inf,10**(a*np.log10(x) + np.exp(-d*(np.log10(x)+c)) + b))

    return (o1+o2)*0.5

def rad_lf(x,met,age,teq,fenv):
    input0 = np.stack((np.full(len(x),met),np.full(len(x),age),np.full(len(x),teq),x,np.full(len(x),fenv)), axis=-1)
    rp = interp_lf14(input0)
    return rp

def rad_zeng(x,cmf):
    input0 = np.stack((x,np.full(len(x),cmf)), axis=-1)
    rp = interp_zeng(input0)
    return rp



## Plot parameters
xmin, xmax, nx = 0.5, 30.0, 50
ymin, ymax     = 0.5, 4.5

x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

# Define initial parameters
init_cmf_iop = 0.3
init_wmf_iop = 0.5
init_teq_iop = 700.0

init_met_lf = 1.0
init_age_lf = 1.0
init_teq_lf = 275.0
init_fenv_lf = 1.0

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
line_iop, = ax.plot(x, rad_iop(x, init_cmf_iop,init_wmf_iop,init_teq_iop),lw=2,color='blue',ls='--',zorder=20)
line_iop_lim, = ax.plot(x, rad_iop_lim(x, init_cmf_iop,init_wmf_iop,init_teq_iop),lw=2,color='blue',zorder=25)
line_lf, = ax.plot(x, rad_lf(x,init_met_lf,init_age_lf,init_teq_lf,init_fenv_lf),lw=2,color='red',zorder=30)
line_zeng, = ax.plot(x, rad_zeng(x,init_cmf_zeng),lw=2,color='brown',zorder=10)

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

# # Add interactive cursors for annotations
# cursor = mplcursors.cursor(catalog_points, hover=True)

# # Define the annotation behavior
# @cursor.connect("add")
# def on_add(sel):
#     index = sel.index
#     sel.annotation.set_text(list_exo_names[index])
#     sel.annotation.set_fontsize(10)
#     sel.annotation.get_bbox_patch().set_alpha(0.8)

# Add hover annotations
annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
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

# Aguichine+2021 Slider

# Label
fig.text(0.05, 0.95, 'Aguichine et al. 2021',weight='bold',
        color='blue',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the CMF
ax_cmf_iop = fig.add_axes([0.08, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
cmf_iop_slider = Slider(
    ax=ax_cmf_iop,
    label='CMF  ',
    valmin=0.0,
    valmax=0.9,
    valinit=init_cmf_iop,
    valfmt=' %1.3f'
)
cmf_iop_slider.label.set_ha('left')
cmf_iop_slider.label.set_position((-0.4, 0.5))
cmf_iop_slider.valtext.set_ha('right')
cmf_iop_slider.valtext.set_position((1.55, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the WMF
ax_wmf_iop = fig.add_axes([0.08, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
wmf_iop_slider = Slider(
    ax=ax_wmf_iop,
    label="WMF  ",
    valmin=0.1,
    valmax=1.0,
    valinit=init_wmf_iop,
    valfmt=' %1.3f'
)
wmf_iop_slider.label.set_ha('left')
wmf_iop_slider.label.set_position((-0.4, 0.5))
wmf_iop_slider.valtext.set_ha('right')
wmf_iop_slider.valtext.set_position((1.55, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Teq
ax_teq_iop = fig.add_axes([0.08, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_iop_slider = Slider(
    ax=ax_teq_iop,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=400.0,
    valmax=1300.0,
    valinit=init_teq_iop,
    valfmt=' %4.0f K'
)
teq_iop_slider.label.set_ha('left')
teq_iop_slider.label.set_position((-0.4, 0.5))
teq_iop_slider.valtext.set_ha('right')
teq_iop_slider.valtext.set_position((1.55, 0.5))  # Shift text to the right outside the slider

# Lopez & Fortney 2014 Slider

# Label
fig.text(0.38, 0.95, 'Lopez & Fortney 2014',weight='bold',
        color='red',
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Metallicity
ax_met_lf = fig.add_axes([0.4, 0.90, 0.15, 0.02])  # [left, bottom, width, height]
met_lf_slider = Slider(
    ax=ax_met_lf,
    label='Met  ',
    valmin=1.0,
    valmax=50.0,
    valinit=init_met_lf,
    valfmt=' x%2.0f Solar'
)
#met_lf_slider.label.set_size(9)
met_lf_slider.label.set_ha('left')
met_lf_slider.label.set_position((-0.4, 0.5))
met_lf_slider.valtext.set_size(9)
met_lf_slider.valtext.set_ha('right')
met_lf_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Age
ax_age_lf = fig.add_axes([0.4, 0.85, 0.15, 0.02])  # [left, bottom, width, height]
age_lf_slider = Slider(
    ax=ax_age_lf,
    label="Age  ",
    valmin=np.log10(0.1),
    valmax=np.log10(10.0),
    valinit=np.log10(init_age_lf),
    #valfmt=' %2.1f'
)
age_lf_slider.label.set_ha('left')
age_lf_slider.label.set_position((-0.4, 0.5))
age_lf_slider.valtext.set_text(f'{init_age_lf: 2.1f} Gyr')
age_lf_slider.valtext.set_ha('right')
age_lf_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider


# Make a horizontal oriented slider to control the Teq
ax_teq_lf = fig.add_axes([0.4, 0.80, 0.15, 0.02])  # [left, bottom, width, height]
teq_lf_slider = Slider(
    ax=ax_teq_lf,
    label=r"T$_{\mathrm{eq}}$  ",
    valmin=160.0,
    valmax=1500.0,
    valinit=init_teq_lf,
    valfmt=' %4.0f K'
)
teq_lf_slider.label.set_ha('left')
teq_lf_slider.label.set_position((-0.4, 0.5))
teq_lf_slider.valtext.set_ha('right')
teq_lf_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

# Make a horizontal oriented slider to control the Envelope fraction
ax_fenv_lf = fig.add_axes([0.4, 0.75, 0.15, 0.02])  # [left, bottom, width, height]
fenv_lf_slider = Slider(
    ax=ax_fenv_lf,
    label=r"f$_{\mathrm{env}}$  ",
    valmin=np.log10(0.01),
    valmax=np.log10(20.0),
    valinit=np.log10(init_fenv_lf),
    #valfmt=" %2.2f %%"
)
fenv_lf_slider.label.set_ha('left')
fenv_lf_slider.label.set_position((-0.4, 0.5))
fenv_lf_slider.valtext.set_text(f'{init_fenv_lf: 2.2f} %')
fenv_lf_slider.valtext.set_ha('right')
fenv_lf_slider.valtext.set_position((1.65, 0.5))  # Shift text to the right outside the slider

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
    age_linear = 10**age_lf_slider.val
    age_lf_slider.valtext.set_text(f'{age_linear: 2.1f} Gyr')  # Update displayed value

    fenv_linear = 10**fenv_lf_slider.val
    fenv_lf_slider.valtext.set_text(f'{fenv_linear: 2.2f} %')  # Update displayed value

    line_iop.set_ydata(rad_iop(x, cmf_iop_slider.val,wmf_iop_slider.val,teq_iop_slider.val))
    line_iop_lim.set_ydata(rad_iop_lim(x, cmf_iop_slider.val,wmf_iop_slider.val,teq_iop_slider.val))
    fig.canvas.draw_idle()
    line_lf.set_ydata(rad_lf(x, met_lf_slider.val,age_linear,teq_lf_slider.val,fenv_linear))
    line_zeng.set_ydata(rad_zeng(x, cmf_zeng_slider.val))

# register the update function with each slider
cmf_iop_slider.on_changed(update)
wmf_iop_slider.on_changed(update)
teq_iop_slider.on_changed(update)
met_lf_slider.on_changed(update)
age_lf_slider.on_changed(update)
teq_lf_slider.on_changed(update)
fenv_lf_slider.on_changed(update)
cmf_zeng_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.84, 0.025, 0.1, 0.04])  # [left, bottom, width, height]
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    cmf_iop_slider.reset()
    wmf_iop_slider.reset()
    teq_iop_slider.reset()
    met_lf_slider.reset()
    age_lf_slider.reset()
    teq_lf_slider.reset()
    fenv_lf_slider.reset()
    cmf_zeng_slider.reset()

button.on_clicked(reset)

plt.show()
