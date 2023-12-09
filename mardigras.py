##############################################
#
#   READ FIT COEFFICIENTS
#
##############################################
# %matplotlib inline
# %matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
# from numpy import genfromtxt
from matplotlib.widgets import Button, Slider

from scipy.interpolate import RegularGridInterpolator

# %matplotlib inline
# from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, Box, HBox, VBox, Label, Layout, Button, Box, FloatText, Textarea, Dropdown, IntSlider
# import ipywidgets as widgets
# from IPython.display import display


#Paths to models
path_models = "./models/"

path_aguichine = path_models + "Aguichine2021_fit_coefficients.dat"
path_aguichine2 = path_models + "Aguichine2021_mr_all.dat"
path_zeng = path_models + "Zeng2016.dat"

#listcmf = []
#listwmf = []
#listteq = []
#lista = []
#listb = []
#listc = []
#listd = []
#i=0
#with open(path_aguichine, 'r') as file:
#    exit
#    for row in file:
#        i=i+1
#        if i > n_header and row != '\n':
#            #print(row.split())
#            cmf, wmf, teq, a, b, d, c, e, f, g, h = row.split()
#            listcmf.append(float(cmf))
#            listwmf.append(float(wmf))
#            listteq.append(float(teq))
#            lista.append(float(a))
#            listb.append(float(b))
#            listc.append(float(c))
#            listd.append(float(d))

# Load fit coefficients from Aguichine et al. 2021
listcmf,listwmf,listteq,lista,listb,listd,listc = np.loadtxt(path_aguichine,skiprows=19,unpack=True,usecols=(0,1,2,3,4,5,6))
        
def radius(mass,a,b,c,d):
    return 10**(a*np.log10(mass) + np.exp(-d*(np.log10(mass)+c)) + b)

# Load limits from Aguichine et al. 2021
listmasslow = []
listmasshigh = []
listm_small = []
liste_small = []
i=0
with open(path_aguichine2, 'r') as file:
    for row in file:
        # Skip header
        if i <= 21:
            i=i+1
        # Take values of small lists
        elif row != '\n':
            cmf, wmf, teq, tb, mb, ma, rb,ra,err = row.split()
            if int(err) == 0:
                listm_small.append(float(mb)+float(ma))
        # Grab values and reset small lists
        else:
            listmasslow.append(listm_small[0])
            listmasshigh.append(listm_small[-1])
            listm_small = []
            liste_small = []

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

interp_a = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_a, method='linear', bounds_error=False, fill_value=None)
interp_b = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_b, method='linear', bounds_error=False, fill_value=None)
interp_c = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_c, method='linear', bounds_error=False, fill_value=None)
interp_d = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_d, method='linear', bounds_error=False, fill_value=None)

interp_masslimlow = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_masslimlow, method='linear', bounds_error=False, fill_value=None)
interp_masslimhigh = RegularGridInterpolator((dimcmf, dimteq, dimwmf), data_masslimhigh, method='linear', bounds_error=False, fill_value=None)



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

interp_lf14 = RegularGridInterpolator((dim_met_lf14, dim_age_lf14, dim_teq_lf14, dim_mass_lf14, dim_fenv_lf14), data_r_lf14, method='linear', bounds_error=False, fill_value=None)


##############################################
#
#   LOAD DATA
#
##############################################

# Exoplanet catalog
list_catalog_rp,list_catalog_rpe1,list_catalog_rpe2,list_catalog_mp,list_catalog_mpe1,list_catalog_mpe2 = np.genfromtxt("./data/PS_2023.12.08_19.54.38.tab",skip_header=18,delimiter="\t",unpack=True,usecols=(2,3,4,5,6,7),filling_values=0.0)

list_rp = []
list_rpe1 = []
list_rpe2 = []
list_mp = []
list_mpe1 = []
list_mpe2 = []
for i in range(len(list_catalog_rp)):
    if list_catalog_rp[i]!=0.0 and list_catalog_mp[i]!=0.0:
        list_rp = np.append(list_rp,[list_catalog_rp[i]])
        list_rpe1 = np.append(list_rpe1,[list_catalog_rpe1[i]])
        list_rpe2 = np.append(list_rpe2,[list_catalog_rpe2[i]])
        list_mp = np.append(list_mp,[list_catalog_mp[i]])
        list_mpe1 = np.append(list_mpe1,[list_catalog_mpe1[i]])
        list_mpe2 = np.append(list_mpe2,[list_catalog_mpe2[i]])

list_catalog_rp,list_catalog_rpe1,list_catalog_rpe2,list_catalog_mp,list_catalog_mpe1,list_catalog_mpe2 = list_rp,list_rpe1,list_rpe2,list_mp,list_mpe1,list_mpe2

# Targets
list_targets_mp,list_targets_mpe2,list_targets_mpe1,list_targets_rp,list_targets_rpe2,list_targets_rpe1 = np.genfromtxt("./data/compo_targets.dat",skip_header=1,unpack=True,usecols=(1,2,3,4,5,6),filling_values=0.0)

# Solar System
list_ssystem_mp,list_ssystem_rp = np.genfromtxt("./data/solarsystem.dat",delimiter='\t',unpack=True,usecols=(0,1))


##############################################
#
#   MAKE PLOT
#
##############################################

## Definition of the plot_iop function, our "callback function".
# def plot_iop(cmf_iop,wmf_iop,teq_iop,cmf_zeng,met_lf14,age_lf14,teq_lf14,fenv_lf14):
#     ## Get parameters and validity range or IOP model
#     masslimlow,masslimhigh=interp_masslimlow([cmf_iop,teq_iop,wmf_iop]),interp_masslimhigh([cmf_iop,teq_iop,wmf_iop])
#     a,b,c,d = interp_a([cmf_iop,teq_iop,wmf_iop]),interp_b([cmf_iop,teq_iop,wmf_iop]),interp_c([cmf_iop,teq_iop,wmf_iop]),interp_d([cmf_iop,teq_iop,wmf_iop])
#     if cmf_iop < 0.0 or cmf_iop > 0.9 or wmf_iop < 0.1 or wmf_iop > 1.0 or teq_iop < 400.0 or teq_iop > 1300.0:
#         masslimlow,masslimhigh = 1.0,1.0

#     ## Get parameters of LF14 model

#     ## Plot parameters
#     xmin, xmax, nx = 0.5, 30.0, 50
#     ymin, ymax     = 0.5, 4.5

#     xmin_lim,xmax_lim = masslimlow,masslimhigh

#     ## Plot the figure
#     # x and y of IOP curves
#     x = np.logspace(np.log10(xmin), np.log10(xmax), nx)
#     r_iop_ext = 10**(a*np.log10(x) + 
#              np.exp(-d*(np.log10(x)+c)) + 
#              b)

#     # x and y of valid region of IOP
#     x_lim = np.logspace(np.log10(xmin_lim), np.log10(xmax_lim), nx)
#     r_iop = 10**(a*np.log10(x_lim) + 
#              np.exp(-d*(np.log10(x_lim)+c)) + 
#              b)

#     plt.figure(figsize=(6,5))

#     # Zeng fixed
#     zeng_width=1.0
#     plt.plot(zeng_mass,zeng_purefe,linewidth=zeng_width,color='black')
#     plt.plot(zeng_mass,zeng_earth,linewidth=zeng_width,color='brown')
#     plt.plot(zeng_mass,zeng_rock,linewidth=zeng_width,color='grey')
#     plt.plot(zeng_mass,zeng_50wat,linewidth=zeng_width,color='cyan')
#     plt.plot(zeng_mass,zeng_100wat,linewidth=zeng_width,color='cyan')

#     # Zeng sliding
#     zeng_radii_slide = np.zeros(nx)
#     for i in range(nx):
#         zeng_radii_slide[i] = interp_zeng([x[i],cmf_zeng])
#     plt.plot(x,zeng_radii_slide,linewidth=2,color='grey')


#     # IOP sliding
#     plt.plot(x_lim, r_iop, linewidth=2,color='blue')
#     plt.plot(x, r_iop_ext, linewidth=2,color='blue',ls='--')

#     # LF14 sliding
#     lf14_radii_slide = np.zeros(nx)
#     for i in range(nx):
#         lf14_radii_slide[i] = interp_lf14([met_lf14,age_lf14,teq_lf14,x[i],fenv_lf14])
#     plt.plot(x,lf14_radii_slide,linewidth=2,color='red')

#     # Exoplanet catalog

#     # User defined points
#     plt.plot([1],[1],marker='+')

#     ## Set up the figure axes, etc.
#     plt.xscale("log")
#     #plt.yscale("log")
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.xlabel('Mass [Me]')
#     plt.ylabel('Radius [Re]')
#     plt.grid(visible=True,which='major', axis='both')


# cmf_iop=FloatSlider(min=0.0, max=0.9, step=0.01, value=0.3,description='CMF')
# wmf_iop=FloatSlider(min=0.1, max=1.0, step=0.01, value=0.5,description='WMF')
# teq_iop=FloatSlider(min=400.0, max=1300.0, step=10.0, value=700.0,description='Teq')

# cmf_zeng=FloatSlider(min=0.0, max=1.0, step=0.01, value=0.325,description='CMF')

# met_lf14=FloatSlider(min=1, max=50, step=1, value=1,description='Z(met)')
# age_lf14=FloatSlider(min=0.1, max=10, step=0.1, value=1,description='Age')
# teq_lf14=FloatSlider(min=160, max=1500, step=10, value=700,description='Teq')
# fenv_lf14=FloatSlider(min=0.01, max=20, step=0.001, value=1,description='f_env')

# box_iop = VBox([Label(value='Aguichine+2021', layout=Layout(display="flex", justify_content="center")),cmf_iop, wmf_iop, teq_iop])
# box_zeng = VBox([Label(value='Zeng+2016',layout=Layout(display="flex", justify_content="center")),cmf_zeng])
# box_lf14 = VBox([Label(value='Lopez&Fortney 2014', layout=Layout(display="flex", justify_content="center")),met_lf14, age_lf14, teq_lf14,fenv_lf14])

# widget_layout = Box([box_lf14,box_iop,box_zeng])

# output_plot = widgets.interactive_output(plot_iop, {'cmf_iop': cmf_iop, 'wmf_iop': wmf_iop, 'teq_iop': teq_iop,'cmf_zeng':cmf_zeng,'met_lf14':met_lf14,'age_lf14':age_lf14,'teq_lf14':teq_lf14,'fenv_lf14':fenv_lf14})

# display(widget_layout,output_plot)

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
    rp=np.zeros(len(x))
    for i in range(len(x)):
        rp[i] = interp_lf14([met,age,teq,x[i],fenv])
    return rp

def rad_zeng(x,cmf):
    rp=np.zeros(len(x))
    for i in range(len(x)):
        rp[i] = interp_zeng([x[i],cmf])
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


# Create the figure and the sliding lines
fig, ax = plt.subplots()

# Figure options
fig.set_figheight(7)
fig.set_figwidth(7)
plt.xscale("log")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('Mass [Me]')
plt.ylabel('Radius [Re]')
plt.grid(visible=True,which='major', axis='both')

# Sliding lines
line_iop, = ax.plot(x, rad_iop(x, init_cmf_iop,init_wmf_iop,init_teq_iop),lw=2,color='blue',ls='--',zorder=20)
line_iop_lim, = ax.plot(x, rad_iop_lim(x, init_cmf_iop,init_wmf_iop,init_teq_iop),lw=2,color='blue',zorder=25)
line_lf, = ax.plot(x, rad_lf(x,init_met_lf,init_age_lf,init_teq_lf,init_fenv_lf),lw=2,color='red',zorder=30)
line_zeng, = ax.plot(x, rad_zeng(x,init_cmf_zeng),lw=2,color='brown',zorder=10)

# Planets
list_catalog_mpe = [abs(list_catalog_mpe2), list_catalog_mpe1]
list_catalog_rpe = [abs(list_catalog_rpe2), list_catalog_rpe1]
list_targets_mpe = [abs(list_targets_mpe2), list_targets_mpe1]
list_targets_rpe = [abs(list_targets_rpe2), list_targets_rpe1]
# scatter_catalog, = 
ax.errorbar(list_catalog_mp,list_catalog_rp,
            yerr=list_catalog_rpe,
            xerr=list_catalog_mpe,
            fmt='o',zorder=-30,
            c="black",alpha=0.2)
# scatter_targets, = ax.errorbar()
ax.errorbar(list_targets_mp,list_targets_rp,
            yerr=list_targets_rpe,
            xerr=list_targets_mpe,
            ls='',c='yellow',elinewidth=3,
            marker='*',mfc='yellow',mec='black', ms=15, mew=1,
            zorder=50)
# scatter_ssystem, = ax.errorbar()
ax.errorbar(list_ssystem_mp,list_ssystem_rp,
            ls='',c='red',
            marker='o',mfc='red',mec='black', ms=5, mew=1,
            zorder=35)

# Zeng fixed
zeng_width=1.0
ax.plot(list_zeng_masses,list_zeng_fe_radii,linewidth=zeng_width,color='black',zorder=5)
ax.plot(list_zeng_masses,list_zeng_ea_radii,linewidth=zeng_width,color='brown',zorder=6)
ax.plot(list_zeng_masses,list_zeng_mg_radii,linewidth=zeng_width,color='grey',zorder=7)
ax.plot(zeng_mass,zeng_50wat,linewidth=zeng_width,color='cyan',zorder=8)
ax.plot(zeng_mass,zeng_100wat,linewidth=zeng_width,color='cyan',zorder=9)


# adjust the main plot to make room for the sliders
fig.subplots_adjust(top=0.7)

# Labels
ax.text(0.05, 0.95, 'Aguichine et al. 2021',weight='bold',
        transform=fig.transFigure,
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the CMF
ax_cmf_iop = fig.add_axes([0.1, 0.90, 0.15, 0.02])
cmf_iop_slider = Slider(
    ax=ax_cmf_iop,
    label='CMF ',
    valmin=0.0,
    valmax=0.9,
    valinit=init_cmf_iop,
    valfmt=' %1.3f'
)

# Make a horizontal oriented slider to control the WMF
ax_wmf_iop = fig.add_axes([0.1, 0.85, 0.15, 0.02])
wmf_iop_slider = Slider(
    ax=ax_wmf_iop,
    label="WMF ",
    valmin=0.1,
    valmax=1.0,
    valinit=init_wmf_iop,
    valfmt=' %1.3f'
)

# Make a horizontal oriented slider to control the Teq
ax_teq_iop = fig.add_axes([0.1, 0.80, 0.15, 0.02])
teq_iop_slider = Slider(
    ax=ax_teq_iop,
    label="Teq [K] ",
    valmin=400.0,
    valmax=1300.0,
    valinit=init_teq_iop,
    valfmt=' %4.0f'
)

# Labels
ax.text(0.38, 0.95, 'Lopez & Fortney 2014',weight='bold',
        transform=fig.transFigure,
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the Metallicity
ax_met_lf = fig.add_axes([0.45, 0.90, 0.15, 0.02])
met_lf_slider = Slider(
    ax=ax_met_lf,
    label='M [Sun] ',
    valmin=1.0,
    valmax=50.0,
    valinit=init_met_lf,
    valfmt=' %2.0f'
)

# Make a horizontal oriented slider to control the Age
ax_age_lf = fig.add_axes([0.45, 0.85, 0.15, 0.02])
age_lf_slider = Slider(
    ax=ax_age_lf,
    label="Age [Gyr] ",
    valmin=0.1,
    valmax=10.0,
    valinit=init_age_lf,
    valfmt=' %2.1f'
)

# Make a horizontal oriented slider to control the Teq
ax_teq_lf = fig.add_axes([0.45, 0.80, 0.15, 0.02])
teq_lf_slider = Slider(
    ax=ax_teq_lf,
    label="Teq [K] ",
    valmin=160.0,
    valmax=1500.0,
    valinit=init_teq_lf,
    valfmt=' %4.0f'
)

# Make a horizontal oriented slider to control the Envelope fraction
ax_fenv_lf = fig.add_axes([0.45, 0.75, 0.15, 0.02])
fenv_lf_slider = Slider(
    ax=ax_fenv_lf,
    label="f [%] ",
    valmin=0.01,
    valmax=20.0,
    valinit=init_fenv_lf,
    valfmt=' %2.2f'
)

# Labels
ax.text(0.75, 0.95, 'Zeng et al. 2016',weight='bold',
        transform=fig.transFigure,
        bbox={'ec': 'white', 'fc':'white','color':'blue', 'pad': 10})

# Make a horizontal slider to control the CMF
ax_cmf_zeng = fig.add_axes([0.75, 0.90, 0.15, 0.02])
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
    line_iop.set_ydata(rad_iop(x, cmf_iop_slider.val,wmf_iop_slider.val,teq_iop_slider.val))
    line_iop_lim.set_ydata(rad_iop_lim(x, cmf_iop_slider.val,wmf_iop_slider.val,teq_iop_slider.val))
    fig.canvas.draw_idle()
    line_lf.set_ydata(rad_lf(x, met_lf_slider.val,age_lf_slider.val,teq_lf_slider.val,fenv_lf_slider.val))
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
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
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
