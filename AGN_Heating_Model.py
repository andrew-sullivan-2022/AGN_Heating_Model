import builtins
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.integrate as int
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, griddata
from scipy.stats import gaussian_kde
from pydl.pydlutils.cooling import read_ds_cooling
import sympy as sp
from numba import njit, prange
from tqdm import tqdm
from RAiSEHD import RAiSE_run

x = sp.symbols('x')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

## DEFINED FUNCTIONS

## Reading in files (as flattened lists)
def read_file_to_list(filename):
    with open(filename, 'r') as file:
        content = file.read()
    content = content.replace('\n', ' ')
    content = content.replace(',', ' ')
    content = content.replace('[', '')
    content = content.replace(']', '')
    elements = content.split()
    return np.loadtxt(elements)

## Reading in files (as nested arrays)
def read_file_to_nested_array(filename):
    ## data shape: [N][M][X]
    blocks = []
    current_block = []
    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                if current_block:
                    blocks.append(np.array(current_block))
                    current_block = []
            else:
                current_block.append(
                    np.fromstring(stripped, sep=" ")
                )
    # catch last block
    if current_block:
        blocks.append(np.array(current_block))
    return np.array(blocks)

## Saving files (as nested arrays)
def save_nested_arrays(filename, data):
    ## data shape: [N][M][X]
    with open(filename, "w") as f:
        for i, block in enumerate(data):      # N blocks
            for array in block:               # M arrays
                np.savetxt(f, np.atleast_1d(array)[None, :])
            if i != len(data) - 1:
                f.write("\n")

## Weighted mean function
def weighted_mean(values, weights, axis=0):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    # reshape weights so they broadcast along chosen axis
    #weights_expanded = np.expand_dims(weights, axis=tuple(range(1, values.ndim)))
    return np.sum(values*weights, axis=axis)/np.sum(weights, axis=axis)

## Weighted percentile function
def weighted_percentile(values, percentiles, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cdf = np.cumsum(weights_sorted, dtype=float)
    cdf /= cdf[-1]
    return np.interp(np.array(percentiles)/100.0, cdf, values_sorted)

## Numba-friendly Heaviside function
@njit
def heaviside(x):
    return 0 if x < 0 else 1

## BUILDING THE JET MODEL

## Number of radio samples
#N = 50000
np.random.seed(15)

## Radial scale
# scale-free radial scale
radius_bins = 500
s = np.logspace(-5, 1, base=10, num=radius_bins)

## Physical constants
# speed of light
c_light = 3*10**8 # m s^-1
# newton's gravitational constant
G = 6.6743*10**(-11) # m^3 kg^-1 s^-2
# the bolztmann constant
kB = 1.3806*10**(-23) # m^2 kg s^-2 K^-1
# proton mass
mp = 1.6726*10**(-27) # kg
# electron volt
eV = 1.602*10**(-19) # J

## Cosmological constants
# hubble parameter
h = 0.6751
# present-day critical density of the universe
rho_crit = 1.8788*10**(-26)*h**2 # kg m^-3
# density parameters
Omega_b = 0.0486
Omega_DM = 0.2589
# fraction of baryon content
f_b = Omega_b/(Omega_b + Omega_DM)
## Unit conversions
# solar mass
Msol = 1.989*10**30 # kg
# year
yr = 3.156*10**7
# megayear
Myr = 3.156*10**13 # s
# gigayear
Gyr = 3.156*10**16 # s
# kiloparsec
kpc = 3.086*10**19 # m
# megaparsec
Mpc = 3.086*10**22 # m

## Gas parameters
# mean molecular weight
mu = 0.6
# mean electron weight
mu_e = 1.148
# mean ion weight
mu_i = (1/mu + 1/mu_e)**(-1)
# adiabatic index
gamma = 5/3

## Perseus cluster
# observed redshift
redshift = 0.017284 # Hitomi+2018
from astropy.cosmology import LambdaCDM
Planck_cosmology = LambdaCDM(H0=h*100, Om0=0.3, Ode0=0.7)
arcmin_to_kpc = Planck_cosmology.kpc_proper_per_arcmin(redshift).value*kpc
# print('At Perseus redshift, 1 arcmin = ', np.round(arcmin_to_kpc/kpc, 2), ' kpc')
# # de Vries+2023 observations of r2500, M2500
# r2500 = 26*arcmin_to_kpc # m
# M2500 = (4/3)*np.pi*r2500**3*2500*rho_crit # kg
# Urban+2014 observations of r500, M500
r500 = 59.7*arcmin_to_kpc # m
# M500 = (4/3)*np.pi*r500**3*500*rho_crit # kg
# # Matsushita+2026 observations of r200, M200
# r200 = 84*arcmin_to_kpc # m
# M200 = (4/3)*np.pi*r200**3*200*rho_crit # kg
# print('Perseus virial radius: r500 = ', np.round(r500/Mpc, 2), ' Mpc, and r200 = ', np.round(r200/Mpc, 2), ' Mpc')
# print('Perseus virial mass: M500 = ', np.round(M500/(10**14*Msol), 2), '*10^14 Msol, and M200 = ', np.round(M200/(10**14*Msol), 2), '*10^14 Msol')
# print('Perseus ratio r500/r200 = ', np.round(r500/r200, 2))
# Perseus halocentric radius
halo_radius = np.multiply(r500, s)
# # truncation radius from Hurier+2019
# r_trunc = 3*r500
# # assuming a mass-concentration relation (Duffy+2008a)
# c200 = 5.71*(M200/(2*10**12*h**(-1)*Msol))**(-0.084)*(1 + redshift)**(-0.47)
# c200_uncertainty_region = np.asarray([5.59, 5.83])*(M200/(2*10**12*h**(-1)*Msol))**np.asarray([-0.09, -0.078])*(1 + redshift)**np.asarray([-0.51, -0.43])
# c200_min, c200_max = min(c200_uncertainty_region), max(c200_uncertainty_region)
# c500, c500_min, c500_max = (r500/r200)*c200, (r500/r200)*c200_min, (r500/r200)*c200_max
# print('Perseus concentration: c500 = ', np.round(c500, 2), '+/-', np.round(np.average([c500_max-c500, c500-c500_min]), 2))

## Perseus' X-ray fits
# Perseus intracluster gas density profile
def Perseus_gas_density(r): # kg m^-3
    return mu_e*mp*(4.6*10**4*(1 + (r/(57*kpc))**2)**(-3*1.2/2) + 3.6*10**3*(1 + (r/(278*kpc))**2)**(-3*0.71/2))
Perseus_gas_density_derivative = sp.lambdify(x, sp.diff(Perseus_gas_density(x), x)) # kg m^-4
# Perseus temperature profile
def Perseus_temperature(r): # K
    return 7*(eV*10**3/kB)*(1 + (r/(73.8*kpc))**3)*(2.3 + (r/(73.8*kpc))**3)**(-1)*(1 + (r/(1600*kpc))**1.7)**(-1)
# # Perseus thermal pressure profile
# def Perseus_th_pressure(r): # Pa
#     return (kB/(mu*mp))*Perseus_gas_density(r)*Perseus_temperature(r)
# Perseus_th_pressure_derivative = sp.lambdify(x, sp.diff(Perseus_th_pressure(x), x)) # kg m^-2 s^-2
# # Perseus gas mass profile
# def Perseus_gas_mass(r):# kg
#     return 4*np.pi*int.quad(lambda r_: Perseus_gas_density(r_)*r_**2, 0, r)[0]
# # Perseus HE mass profile
# def Perseus_HE_mass(r): # kg
#     return -(r**2/(G*Perseus_gas_density(r)))*Perseus_th_pressure_derivative(r)
#
# ## Universal gas fraction fits
# # halo mass-dependent median estimate from Eckert+2021
# Eckert_univ_gas_frac_500 = 0.079*(M500/(10**14*Msol))**0.22
# print('Eckert+21 universal gas fraction at r500 = ', np.round(Eckert_univ_gas_frac_500, 3))
# # halo mass-dependent gas fraction fit from Angelinelli+2022
# def Angelinelli22_univ_gas_fit(s, M500, p1=0.733, p2=0.223, p3=0.674, p4=-0.708):
#     w = (M500)/(5*10**14*Msol/h)
#     def depletion_factor(s, p1, p2, p3, p4):
#         return p1*w**p2*s**(p3 + p4*w)
#     return f_b*depletion_factor(s, p1, p2, p3, p4)
# # using the simulated gas fractions from Angelinelli et al. 2022 as upper bound
# fgas_500_lo = np.round(Angelinelli22_univ_gas_fit(1, M500), 3)
# fgas_200_lo = np.round(Angelinelli22_univ_gas_fit(r200/r500, M500), 3)
# print('Angelinelli+22 gas fractions: at r500 = ', fgas_500_lo, 'and at r200 = ', fgas_200_lo)
# # using the observed gas fractions from Matsushita et al. 2026 as lower bound
# fgas_500_hi = 0.13
# fgas_200_hi = 0.15
# # median gas fractions
# fgas_500_median, fgas_200_median = np.median([fgas_500_lo, fgas_500_hi]), np.median([fgas_200_lo, fgas_200_hi])
# print('Median gas fractions: at r500 = ', np.round(fgas_500_median, 3), 'and at r200 = ', np.round(fgas_200_median, 3))

## Loading Perseus' gas profiles
# intracluster gas density and temperature profiles
gas_density_profile = np.empty(radius_bins)
temperature_profile = np.empty(radius_bins)
for i in range(radius_bins):
    gas_density_profile[i] = Perseus_gas_density(halo_radius[i])
    temperature_profile[i] = Perseus_temperature(halo_radius[i])
print('Perseus cluster profiles loaded')
# # thermal velocity
# th_velocity_profile = np.sqrt(np.multiply(temperature_profile, 3*kB/(mu*mp)))  # m s^-1
# # gas and HE mass profiles
# gas_mass_profile = np.empty(radius_bins)
# HE_mass_profile = np.empty(radius_bins)
# for i in range(radius_bins):
#     gas_mass_profile[i] = Perseus_gas_mass(halo_radius[i]) # kg
#     HE_mass_profile[i] = Perseus_HE_mass(halo_radius[i]) # kg
# # gas fraction profile
# HE_gas_frac_profile = np.divide(gas_mass_profile, HE_mass_profile)
#
# ## Sullivan et al. 2024 NTP fraction model
# # equilibrium entropy slope profile
# def idealised_entropy_slope(s, c, a, d, e, eta):
#     def u_func(c, a):
#         return 1/(int.quad(lambda s: (s**(2 - a))/((1 + c*s)**(3 - a)), 0, 1)[0])
#     def C_func(c, a, d, e):
#         return (d*(a-e) + c*(3-e))/(3-a)
#     def U_func(c, a, d, e):
#         return 1/int.quad(lambda s: s**(2-e)/(1 + C_func(c, a, d, e)*s)**(3-e), 0, 1)[0]
#     def I_func(s, c, a, d, e, eta):
#         return int.quad(lambda s_: ((1 - eta*f_b)*u_func(c, a)*int.quad(lambda s__: s__**(2-a)/(1 + c*s__)**(3-a), 0, s_)[0] + eta*f_b*U_func(c, a, d, e)*int.quad(lambda s__: s__**(2-e)/(1 + C_func(c, a, d, e)*s__)**(3-e), 0, s_)[0])/(s_**(2+e)*(1 + C_func(c, a, d, e)*s_)**(3-e)), s, np.infty)[0]
#     return (5/3)*(e + (3-e)*C_func(c, a, d, e)*s/(1 + C_func(c, a, d, e)*s)) - ((1 - eta*f_b)*u_func(c, a)*int.quad(lambda s_: s_**(2-a)/(1 + c*s_)**(3-a), 0, s)[0] + eta*f_b*U_func(c, a, d, e)*int.quad(lambda s_: s_**(2-e)/(1 + C_func(c, a, d, e)*s_)**(3-e), 0, s)[0])/(I_func(s, c, a, d, e, eta)*s**(e+1)*(1 + C_func(c, a, d, e)*s)**(3-e))
# # entropy slope profile for Perseus-like idealisation
# Perseus_like_entropy_slope = np.empty(radius_bins)
# for i in range(radius_bins):
#     Perseus_like_entropy_slope[i] = idealised_entropy_slope(s[i], 2.5, 1, 1, 0, 0.8)
# # weighting function from Sullivan et al. 2024
# def weighting_func(s, midpoint=0.4, steepness=5):
#     return 0.8 + 1/(5*(1 + np.exp(steepness*(np.log10(s) + midpoint))))
# weighting_func_profile = weighting_func(s)
# # NTP fraction profile in the outskirts
# Perseus_like_NTP_fraction_profile = np.empty(radius_bins)
# for i in range(radius_bins):
#     Perseus_like_NTP_fraction_profile[i] = 1 - np.exp(np.trapz(np.divide(np.multiply(Perseus_like_entropy_slope[:i], np.subtract(weighting_func_profile[:i], 1)), s[:i]), s[:i]))
# # calculating the linear-log slope
# Perseus_like_NTP_fraction_derivative = np.gradient(Perseus_like_NTP_fraction_profile, halo_radius)
# Perseus_like_NTP_fraction_linear_log_slope = np.multiply(Perseus_like_NTP_fraction_derivative, halo_radius)
# NTP_linear_log_derivative = np.round(np.average(Perseus_like_NTP_fraction_linear_log_slope[(halo_radius>=r500) & (halo_radius<=2*r500)]), 2)
# print('Linear-log deriv = ', NTP_linear_log_derivative)
#
# ## Fitting Perseus' outskirts NTP fraction
# # fitting the NTP value to a quadratic solution using gas fraction constraints
# def calc_NTP_fraction_from_fgas(radii, fgas_vals):
#     frac_ratio = np.empty(len(radii))
#     for i in range(len(radii)):
#         HE_frac = Perseus_gas_mass(radii[i])/Perseus_HE_mass(radii[i])
#         frac_ratio[i] = HE_frac/fgas_vals[i]
#     NTP_vals = np.empty(len(radii))
#     for i in range(len(radii)):
#         NTP_vals[i] = 1 - (1/(2*frac_ratio[i]))*(1 + np.sqrt(1 - 4*frac_ratio[i]*radii[i]*NTP_linear_log_derivative*Perseus_th_pressure(radii[i])/(G*Perseus_HE_mass(radii[i])*Perseus_gas_density(radii[i]))))
#     return NTP_vals
# # median NTP fractions at r500, r200
# NTP_fraction_fgas_constraint_median = calc_NTP_fraction_from_fgas([r500, r200], [fgas_500_median, fgas_200_median])
# # uncertainty in the NTP fractions at r500, r200
# NTP_fraction_fgas_constraint_min, NTP_fraction_fgas_constraint_max = calc_NTP_fraction_from_fgas([r500, r200], [fgas_500_hi, fgas_200_hi]), calc_NTP_fraction_from_fgas([r500, r200], [fgas_500_lo, fgas_200_lo])
# NTP_fraction_fgas_constraint_uncertainty = [np.average([NTP_fraction_fgas_constraint_median[0] - NTP_fraction_fgas_constraint_min[0], NTP_fraction_fgas_constraint_max[0] - NTP_fraction_fgas_constraint_median[0]]), np.average([NTP_fraction_fgas_constraint_median[1] - NTP_fraction_fgas_constraint_min[1], NTP_fraction_fgas_constraint_max[1] - NTP_fraction_fgas_constraint_median[1]])]
# print('NTP fraction at r500 = ', np.round(100*NTP_fraction_fgas_constraint_median[0], 2), '% and at r200 = ', np.round(100*NTP_fraction_fgas_constraint_median[1], 2), '%')
# print('NTP fraction uncertainty at r500 = ', np.round(100*NTP_fraction_fgas_constraint_uncertainty[0], 2), '% and at r200 = ', np.round(100*NTP_fraction_fgas_constraint_uncertainty[1], 2), '%')
# # average scaling-up of Sullivan+ 2024 profile
# NTP_fraction_scaling_from_Sullivan24 = [NTP_fraction_fgas_constraint_median[0]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r500))], NTP_fraction_fgas_constraint_median[1]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r200))]]
# NTP_fraction_scaling_error_from_Sullivan24 = [np.average([NTP_fraction_fgas_constraint_max[0]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r500))] - NTP_fraction_scaling_from_Sullivan24[0], NTP_fraction_scaling_from_Sullivan24[0] - NTP_fraction_fgas_constraint_min[0]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r500))]]), np.average([NTP_fraction_fgas_constraint_max[1]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r200))] - NTP_fraction_scaling_from_Sullivan24[1], NTP_fraction_scaling_from_Sullivan24[1] - NTP_fraction_fgas_constraint_min[1]/Perseus_like_NTP_fraction_profile[np.argmin(np.abs(halo_radius - r200))]])]
# print('NTP fraction is larger by approximately ', np.round(100*(NTP_fraction_scaling_from_Sullivan24[0]-1), 2), '% +/- ', np.round(100*(NTP_fraction_scaling_error_from_Sullivan24[0]), 2), '% at r500, and ', np.round(100*(NTP_fraction_scaling_from_Sullivan24[1]-1), 2), '% +/- ',  np.round(100*(NTP_fraction_scaling_error_from_Sullivan24[1]), 2), '% at r200 than Sullivan+2024 profile')
# # extrapolating Sullivan+ 2024 profiles via scale-up
# NTP_fraction_outskirts = np.multiply(1.50, Perseus_like_NTP_fraction_profile)
# NTP_fraction_outskirts_min = np.multiply((1.50 - 0.35), Perseus_like_NTP_fraction_profile)
# NTP_fraction_outskirts_max = np.multiply((1.50 + 0.35), Perseus_like_NTP_fraction_profile)
# # gas velocity kicks in the cluster outskirts
# v_kick_outskirts = np.divide(th_velocity_profile, np.sqrt(np.subtract(np.reciprocal(NTP_fraction_outskirts), 1)))
# v_kick_outskirts_min, v_kick_outskirts_max = np.divide(th_velocity_profile, np.sqrt(np.subtract(np.reciprocal(NTP_fraction_outskirts_min), 1))), np.divide(th_velocity_profile, np.sqrt(np.subtract(np.reciprocal(NTP_fraction_outskirts_max), 1)))




## Coded function
# calculating profiles for a given jet power range
def func_calculate_feedback(log_Qjet_vals, log_active_age_vals, duty_cycle, redshift, gas_density_profile, temperature_profile, halo_radius, log_Qjet=0.01, log_dt=0.1):
    ## Running RAiSE
    # loading jet powers
    log_Qjet_vals = np.round(log_Qjet_vals, builtins.int(-np.log10(log_Qjet)))
    power_res = 1 if np.isscalar(log_Qjet_vals) else len(log_Qjet_vals)
    # loading active ages
    log_active_age_vals = np.round(log_active_age_vals, builtins.int(-np.log10(log_dt)))
    # source age time steps
    log_age_steps = np.round(np.linspace(0.1, 9, num=builtins.int((9 - 0.1)/log_dt) + 1), 2)
    log_active_age_indices = np.searchsorted(log_age_steps, np.atleast_1d(log_active_age_vals))
    # loading active ages, indexed from source age time steps
    log_active_age_vals = log_age_steps[log_active_age_indices]
    age_res = 1 if np.isscalar(log_active_age_vals) else len(log_active_age_vals)
    ## RAiSE inputs
    # variables with fixed values
    axis_ratio = 2.83  # default
    equipartition = -1.5  # default
    jet_lorentz = 5  # default
    spectral_index = 0.7  # default
    # angular components
    angular_res = 64
    angles = np.arange(0, angular_res, 1).astype(np.int_)[::-1]
    dtheta = (np.pi / 2) / (angular_res - 1)
    theta = dtheta * angles
    solid_angles = np.empty(64)
    for i in range(angular_res):
        if theta[i] == 0:
            solid_angles[i] = 2*np.pi*(np.cos(theta[i]) - np.cos(theta[i] - dtheta/2))
        elif theta[i] == np.pi/2:
            solid_angles[i] = 2*np.pi*(np.cos(theta[i] - dtheta/2) - np.cos(theta[i]))
        else:
            solid_angles[i] = 2*np.pi*(np.cos(theta[i] - dtheta/2) - np.cos(theta[i] + dtheta/2))
    # log gas density slopes into RAiSE
    gas_density_log_slope = np.multiply(np.gradient(gas_density_profile, halo_radius), np.divide(halo_radius, gas_density_profile))
    Perseus_betas = -(gas_density_log_slope[1:] + gas_density_log_slope[:-1])/2

    ## Running RAiSE
    # run (turn on/off accordingly)
    RAiSE_run(-1, redshift=redshift, axis_ratio=axis_ratio, jet_power=log_Qjet_vals, source_age=log_age_steps, angle=0., rho0Value=gas_density_profile[0], betas=Perseus_betas, regions=halo_radius[:-1], temperature=temperature_profile[0], active_age=np.max(log_active_age_vals), brightness=False, resolution=None, particle_data=False)

    ## Environmental profiles
    # number density profiles
    number_density_profile = np.divide(gas_density_profile, (mu*mp))  # m^-3
    hydrogen_density_profile = np.multiply((4/9), number_density_profile)  # m^-3
    # thermal pressure profiles
    iso_th_pressure_profile = np.multiply(gas_density_profile, kB * temperature_profile[0] / (mu * mp))  # Pa
    th_pressure_profile = np.multiply(gas_density_profile, kB*temperature_profile/(mu*mp)) # Pa
    th_pressure_derivative = np.gradient(th_pressure_profile, halo_radius) # Pa m^-1
    # gravitational potential energy (assuming hydrostatic environment)
    grav_potential = (kB/(mu*mp))*np.multiply(temperature_profile, np.log(gas_density_profile)) # m^2 s^-2
    # thermal velocity
    th_velocity_profile = np.sqrt(np.multiply(temperature_profile, 3*kB/(mu*mp)))  # m s^-1

    ## Cooling function
    # loading cooling function
    abundance_file = 'm-05.cie'  # metallicity for Sutherland & Dopita (1993) cooling function, here corresponding to Z = 0.3 Zsolar
    # interpolating from the gas temperature to the cooling function
    logT_cool, logLambda_cool = read_ds_cooling(abundance_file)  # T in K, Lambda 13 dex higher than SI value
    logLambda_cool_interp = np.interp(np.log10(temperature_profile), logT_cool, logLambda_cool - 13)  # log ( W m^3 )
    # interpolated cooling function
    Lambda_cool_interp = np.asarray([10 ** i for i in logLambda_cool_interp])  # W m^3
    # cooling rate
    cooling_rate = np.multiply(np.square(hydrogen_density_profile), Lambda_cool_interp)  # W m^-3

    ## Feedback relationship
    # velocity kick from Sullivan et al. 2025
    velocity_per_volumetric_energy_rate = np.divide((gamma - 1), np.add(th_pressure_derivative, 2*gamma*np.divide(th_pressure_profile, halo_radius)))  # m^2 s^2 kg^-1

    ## RAiSE outputs
    # cocoon lengths over parameters
    R_cocoon = np.empty((power_res, age_res))
    R_cocoon_lengths = np.empty((power_res, age_res, angular_res))
    # shock lengths over parameters
    R_shock = np.empty((power_res, age_res))
    R_shock_lengths = np.empty((power_res, age_res, angular_res))
    shock_pressure = np.empty((power_res, age_res))
    # calling the RAiSE outputs
    for i in range(power_res):
            file1 = ('LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.4f}_Q={:.2f}_s={:.2f}_T={:.2f}_v={:.2f}_y={:.2f}_z={:.2f}'.format(axis_ratio, np.abs(equipartition), np.round(-np.log10(gas_density_profile[0]), decimals=4), log_Qjet_vals[i], 2*np.abs(spectral_index)+1, np.max(log_active_age_vals), 0., jet_lorentz, redshift))
            df1 = pd.read_csv(file1+'.csv')
            for j, index in enumerate(log_active_age_indices):
                R_cocoon_lengths[i, j] = np.fromstring(df1.iloc[index, 1].strip('[]'), sep=' ')[::-1]*kpc/2 # m
                R_cocoon[i, j] = R_cocoon_lengths[i, j, -1] # m
                R_shock_lengths[i, j] = np.fromstring(df1.iloc[index, 2].strip('[]'), sep=' ')[::-1]*kpc/2 # m
                R_shock[i, j] = R_shock_lengths[i, j, -1] # m
                shock_pressure[i, j] = df1.iloc[index, 3]  # Pa

    ## Outburst sizes
    # shock size, minor axis, height and length
    R_shock_minor = R_shock_lengths[:, :, 0]
    R_shock_lengths_yval = np.multiply(R_shock_lengths, np.sin(theta))
    R_shock_lengths_xval = np.multiply(R_shock_lengths, np.cos(theta))
    # volume of the cocoon, shock and shocked shell
    @njit(parallel=True)
    def compute_volumes(R_cocoon_lengths, R_shock_lengths, solid_angles):
        V_cocoon = np.empty((power_res, age_res))
        V_shock = np.empty((power_res, age_res))
        V_shock_shell = np.empty((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                V_cocoon[i, j] = (1/3)*np.sum(np.multiply(solid_angles, np.power(R_cocoon_lengths[i, j], 3)))
                V_shock[i, j] = (1/3)*np.sum(np.multiply(solid_angles, np.power(R_shock_lengths[i, j], 3)))
                V_shock_shell[i, j] = (1/3)*np.sum(np.multiply(solid_angles, np.subtract(np.power(R_shock_lengths[i, j], 3), np.power(R_cocoon_lengths[i, j], 3))))
        return V_cocoon, V_shock, V_shock_shell
    V_cocoon, V_shock, V_shock_shell = compute_volumes(R_cocoon_lengths, R_shock_lengths, solid_angles)
    # assuming spherical geometry
    R_cocoon_sphere = np.cbrt(3*V_cocoon/(4*np.pi))

    ## Filling factor profiles
    # filling factor of the cocoon, shock and shock shell
    @njit(parallel=True)
    def compute_filling_factors(R_cocoon_lengths, R_shock_lengths, solid_angles, halo_radius):
        cocoon_filling_factors = np.empty((power_res, age_res, radius_bins))
        shock_filling_factors = np.empty((power_res, age_res, radius_bins))
        shock_shell_filling_factors = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
                for j in range(age_res):
                    for k in range(radius_bins):
                        cocoon_angles_filled = 0
                        shock_angles_filled = 0
                        shock_shell_angles_filled = 0
                        for m in range(angular_res):
                            cocoon_angles_filled += solid_angles[m]*heaviside(R_cocoon_lengths[i, j, m] - halo_radius[k])
                            shock_angles_filled += solid_angles[m]*heaviside(R_shock_lengths[i, j, m] - halo_radius[k])
                            shock_shell_angles_filled += solid_angles[m]*heaviside(R_shock_lengths[i, j, m] - halo_radius[k])*heaviside(halo_radius[k] - R_cocoon_lengths[i, j, m])
                        cocoon_filling_factors[i, j, k] = 2*cocoon_angles_filled/(4*np.pi)
                        shock_filling_factors[i, j, k] = 2*shock_angles_filled/(4*np.pi)
                        shock_shell_filling_factors[i, j, k] = 2*shock_shell_angles_filled/(4*np.pi)
        return cocoon_filling_factors, shock_filling_factors, shock_shell_filling_factors
    cocoon_filling_factors, shock_filling_factors, shock_shell_filling_factors = compute_filling_factors(R_cocoon_lengths, R_shock_lengths, solid_angles, halo_radius)
    # function to calculate the filling factor matrix of the bubble as it rises through each cylindrical band
    @njit
    def calc_bubble_filling_factor(r, z_min, z_max, R_minor, y_min):
        # angular condition #1
        lower_1 = np.nanmin([np.arcsin(y_min/r), np.pi/2])
        upper_1 = np.nanmin([np.arcsin(R_minor/r), np.pi/2])
        # angular condition #2
        lower_2 = np.nanmax([0, np.arccos(z_max/r)])
        upper_2 = np.nanmax([0, np.arccos(z_min/r)])
        # min, max angles
        theta_min = max(lower_1, lower_2)
        theta_max = min(upper_1, upper_2)
        # checking min <= max
        if theta_min <= theta_max:
            solid_angle = 2*np.pi*(np.cos(theta_min) - np.cos(theta_max))
            return 2*solid_angle/(4*np.pi)
        else:
            return 0
    # cylindrical propagation axis of the bubble and the bands it can fill
    prop_axis_bubble = halo_radius
    prop_bins = len(prop_axis_bubble)
    # axis of the rear of the bubble
    prop_axis_bubble_rear = np.zeros_like(prop_axis_bubble)
    prop_axis_bubble_rear[1:] = prop_axis_bubble[:-1]
    # filling factor matrix of the bubble as a function of r, z
    @njit(parallel=True)
    def compute_bubble_filling_factors_matrix(prop_axis_bubble, prop_axis_bubble_rear, R_shock_minor, R_shock_lengths_xval, R_shock_lengths_yval, halo_radius):
        band_widths_dz = np.subtract(prop_axis_bubble, prop_axis_bubble_rear)
        bubble_filling_factors_matrix = np.zeros((power_res, age_res, radius_bins, prop_bins))
        for i in prange(power_res):
            for j in range(age_res):
                for k in range(radius_bins):
                    for m in range(prop_bins):
                        z_min_val = prop_axis_bubble[m] - band_widths_dz[m]
                        z_max_val = prop_axis_bubble[m]
                        y_min_val = R_shock_lengths_yval[i, j, np.argmin(np.abs(R_shock_lengths_xval[i, j] - (z_min_val+z_max_val)/2))]
                        bubble_filling_factors_matrix[i, j, k, m] = calc_bubble_filling_factor(halo_radius[k], z_min_val, z_max_val, R_shock_minor[i, j], y_min_val)
        return bubble_filling_factors_matrix
    bubble_filling_factors_matrix = compute_bubble_filling_factors_matrix(prop_axis_bubble, prop_axis_bubble_rear, R_shock_minor, R_shock_lengths_xval, R_shock_lengths_yval, halo_radius)
    bubble_filling_factors_matrix[:, :, :, halo_radius > r2500] = 0
    # total filling factor of the bubble
    bubble_filling_factors = np.sum(bubble_filling_factors_matrix, axis=3)
    # effective cylindrical filling factor
    @njit(parallel=True)
    def compute_cylinder_filling_factor(R_shock_minor, halo_radius):
        cylinder_filling_factor = np.ones((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                h_index = np.searchsorted(halo_radius, R_shock_minor[i, j])
                cylinder_filling_factor[i, j, h_index:] = 1 - np.sqrt(1 - np.square(np.divide(R_shock_minor[i, j], halo_radius[h_index:])))
        return cylinder_filling_factor
    cylinder_filling_factor = compute_cylinder_filling_factor(R_shock_minor, halo_radius)
    # cluster cooling filling factor
    cooling_filling_factor = np.subtract(1, cylinder_filling_factor)

    ## Shocked shell heating rate profiles
    # internal energy of cocoon and shocked shell
    @njit(parallel=True)
    def compute_internal_energies(V_cocoon, V_shock_shell, cocoon_filling_factors, shock_shell_filling_factors, shock_pressure, iso_th_pressure_profile, halo_radius):
        U_cocoon = np.zeros((power_res, age_res))
        U_shock_shell = np.zeros((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                if V_cocoon[i, j] > 0:
                    th_pressure_over_V_cocoon = 2*np.pi*np.trapz(np.multiply(np.square(halo_radius), np.multiply(cocoon_filling_factors[i, j], iso_th_pressure_profile)))/V_cocoon[i, j]
                    th_pressure_over_V_shock_shell = 2*np.pi*np.trapz(np.multiply(np.square(halo_radius), np.multiply(shock_shell_filling_factors[i, j], iso_th_pressure_profile)))/V_shock_shell[i, j]
                    U_cocoon[i, j] = (1/(gamma-1))*(shock_pressure[i, j] - th_pressure_over_V_cocoon)*V_cocoon[i, j]
                    U_shock_shell[i, j] = (1/(gamma-1))*(shock_pressure[i, j] - th_pressure_over_V_shock_shell)*V_shock_shell[i, j]
        return U_cocoon, U_shock_shell
    U_cocoon, U_shock_shell = compute_internal_energies(V_cocoon, V_shock_shell, cocoon_filling_factors, shock_shell_filling_factors, shock_pressure, iso_th_pressure_profile, halo_radius)
    # shocked shell volumetric heating rate
    @njit(parallel=True)
    def compute_shock_shell_volumetric_heating_rate(log_active_age_vals, V_shock, U_shock_shell, duty_cycle):
        q_shock_shell = np.zeros((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                q_shock_shell[i, j] = np.sqrt(duty_cycle)*U_shock_shell[i, j]/(V_shock[i, j]*(10**log_active_age_vals[j]*yr))
        return q_shock_shell
    q_shock_shell = compute_shock_shell_volumetric_heating_rate(log_active_age_vals, V_shock, U_shock_shell, duty_cycle)
    # gravitational potential energy of the shocked region
    @njit(parallel=True)
    def compute_gravitational_potential_energy_of_shock(R_cocoon_lengths, R_shock_lengths, shock_filling_factors, solid_angles, grav_potential, gas_density_profile, halo_radius):
        # shock shell density profile
        shock_shell_density_profiles = np.zeros((power_res, age_res, angular_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                for k in range(angular_res):
                    index_cocoon = np.searchsorted(halo_radius, R_cocoon_lengths[i, j, k])
                    index_shock = np.searchsorted(halo_radius, R_shock_lengths[i, j, k])
                    # check the minimum cocoon size
                    if R_cocoon_lengths[i, j, k] < halo_radius[0]:
                        index_cocoon = 0
                    # check the shock is larger than the cocoon
                    if index_shock <= index_cocoon:
                        continue
                    mass_swept = solid_angles[k]*np.trapz(np.multiply(np.square(halo_radius[:index_shock+1]), gas_density_profile[:index_shock+1]), halo_radius[:index_shock+1])
                    radial_band = halo_radius[index_cocoon:index_shock+1]
                    norm = mass_swept/(solid_angles[k]*np.trapz(np.multiply(np.square(radial_band), gas_density_profile[index_cocoon:index_shock+1]), radial_band))
                    shock_shell_density_profiles[i, j, k, index_cocoon:index_shock+1] = norm*gas_density_profile[index_cocoon:index_shock+1]
        # gravitational potential energy of the shocked region initially
        U_shock_grav_initial = np.zeros((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                U_shock_grav_initial[i, j] = - 2*np.pi*np.trapz(np.multiply(np.multiply(np.square(halo_radius), shock_filling_factors[i, j]), np.multiply(gas_density_profile, grav_potential)), halo_radius)
        # change in gravitational potential energy of the shocked mass at end of active age
        U_shock_shells_grav = np.zeros((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                for k in range(angular_res):
                    U_shock_shells_grav[i, j] += - solid_angles[k]*np.trapz(np.multiply(np.square(halo_radius), np.multiply(shock_shell_density_profiles[i, j, k], grav_potential)), halo_radius)
        U_shock_shell_grav = np.subtract(U_shock_shells_grav, U_shock_grav_initial)
        return U_shock_shell_grav
    U_shock_shell_grav = compute_gravitational_potential_energy_of_shock(R_cocoon_lengths, R_shock_lengths, shock_filling_factors, solid_angles, grav_potential, gas_density_profile, halo_radius)

    ## Bubble heating rate profiles
    # cocoon rest mass
    Lorentz_factor = 5  ## from RAiSE
    cocoon_rest_mass = (10**log_Qjet_vals[:, None])*(10**log_active_age_vals[None, :]*yr)/((Lorentz_factor - 1)*c_light**2)
    # gravitational potential energy of the cocoon
    @njit(parallel=True)
    def compute_gravitational_potential_energy_of_cocoon(cocoon_rest_mass, R_cocoon, cocoon_filling_factors, grav_potential, gas_density_profile, halo_radius):
        cocoon_density_profile = np.zeros((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                norm = cocoon_rest_mass[i, j]/(2*np.pi*np.trapz(np.multiply(np.multiply(np.square(halo_radius), cocoon_filling_factors[i, j]), gas_density_profile), halo_radius))
                cocoon_density_profile[i, j, 0:np.argmin(np.abs(R_cocoon[i, j] - halo_radius))+1] = norm*gas_density_profile[0:np.argmin(np.abs(R_cocoon[i, j] - halo_radius))+1]
        # cocoon density contrast
        cocoon_density_contrast = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                cocoon_density_contrast[i, j] = np.subtract(gas_density_profile, cocoon_density_profile[i, j])
        # gravitational potential energy of the bubble at end of active age
        U_bubble_grav = np.empty((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                U_bubble_grav[i, j] = - 2*np.pi*np.trapz(np.multiply(np.multiply(np.square(halo_radius), cocoon_filling_factors[i, j]), np.multiply(cocoon_density_contrast[i, j], grav_potential)), halo_radius)
        return U_bubble_grav
    U_bubble_grav = compute_gravitational_potential_energy_of_cocoon(cocoon_rest_mass, R_cocoon, cocoon_filling_factors, grav_potential, gas_density_profile, halo_radius)
    # total bubble energy
    E_bubble = np.add(U_bubble_grav, U_cocoon)
    # associated bubble power (x2 bubbles per duty cycle)
    Q_bubble = np.empty((power_res, age_res))
    for i in prange(power_res):
        for j in range(age_res):
            Q_bubble[i, j] = E_bubble[i, j]/((10**log_active_age_vals[j])*yr/duty_cycle)
    # axis to calculate bubble properties
    prop_axis_for_bubble_properties = R_shock_minor[:, :, None] + prop_axis_bubble_rear[None, None, :]
    # thermal pressure profiles along this axis
    th_pressure_profile_along_prop_axis = np.empty((power_res, age_res, prop_bins))
    th_pressure_derivative_along_prop_axis = np.empty((power_res, age_res, prop_bins))
    for i in range(power_res):
        for j in range(age_res):
            for k in range(prop_bins):
                th_pressure_profile_along_prop_axis[i, j, k] = Perseus_th_pressure(prop_axis_for_bubble_properties[i, j, k])
                th_pressure_derivative_along_prop_axis[i, j, k] = Perseus_th_pressure_derivative(prop_axis_for_bubble_properties[i, j, k])
    th_pressure_log_slope_along_prop_axis = np.multiply(th_pressure_derivative_along_prop_axis, np.divide(prop_axis_for_bubble_properties, th_pressure_profile_along_prop_axis))
    @njit(parallel=True)
    def compute_bubble_volumetric_heating_rate(prop_axis_for_bubble_properties, th_pressure_profile_along_prop_axis, th_pressure_log_slope_along_prop_axis, Q_bubble, bubble_filling_factors_matrix, bubble_filling_factors, halo_radius):
        # bubble volumetric heating rate function along the propagation axis
        q_bubble_func = np.empty((power_res, age_res, prop_bins))
        for i in prange(power_res):
            for j in range(age_res):
                q_bubble_func[i, j] = - np.multiply(np.divide(np.power(th_pressure_profile_along_prop_axis[i, j], ((gamma-1)/gamma)), np.square(prop_axis_for_bubble_properties[i, j])), np.subtract(((gamma-1)/gamma)*th_pressure_log_slope_along_prop_axis[i, j], 1))
        # normalising the bubble volumetric heating rate over the spherical volume containing bubble pair ## query minimum ##
        q_bubble_norm = np.empty((power_res, age_res))
        for i in prange(power_res):
            for j in range(age_res):
                q_bubble_norm[i, j] = 2*np.pi*np.trapz(np.multiply(np.square(halo_radius), bubble_filling_factors_matrix[i, j]@q_bubble_func[i, j]), halo_radius)
        q_bubble_along_prop_axis = np.empty((power_res, age_res, prop_bins))
        for i in prange(power_res):
            for j in range(age_res):
                q_bubble_along_prop_axis[i, j] = Q_bubble[i, j]*q_bubble_func[i, j]/q_bubble_norm[i, j]
        # bubble volumetric heating rate (with contributions from each band)
        q_bubble = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                q_bubble[i, j] = np.divide(bubble_filling_factors_matrix[i, j]@q_bubble_along_prop_axis[i, j], bubble_filling_factors[i, j])
        # squared bubble volumetric heating rate (with contributions from each band)
        q_bubble_squared = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                q_bubble_squared[i, j] = np.divide(bubble_filling_factors_matrix[i, j]@np.square(q_bubble_along_prop_axis[i, j]), bubble_filling_factors[i, j])
        return q_bubble, q_bubble_squared
    q_bubble, q_bubble_squared = compute_bubble_volumetric_heating_rate(prop_axis_for_bubble_properties, th_pressure_profile_along_prop_axis, th_pressure_log_slope_along_prop_axis, Q_bubble, bubble_filling_factors_matrix, bubble_filling_factors, halo_radius)
    # removing NaNs
    q_bubble[np.isnan(q_bubble)] = 0
    q_bubble_squared[np.isnan(q_bubble_squared)] = 0

    ## Effective heating rate profiles
    # volumetrically-weighted heating and cooling rate profiles
    @njit(parallel=True)
    def compute_heating_and_cooling_rates(cooling_rate, q_shock_shell, q_bubble_squared, cooling_filling_factor, shock_filling_factors, bubble_filling_factors):
        cooling_injection_rate = np.empty((power_res, age_res, radius_bins))
        shock_energy_injection_rate = np.empty((power_res, age_res, radius_bins))
        bubble_energy_injection_rate = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                cooling_injection_rate[i, j] = np.sqrt(np.multiply(cooling_filling_factor[i, j], np.square(cooling_rate)))
                shock_energy_injection_rate[i, j] = np.sqrt(np.multiply(shock_filling_factors[i, j], np.square(q_shock_shell[i, j])))
                bubble_energy_injection_rate[i, j] = np.sqrt(np.multiply(bubble_filling_factors[i, j], q_bubble_squared[i, j]))
        return cooling_injection_rate, shock_energy_injection_rate, bubble_energy_injection_rate
    cooling_injection_rate, shock_energy_injection_rate, bubble_energy_injection_rate = compute_heating_and_cooling_rates(cooling_rate, q_shock_shell, q_bubble_squared, cooling_filling_factor, shock_filling_factors, bubble_filling_factors)

    ## Kinetic feedback profiles
    # effective volumetric heating rate
    @njit(parallel=True)
    def compute_Q_eff(cooling_injection_rate, shock_energy_injection_rate, bubble_energy_injection_rate):
        effective_energy_injection_rate = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                effective_energy_injection_rate[i, j] = np.sqrt(np.add(np.square(cooling_injection_rate[i, j]), np.add(np.square(shock_energy_injection_rate[i, j]), np.square(bubble_energy_injection_rate[i, j])))) # W m^-3
        return effective_energy_injection_rate
    effective_energy_injection_rate = compute_Q_eff(cooling_injection_rate, shock_energy_injection_rate, bubble_energy_injection_rate)
    # velocity kick in the core
    @njit(parallel=True)
    def compute_v_kick(velocity_per_volumetric_energy_rate, effective_energy_injection_rate):
        velocity_kick_in_core = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                velocity_kick_in_core[i, j] = np.multiply(velocity_per_volumetric_energy_rate, effective_energy_injection_rate[i, j]) # m s^-1
        return velocity_kick_in_core
    velocity_kick_in_core = compute_v_kick(velocity_per_volumetric_energy_rate, effective_energy_injection_rate)
    # NTP fraction in the core
    @njit(parallel=True)
    def compute_NTP_fraction(th_velocity_profile, velocity_kick_in_core):
        NTP_fraction_in_core = np.empty((power_res, age_res, radius_bins))
        for i in prange(power_res):
            for j in range(age_res):
                NTP_fraction_in_core[i, j] = np.divide(np.square(velocity_kick_in_core[i, j]), np.add(np.square(velocity_kick_in_core[i, j]), np.square(th_velocity_profile)))
        return NTP_fraction_in_core
    NTP_fraction_in_core = compute_NTP_fraction(th_velocity_profile, velocity_kick_in_core)

    ## SAVING THE DATA
    # setting the descriptor for output files
    if len(log_active_age_vals) == 1:
        if len(log_Qjet_vals) == 1:
            descriptor = 'T_'+ str(log_active_age_vals.item()) + '_Q_' + str(log_Qjet_vals.item())
        elif len(log_Qjet_vals) > 1:
            descriptor = 'T_' + str(log_active_age_vals.item()) + '_Q_' + str(np.min(log_Qjet_vals)) + '_' + str(np.max(log_Qjet_vals))
    elif len(log_active_age_vals) > 1:
        if len(log_Qjet_vals) == 1:
            descriptor = 'T_'+ str(np.min(log_active_age_vals)) + '_' + str(np.max(log_active_age_vals)) + '_Q_' + str(log_Qjet_vals.item())
        elif len(log_Qjet_vals) > 1:
            descriptor = 'T_' + str(np.min(log_active_age_vals)) + '_' + str(np.max(log_active_age_vals)) + '_Q_' + str(np.min(log_Qjet_vals)) + '_' + str(np.max(log_Qjet_vals))
    # saving output files
    save_nested_arrays("R_cocoon_" + descriptor + ".txt", R_cocoon)
    save_nested_arrays("R_shock_" + descriptor + ".txt", R_shock)
    save_nested_arrays("Q_eff_" + descriptor + ".txt", effective_energy_injection_rate)
    save_nested_arrays("v_kick_" + descriptor + ".txt", velocity_kick_in_core)
    save_nested_arrays("NTP_fraction_" + descriptor + ".txt", NTP_fraction_in_core)

# active ages
log_active_age_N = 6.44
log_active_age_S = 6.16
# duty cycle
duty_cycle = 0.30
# cavity jet powers
log_Qjet_N_median = np.round(np.log10(6.18*10**37), 2)
log_Qjet_S_median = np.round(np.log10(2.21*10**37), 2)
log_Qjet_N_sigma_hi, log_Qjet_N_sigma_lo = np.round(np.log10((6.18 + 1.15)/6.18), 2), np.round(np.log10(6.18/(6.18 - 0.18)), 2)
log_Qjet_S_sigma_hi, log_Qjet_S_sigma_lo = np.round(np.log10((2.21 + 0.63)/2.21), 2), np.round(np.log10(2.21/(2.21 - 0.12)), 2)
# spanning jet powers
log_Qjet_N_dist = np.round(np.arange(log_Qjet_N_median - 10*log_Qjet_N_sigma_lo, log_Qjet_N_median + 10*log_Qjet_N_sigma_hi, 0.01), 2)
log_Qjet_S_dist = np.round(np.arange(log_Qjet_S_median - 10*log_Qjet_S_sigma_lo, log_Qjet_S_median + 10*log_Qjet_S_sigma_hi, 0.01), 2)



## Calling the model function
# simulation_specification: 0, 1, 2 = N, S, CCA
# func_calculate_feedback(log_Qjet_vals=log_Qjet_N_dist, log_active_age_vals=log_active_age_N, duty_cycle=duty_cycle, redshift=redshift, gas_density_profile=gas_density_profile, temperature_profile=temperature_profile, halo_radius=halo_radius, log_Qjet=0.01, log_dt=0.01)
# print('Done running N-cavity profiles')
# func_calculate_feedback(log_Qjet_vals=log_Qjet_S_dist, log_active_age_vals=log_active_age_S, duty_cycle=duty_cycle, redshift=redshift, gas_density_profile=gas_density_profile, temperature_profile=temperature_profile, halo_radius=halo_radius, log_Qjet=0.01, log_dt=0.01)
# print('Done running S-cavity profiles')
# func_calculate_feedback(2)
# print('Done running CCA distribution profiles')




# sample size
sample_size = 50000
# sampling the (N) jet powers
z_vals_N = np.random.normal(size=sample_size)
log_Qjet_N_samples = np.where(z_vals_N < 0, log_Qjet_N_median + z_vals_N*log_Qjet_N_sigma_lo, log_Qjet_N_median + z_vals_N*log_Qjet_N_sigma_hi)
# frequency each (N) jet power is sampled
log_Qjet_N_samples_indices = np.searchsorted(log_Qjet_N_dist, log_Qjet_N_samples)
log_Qjet_N_frequencies = np.bincount(log_Qjet_N_samples_indices, minlength=len(log_Qjet_N_dist))

# sampling the (S) jet powers
z_vals_S = np.random.normal(size=sample_size)
log_Qjet_S_samples = np.where(z_vals_S < 0, log_Qjet_S_median + z_vals_S*log_Qjet_S_sigma_lo, log_Qjet_S_median + z_vals_S*log_Qjet_S_sigma_hi)
# frequency each (S) jet power is sampled
log_Qjet_S_samples_indices = np.searchsorted(log_Qjet_S_dist, log_Qjet_S_samples)
log_Qjet_S_frequencies = np.bincount(log_Qjet_S_samples_indices, minlength=len(log_Qjet_S_dist))


## Opening saved data for N/S-aligned cavities
# descriptors
N_descriptor = 'T_' + str(log_active_age_N) + '_Q_' + str(np.min(log_Qjet_N_dist)) + '_' + str(np.max(log_Qjet_N_dist))
S_descriptor = 'T_' + str(log_active_age_S) + '_Q_' + str(np.min(log_Qjet_S_dist)) + '_' + str(np.max(log_Qjet_S_dist))
# cocoon sizes
R_cocoon_N_array, R_cocoon_S_array = read_file_to_nested_array('R_cocoon_' + N_descriptor + '.txt'), read_file_to_nested_array('R_cocoon_' + S_descriptor +'.txt')
R_cocoon_N_median, R_cocoon_S_median = weighted_percentile(R_cocoon_N_array[:, :, None].ravel(), 50, log_Qjet_N_frequencies.ravel()), weighted_percentile(R_cocoon_S_array[:, :, None].ravel(), 50, log_Qjet_S_frequencies.ravel())
# shock cocoon sizes
R_shock_N_array, R_shock_S_array = read_file_to_nested_array('R_shock_' + N_descriptor +'.txt'), read_file_to_nested_array('R_shock_' + S_descriptor +'.txt')
R_shock_N_median, R_shock_S_median = weighted_percentile(R_shock_N_array[:, :, None].ravel(), 50, log_Qjet_N_frequencies.ravel()), weighted_percentile(R_shock_S_array[:, :, None].ravel(), 50, log_Qjet_S_frequencies.ravel())
# average size of N/S cavities
R_cocoon_av_median = np.mean([R_cocoon_N_median, R_cocoon_S_median])
R_shock_av_median = np.mean([R_shock_N_median, R_shock_S_median])
# effective energy injections
Q_eff_N_array, Q_eff_S_array = read_file_to_nested_array('Q_eff_' + N_descriptor +'.txt'), read_file_to_nested_array('Q_eff_' + S_descriptor +'.txt')
Q_eff_N_mean, Q_eff_S_mean = np.sum(Q_eff_N_array*log_Qjet_N_frequencies[:, None], axis=0)/np.sum(log_Qjet_N_frequencies, axis=0), np.sum(Q_eff_S_array*log_Qjet_S_frequencies[:, None], axis=0)/np.sum(log_Qjet_S_frequencies, axis=0)
Q_eff_N_p16, Q_eff_N_median, Q_eff_N_p84, Q_eff_S_p16, Q_eff_S_median, Q_eff_S_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
for i in range(radius_bins):
    Q_eff_N_p16[i], Q_eff_N_median[i], Q_eff_N_p84[i] = weighted_percentile(Q_eff_N_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_N_frequencies.ravel())
    Q_eff_S_p16[i], Q_eff_S_median[i], Q_eff_S_p84[i] = weighted_percentile(Q_eff_S_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_S_frequencies.ravel())
# velocity kicks
v_kick_N_array, v_kick_S_array = read_file_to_nested_array('v_kick_' + N_descriptor +'.txt'), read_file_to_nested_array('v_kick_' + S_descriptor +'.txt')
v_kick_N_mean, v_kick_S_mean = np.sum(v_kick_N_array*log_Qjet_N_frequencies[:, None], axis=0)/np.sum(log_Qjet_N_frequencies, axis=0), np.sum(v_kick_S_array*log_Qjet_S_frequencies[:, None], axis=0)/np.sum(log_Qjet_S_frequencies, axis=0)
v_kick_N_p16, v_kick_N_median, v_kick_N_p84, v_kick_S_p16, v_kick_S_median, v_kick_S_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
for i in range(radius_bins):
    v_kick_N_p16[i], v_kick_N_median[i], v_kick_N_p84[i] = weighted_percentile(v_kick_N_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_N_frequencies.ravel())
    v_kick_S_p16[i], v_kick_S_median[i], v_kick_S_p84[i] = weighted_percentile(v_kick_S_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_S_frequencies.ravel())
# NTP fractions
NTP_fraction_N_array, NTP_fraction_S_array = read_file_to_nested_array('NTP_fraction_' + N_descriptor +'.txt'), read_file_to_nested_array('NTP_fraction_' + S_descriptor +'.txt')
NTP_fraction_N_mean, NTP_fraction_S_mean = np.sum(NTP_fraction_N_array*log_Qjet_N_frequencies[:, None], axis=0)/np.sum(log_Qjet_N_frequencies, axis=0), np.sum(NTP_fraction_S_array*log_Qjet_S_frequencies[:, None], axis=0)/np.sum(log_Qjet_S_frequencies, axis=0)
NTP_fraction_N_p16, NTP_fraction_N_median, NTP_fraction_N_p84, NTP_fraction_S_p16, NTP_fraction_S_median, NTP_fraction_S_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
for i in range(radius_bins):
    NTP_fraction_N_p16[i], NTP_fraction_N_median[i], NTP_fraction_N_p84[i] = weighted_percentile(NTP_fraction_N_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_N_frequencies.ravel())
    NTP_fraction_S_p16[i], NTP_fraction_S_median[i], NTP_fraction_S_p84[i] = weighted_percentile(NTP_fraction_S_array[:, :, i].ravel(), [16, 50, 84], log_Qjet_S_frequencies.ravel())
# average AGN profile of N/S cavities
v_kick_av_p16, v_kick_av_median, v_kick_av_p84 = np.average([v_kick_N_p16, v_kick_S_p16], axis=0), np.average([v_kick_N_median, v_kick_S_median], axis=0), np.average([v_kick_N_p84, v_kick_S_p84], axis=0)
NTP_fraction_av_p16, NTP_fraction_av_median, NTP_fraction_av_p84 = np.average([NTP_fraction_N_p16, NTP_fraction_S_p16], axis=0), np.average([NTP_fraction_N_median, NTP_fraction_S_median], axis=0), np.average([NTP_fraction_N_p84, NTP_fraction_S_p84], axis=0)
# properties of the N/S cavity NTP fraction profiles
print('The median cocoon length in the N cavity = ', np.round(R_cocoon_N_median/kpc, 2), ' kpc, and in the S cavity = ', np.round(R_cocoon_S_median/kpc, 2), ' kpc, with the median = ', np.round(R_cocoon_av_median/kpc, 2), ' kpc')
print('The median shock length in the N cavity = ', np.round(R_shock_N_median/kpc, 2), ' kpc, and in the S cavity = ', np.round(R_shock_S_median/kpc, 2), ' kpc, with the median = ', np.round(R_shock_av_median/kpc, 2), ' kpc')
NTP_fraction_N_peak, NTP_fraction_S_peak, NTP_fraction_median_peak = np.round(100*np.nanmax(NTP_fraction_N_median[halo_radius < r2500]), 2), np.round(100*np.nanmax(NTP_fraction_S_median[halo_radius < r2500]), 2), np.round(100*np.nanmax(NTP_fraction_av_median[halo_radius < r2500]), 2)
print('The median NTP fraction peak in N cavity = ', NTP_fraction_N_peak, ' %; and in S cavity = ', NTP_fraction_S_peak, ' %; with the median = ', NTP_fraction_median_peak, ' %')
NTP_fraction_N_at_cocoon, NTP_fraction_S_at_cocoon, NTP_fraction_median_at_cocoon = np.round(100*(NTP_fraction_N_median[np.argmin(np.abs(R_cocoon_N_median - halo_radius))]), 2), np.round(100*(NTP_fraction_S_median[np.argmin(np.abs(R_cocoon_S_median - halo_radius))]), 2), np.round(100*(NTP_fraction_av_median[np.argmin(np.abs(R_cocoon_av_median - halo_radius))]), 2)
print('The median NTP fraction at the edge of the N cocoon = ', NTP_fraction_N_at_cocoon, ' %; and at the edge of the S cocoon = ', NTP_fraction_S_at_cocoon, ' %; with the median = ', NTP_fraction_median_at_cocoon, ' %')
NTP_fraction_N_at_shock, NTP_fraction_S_at_shock, NTP_fraction_median_at_shock = np.round(100*(NTP_fraction_N_median[np.argmin(np.abs(R_shock_N_median - halo_radius))]), 2), np.round(100*(NTP_fraction_S_median[np.argmin(np.abs(R_shock_S_median - halo_radius))]), 2), np.round(100*(NTP_fraction_av_median[np.argmin(np.abs(R_shock_av_median - halo_radius))]), 2)
print('The median NTP fraction at the edge of the N shock = ', NTP_fraction_N_at_shock, ' %; and at the edge of the S shock = ', NTP_fraction_S_at_shock, ' %; with the median = ', NTP_fraction_median_at_shock, ' %')
radius_NTP_fraction_N_to_zero, radius_NTP_fraction_S_to_zero = halo_radius[np.where(NTP_fraction_N_median == NTP_fraction_N_median[halo_radius > R_cocoon_N_median][NTP_fraction_N_median[halo_radius > R_cocoon_N_median] < 0.005][0])][0], halo_radius[np.where(NTP_fraction_S_median == NTP_fraction_S_median[halo_radius > R_cocoon_S_median][NTP_fraction_S_median[halo_radius > R_cocoon_S_median] < 0.005][0])][0]
radius_NTP_fraction_median_to_zero = halo_radius[np.where(NTP_fraction_av_median == NTP_fraction_av_median[halo_radius > R_cocoon_av_median][NTP_fraction_av_median[halo_radius > R_cocoon_av_median] < 0.005][0])][0]
print('The radius in which the median NTP fraction profile goes to zero in N cavity = ', np.round(radius_NTP_fraction_N_to_zero/kpc, 2), ' kpc; and in S cavity = ', np.round(radius_NTP_fraction_S_to_zero/kpc, 2), ' kpc; with the median = ', np.round(radius_NTP_fraction_median_to_zero/kpc, 2), ' kpc')

# ## Opening saved data for CCA distribution
# # cocoon sizes
# R_cocoon_CCA_dist_array = read_file_to_nested_array('R_cocoon_CCA_dist.txt')
# R_cocoon_CCA_dist_mean = np.average(R_cocoon_CCA_dist_array.ravel(), weights=CCA_sample_weights_array.ravel(), axis=0)
# R_cocoon_CCA_dist_median = weighted_percentile(R_cocoon_CCA_dist_array[:, :, None].ravel(), 50, CCA_sample_weights_array.ravel())
# # shock cocoon sizes
# R_shock_CCA_dist_array = read_file_to_nested_array('R_shock_CCA_dist.txt')
# R_shock_CCA_dist_mean = np.average(R_shock_CCA_dist_array.ravel(), weights=CCA_sample_weights_array.ravel(), axis=0)
# R_shock_CCA_dist_median = weighted_percentile(R_shock_CCA_dist_array[:, :, None].ravel(), 50, CCA_sample_weights_array.ravel())
# # effective energy injections
# Q_eff_CCA_dist_array = read_file_to_nested_array('Q_eff_CCA_dist.txt')
# Q_eff_CCA_dist_mean = np.sum(Q_eff_CCA_dist_array*CCA_sample_weights_array[:, :, None], axis=(0, 1))/np.sum(CCA_sample_weights_array)
# Q_eff_CCA_dist_p16, Q_eff_CCA_dist_median, Q_eff_CCA_dist_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
# for i in range(radius_bins):
#     Q_eff_CCA_dist_p16[i], Q_eff_CCA_dist_median[i], Q_eff_CCA_dist_p84[i] = weighted_percentile(Q_eff_CCA_dist_array[:, :, i].ravel(), [16, 50, 84], CCA_sample_weights_array.ravel())
# # velocity kicks
# v_kick_CCA_dist_array = read_file_to_nested_array('v_kick_CCA_dist.txt')
# v_kick_CCA_dist_mean = np.sum(v_kick_CCA_dist_array*CCA_sample_weights_array[:, :, None], axis=(0, 1))/np.sum(CCA_sample_weights_array)
# v_kick_CCA_dist_p16, v_kick_CCA_dist_median, v_kick_CCA_dist_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
# for i in range(radius_bins):
#     v_kick_CCA_dist_p16[i], v_kick_CCA_dist_median[i], v_kick_CCA_dist_p84[i] = weighted_percentile(v_kick_CCA_dist_array[:, :, i].ravel(), [16, 50, 84], CCA_sample_weights_array.ravel())
# # NTP fractions
# NTP_fraction_CCA_dist_array = read_file_to_nested_array('NTP_fraction_CCA_dist.txt')
# NTP_fraction_CCA_dist_mean = np.sum(NTP_fraction_CCA_dist_array*CCA_sample_weights_array[:, :, None], axis=(0, 1))/np.sum(CCA_sample_weights_array)
# NTP_fraction_CCA_dist_p16, NTP_fraction_CCA_dist_median, NTP_fraction_CCA_dist_p84 = np.empty(radius_bins), np.empty(radius_bins), np.empty(radius_bins)
# for i in range(radius_bins):
#     NTP_fraction_CCA_dist_p16[i], NTP_fraction_CCA_dist_median[i], NTP_fraction_CCA_dist_p84[i] = weighted_percentile(NTP_fraction_CCA_dist_array[:, :, i].ravel(), [16, 50, 84], CCA_sample_weights_array.ravel())
# # properties of the mean CCA NTP fraction profile
# print('The median CCA cocoon length = ', np.round(R_cocoon_CCA_dist_median/kpc, 2), ' kpc')
# print('The median CCA shock length = ', np.round(R_shock_CCA_dist_median/kpc, 2), ' kpc')
# NTP_fraction_CCA_dist_median_peak = np.round(100*np.nanmax(NTP_fraction_CCA_dist_median), 2)
# print('The median CCA NTP fraction peak = ', NTP_fraction_CCA_dist_median_peak, ' %')
# NTP_fraction_CCA_dist_median_at_cocoon = np.round(100*(NTP_fraction_CCA_dist_median[np.argmin(np.abs(R_cocoon_CCA_dist_median - halo_radius))]), 2)
# print('The median CCA NTP fraction at the edge of the cocoon = ', NTP_fraction_CCA_dist_median_at_cocoon, ' %')
# NTP_fraction_CCA_dist_median_at_shock = np.round(100*(NTP_fraction_CCA_dist_median[np.argmin(np.abs(R_shock_CCA_dist_median - halo_radius))]), 2)
# print('The median CCA NTP fraction at the edge of the shock = ', NTP_fraction_CCA_dist_median_at_shock, ' %')
# radius_NTP_fraction_CCA_dist_median_to_zero = halo_radius[np.where(NTP_fraction_CCA_dist_median == NTP_fraction_CCA_dist_median[halo_radius > R_cocoon_CCA_dist_median][NTP_fraction_CCA_dist_median[halo_radius > R_cocoon_CCA_dist_median] < 0.005][0])][0]
# print('The radius in which the median CCA NTP fraction profile goes to zero = ', np.round(radius_NTP_fraction_CCA_dist_median_to_zero/kpc, 2), ' kpc')

## Total gas velocity kick profiles
# total median gas velocity kick profiles (outskirts + choice of AGN model)
v_kick_total_N_median = np.add(v_kick_N_median, v_kick_outskirts)
v_kick_total_S_median = np.add(v_kick_S_median, v_kick_outskirts)
# v_kick_total_av_median = np.add(v_kick_av_median, v_kick_outskirts)
# v_kick_total_CCA_dist_median = np.add(v_kick_CCA_dist_median, v_kick_outskirts)
# total p16, p84 gas velocity kick profiles (outskirts + choice of AGN model)
v_kick_total_N_p16, v_kick_total_N_p84 = np.add(v_kick_N_p16, v_kick_outskirts_min), np.add(v_kick_N_p84, v_kick_outskirts_max)
v_kick_total_S_p16, v_kick_total_S_p84 = np.add(v_kick_S_p16, v_kick_outskirts_min), np.add(v_kick_S_p84, v_kick_outskirts_max)
# v_kick_total_av_p16, v_kick_total_av_p84 = np.add(v_kick_av_p16, v_kick_outskirts_min), np.add(v_kick_av_p84, v_kick_outskirts_max)
# v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_p84 = np.add(v_kick_CCA_dist_p16, v_kick_outskirts_min), np.add(v_kick_CCA_dist_p84, v_kick_outskirts_max)

## Total NTP fraction profiles
# total median NTP fraction profiles (outskirts + choice of AGN model)
NTP_fraction_total_N_median = np.divide(np.square(v_kick_total_N_median), np.add(np.square(v_kick_total_N_median), np.square(th_velocity_profile)))
NTP_fraction_total_S_median = np.divide(np.square(v_kick_total_S_median), np.add(np.square(v_kick_total_S_median), np.square(th_velocity_profile)))
# NTP_fraction_total_av_median = np.divide(np.square(v_kick_total_av_median), np.add(np.square(v_kick_total_av_median), np.square(th_velocity_profile)))
# NTP_fraction_total_CCA_dist_median = np.divide(np.square(v_kick_total_CCA_dist_median), np.add(np.square(v_kick_total_CCA_dist_median), np.square(th_velocity_profile)))
# total p16, p84 NTP fraction profiles (outskirts + choice of AGN model)
NTP_fraction_total_N_p16, NTP_fraction_total_N_p84 = np.divide(np.square(v_kick_total_N_p16), np.add(np.square(v_kick_total_N_p16), np.square(th_velocity_profile))), np.divide(np.square(v_kick_total_N_p84), np.add(np.square(v_kick_total_N_p84), np.square(th_velocity_profile)))
NTP_fraction_total_S_p16, NTP_fraction_total_S_p84 = np.divide(np.square(v_kick_total_S_p16), np.add(np.square(v_kick_total_S_p16), np.square(th_velocity_profile))), np.divide(np.square(v_kick_total_S_p84), np.add(np.square(v_kick_total_S_p84), np.square(th_velocity_profile)))
# NTP_fraction_total_av_p16, NTP_fraction_total_av_p84 = np.divide(np.square(v_kick_total_av_p16), np.add(np.square(v_kick_total_av_p16), np.square(th_velocity_profile))), np.divide(np.square(v_kick_total_av_p84), np.add(np.square(v_kick_total_av_p84), np.square(th_velocity_profile)))
# NTP_fraction_total_CCA_dist_p16, NTP_fraction_total_CCA_dist_p84 = np.divide(np.square(v_kick_total_CCA_dist_p16), np.add(np.square(v_kick_total_CCA_dist_p16), np.square(th_velocity_profile))), np.divide(np.square(v_kick_total_CCA_dist_p84), np.add(np.square(v_kick_total_CCA_dist_p84), np.square(th_velocity_profile)))

# ## Hi/lo jet power and age profiles
# # log age steps indexed for CCA dist
# log_active_age_CCA_dist_vals = log_age_steps_lo_res[np.searchsorted(log_age_steps_lo_res, log_active_age_CCA_dist)]
# log_active_age_CCA_dist_median_index = np.searchsorted(log_active_age_CCA_dist_vals, np.round(log_active_age_median, 1))
# # jet power +/- sigma
# v_kick_hi_Qjet = np.add(v_kick_CCA_dist_array[np.searchsorted(log_Qjet_CCA_dist, np.round(log_Qjet_median + log_Qjet_CCA_sigma, 2)), log_active_age_CCA_dist_median_index, :], v_kick_outskirts)
# NTP_fraction_hi_Qjet = np.divide(np.square(v_kick_hi_Qjet), np.add(np.square(v_kick_hi_Qjet), np.square(th_velocity_profile)))
# v_kick_lo_Qjet = np.add(v_kick_CCA_dist_array[np.searchsorted(log_Qjet_CCA_dist, np.round(log_Qjet_median - log_Qjet_CCA_sigma, 2)), log_active_age_CCA_dist_median_index, :], v_kick_outskirts)
# NTP_fraction_lo_Qjet = np.divide(np.square(v_kick_lo_Qjet), np.add(np.square(v_kick_lo_Qjet), np.square(th_velocity_profile)))
# # active age +/- sigma
# v_kick_hi_active_age = np.add(v_kick_CCA_dist_array[log_Qjet_CCA_dist_median_index, np.searchsorted(log_active_age_CCA_dist_vals, np.round(log_active_age_median + log_active_age_sigma, 1)), :], v_kick_outskirts)
# NTP_fraction_hi_active_age = np.divide(np.square(v_kick_hi_active_age), np.add(np.square(v_kick_hi_active_age), np.square(th_velocity_profile)))
# v_kick_lo_active_age = np.add(v_kick_CCA_dist_array[log_Qjet_CCA_dist_median_index, np.searchsorted(log_active_age_CCA_dist_vals, np.round(log_active_age_median - log_active_age_sigma, 1)), :], v_kick_outskirts)
# NTP_fraction_lo_active_age = np.divide(np.square(v_kick_lo_active_age), np.add(np.square(v_kick_lo_active_age), np.square(th_velocity_profile)))


R_transition_N = halo_radius[halo_radius >= R_shock_N_median][np.argmin(np.add(NTP_fraction_N_median[halo_radius >= R_shock_N_median], NTP_fraction_outskirts[halo_radius >= R_shock_N_median]))]
R_transition_S = halo_radius[halo_radius >= R_shock_S_median][np.argmin(np.add(NTP_fraction_S_median[halo_radius >= R_shock_S_median], NTP_fraction_outskirts[halo_radius >= R_shock_S_median]))]
#R_transition_CCA = halo_radius[halo_radius >= R_shock_CCA_dist_median][np.argmin(np.add(NTP_fraction_CCA_dist_median[halo_radius >= R_shock_CCA_dist_median], NTP_fraction_outskirts[halo_radius >= R_shock_CCA_dist_median]))]

print('Feedback transition occurs at ', np.round(R_transition_N/kpc, 2), ' kpc in the N direction')
print('Feedback transition occurs at ', np.round(R_transition_S/kpc, 2), ' kpc in the S direction')
#print('Feedback transition occurs at ', np.round(R_transition_CCA/kpc, 2), ' kpc for the CCA median distribution')

## Plot showing literature comparisons of the NTP fraction profiles
# the XRISM observational constraints
NTP_XRISM_Perseus = 10**(-2)*np.asarray([5.0, 2.8, 0.6, 3.7, 1.3, 2.8])
NTP_XRISM_Perseus_upper_lims = 10**(-2)*np.asarray([6, 3.2, 0.9, 4.5, 1.9, 3.8])
NTP_XRISM_Perseus_lower_lims = 10**(-2)*np.asarray([4.2, 2.5, 0.4, 3, 0.8, 2])
NTP_XRISM_Perseus_radii = arcmin_to_kpc*np.asarray([9/60, 0.9, 2.5, 4.3, 6.1, 8.2, 11.8])
print(NTP_XRISM_Perseus_radii/kpc)
reg1 = np.logspace(-4, np.log10(NTP_XRISM_Perseus_radii[1]/r500))
reg2 = np.logspace(np.log10(NTP_XRISM_Perseus_radii[1]/r500), np.log10(NTP_XRISM_Perseus_radii[2]/r500))
reg3 = np.logspace(np.log10(NTP_XRISM_Perseus_radii[2]/r500), np.log10(NTP_XRISM_Perseus_radii[3]/r500))
reg4 = np.logspace(np.log10(NTP_XRISM_Perseus_radii[3]/r500), np.log10(NTP_XRISM_Perseus_radii[4]/r500))
reg5 = np.logspace(np.log10(NTP_XRISM_Perseus_radii[4]/r500), np.log10(NTP_XRISM_Perseus_radii[5]/r500))
reg6 = np.logspace(np.log10(NTP_XRISM_Perseus_radii[5]/r500), np.log10(NTP_XRISM_Perseus_radii[6]/r500))

sky_regions = {
    1: reg1*r500,
    2: reg2*r500,
    3: reg3*r500,
    4: reg4*r500,
    5: reg5*r500,
    6: reg6*r500
}


# sub-FOV detector regions
sky_1_pixels = [[1, 1], [1, 2], [2, 1], [-1, 1], [-1, 2], [-2, 1], [1, -1], [1, -2], [2, -1], [-1, -1], [-1, -2], [-2, -1]]
sky_2_pixels = [[1, 3], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3], [2, -2], [2, -3], [3, -1], [3, -2], [3, -3], [-1, 3], [-2, 2], [-2, 3], [-3, 1], [-3, 2], [-1, -3], [-2, -2], [-2, -3], [-3, -1], [-3, -2], [-3, -3], [3, 4], [4, 3], [4, 2], [4, 1], [5, 1]]
sky_3_pixels = [[3, 5], [4, 4], [4, 5], [4, 6], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [8, 1], [8, 2], [8, 3]]
sky_4_pixels = [[7, 6], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [10, 4], [10, 5], [10, 6], [10, 7], [11, 5]]
sky_5_pixels = [[9, 9], [10, 8], [10, 9], [11, 6], [11, 7], [11, 8], [11, 9], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [14, 7], [14, 8]]
sky_6_pixels = [[13, 10], [13, 11], [14, 9], [14, 10], [14, 11], [14, 12], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12]]

def calc_coordinate_aperture(x, y):
    pixel_width = 30*arcmin_to_kpc/60
    pixel_centre = pixel_width/2
    x_centre = pixel_width*x - np.sign(x)*pixel_centre
    y_centre = pixel_width*y - np.sign(y)*pixel_centre
    aperture_loc = np.linalg.norm([x_centre, y_centre], axis=0)
    return aperture_loc

sky_apertures = {
1: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_1_pixels]),
2: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_2_pixels]),
3: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_3_pixels]),
4: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_4_pixels]),
5: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_5_pixels]),
6: np.asarray([calc_coordinate_aperture(coord[0], coord[1]) for coord in sky_6_pixels])
}

pixel_width = 30*arcmin_to_kpc/60
pixel_centre = pixel_width/2


## X-ray emissivity weighting
# electron density profiles
electron_density_profile = np.divide(gas_density_profile, (mu_e*mp))  # m^-3
# loading cooling function
abundance_file = 'm-05.cie'  # metallicity for Sutherland & Dopita (1993) cooling function, here corresponding to Z = 0.3 Zsolar
# interpolating from the gas temperature to the cooling function
logT_cool, logLambda_cool = read_ds_cooling(abundance_file)  # T in K, Lambda 13 dex higher than SI value
logLambda_cool_interp = np.interp(np.log10(temperature_profile), logT_cool, logLambda_cool - 13)  # log ( W m^3 )
# interpolated cooling function
Lambda_cool_interp = np.asarray([10 ** i for i in logLambda_cool_interp])  # W m^3
# surface X-ray emissivity profile
Xray_emission_proportionality = np.multiply(np.square(electron_density_profile), Lambda_cool_interp)

## Observational comparison in XRISM regions
# function for calculating projected NTP
def calc_profile_in_XRISM_bins(bin, velocities, detector=True):
    velocities = np.nan_to_num(velocities)
    n_components = len(velocities)
    Xray_weighted_aperture_velocities_sq = np.empty(n_components)
    Xray_weighted_NTP_fractions = np.empty(n_components)
    # V1 - define XRISM bin apertures from its pixels
    if detector==True:
        # define XRISM bin apertures from its pixels
        sky_aperture_radius = sky_apertures[bin]
        sky_aperture_bins = len(sky_aperture_radius)
        # calculate the X-ray surface emissivity in each pixel
        surface_Xray_emissivity_profile = np.empty(sky_aperture_bins)
        for i in range(sky_aperture_bins):
            aperture_mask = (halo_radius > sky_aperture_radius[i]) & (halo_radius < r_trunc)
            surface_Xray_emissivity_profile[i] = 2*np.trapz(np.multiply(Xray_emission_proportionality[aperture_mask], np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])  # kg^2 m^-5
        # calculate the temperature (weighted by X-ray emissivity) in each pixel
        Xray_weighted_los_temperature_integral = np.empty(sky_aperture_bins)
        for i in range(sky_aperture_bins):
            aperture_mask = (halo_radius > sky_aperture_radius[i]) & (halo_radius < r_trunc)
            Xray_weighted_los_temperature_integral[i] = 2*np.trapz(np.multiply(np.multiply(Xray_emission_proportionality[aperture_mask], temperature_profile[aperture_mask]), np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])
        # calculate the los velocity dispersion (weighted by X-ray emissivity) in each pixel
        Xray_weighted_los_velocity_sq_integrals = np.empty((n_components, sky_aperture_bins))
        for i in range(sky_aperture_bins):
            aperture_mask = (halo_radius > sky_aperture_radius[i]) & (halo_radius < r_trunc)
            for j in range(n_components):
                Xray_weighted_los_velocity_sq_integrals[j, i] = 2*np.trapz(np.multiply(np.multiply((1/3)*np.square(velocities[j][aperture_mask]), Xray_emission_proportionality[aperture_mask]), np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])
        # calculate the aperture average temperature
        Xray_weighted_aperture_temperature = np.sum(Xray_weighted_los_temperature_integral, axis=0)/np.sum(surface_Xray_emissivity_profile, axis=0)
        # calculate the aperture average los velocity dispersion
        for i in range(n_components):
            Xray_weighted_aperture_velocities_sq[i] = np.sum(Xray_weighted_los_velocity_sq_integrals[i], axis=0)/np.sum(surface_Xray_emissivity_profile, axis=0)
    # V1 - define XRISM bin apertures from its sky regions
    elif detector==False:
        # define XRISM bin apertures from its sky regions
        sky_aperture_radius = sky_regions[bin]
        sky_aperture_bins = len(sky_aperture_radius)
        # calculate the X-ray surface emissivity in each sky region
        surface_Xray_emissivity_profile = np.empty(sky_aperture_bins)
        for i in range(sky_aperture_bins):
            aperture_mask = halo_radius > sky_aperture_radius[i]
            surface_Xray_emissivity_profile[i] = 2*np.trapz(np.multiply(Xray_emission_proportionality[aperture_mask], np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])  # kg^2 m^-5
        # calculate the temperature (weighted by X-ray emissivity) in each sky region
        Xray_weighted_los_temperature_integral = np.empty(sky_aperture_bins)
        for i in range(sky_aperture_bins):
            aperture_mask = halo_radius > sky_aperture_radius[i]
            Xray_weighted_los_temperature_integral[i] = 2*np.trapz(np.multiply(np.multiply(Xray_emission_proportionality[aperture_mask], temperature_profile[aperture_mask]), np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])
        # calculate the los velocity dispersion (weighted by X-ray emissivity) in each sky region
        Xray_weighted_los_velocity_sq_integrals = np.empty((n_components, sky_aperture_bins))
        for i in range(sky_aperture_bins):
            aperture_mask = halo_radius > sky_aperture_radius[i]
            for j in range(n_components):
                Xray_weighted_los_velocity_sq_integrals[j, i] = 2*np.trapz(np.multiply(np.multiply((1/3)*np.square(velocities[j][aperture_mask]), Xray_emission_proportionality[aperture_mask]), np.divide(halo_radius[aperture_mask], np.sqrt(np.subtract(np.square(halo_radius[aperture_mask]), np.square(sky_aperture_radius[i]))))), halo_radius[aperture_mask])
        # calculate the aperture average temperature
        Xray_weighted_aperture_temperature = np.trapz(np.multiply(sky_aperture_radius, Xray_weighted_los_temperature_integral), sky_aperture_radius)/np.trapz(np.multiply(sky_aperture_radius, surface_Xray_emissivity_profile), sky_aperture_radius)
        # calculate the aperture average los velocity dispersion
        for i in range(n_components):
            Xray_weighted_aperture_velocities_sq[i] = np.trapz(np.multiply(sky_aperture_radius, Xray_weighted_los_velocity_sq_integrals[i]), sky_aperture_radius)/np.trapz(np.multiply(sky_aperture_radius, surface_Xray_emissivity_profile), sky_aperture_radius)
    # calculate the NTP fractions
    for i in range(n_components):
        Xray_weighted_NTP_fractions[i] = Xray_weighted_aperture_velocities_sq[i]/(Xray_weighted_aperture_velocities_sq[i] + kB*Xray_weighted_aperture_temperature/(mu*mp))
    return np.asarray(Xray_weighted_NTP_fractions)
# calculating the NTP fractions in each bin -- outskirts model
reg1_NTP_fraction_outskirts_min, reg1_NTP_fraction_outskirts_median, reg1_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(1, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
reg2_NTP_fraction_outskirts_min, reg2_NTP_fraction_outskirts_median, reg2_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(2, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
reg3_NTP_fraction_outskirts_min, reg3_NTP_fraction_outskirts_median, reg3_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(3, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
reg4_NTP_fraction_outskirts_min, reg4_NTP_fraction_outskirts_median, reg4_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(4, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
reg5_NTP_fraction_outskirts_min, reg5_NTP_fraction_outskirts_median, reg5_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(5, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
reg6_NTP_fraction_outskirts_min, reg6_NTP_fraction_outskirts_median, reg6_NTP_fraction_outskirts_max = calc_profile_in_XRISM_bins(6, [v_kick_outskirts_min, v_kick_outskirts, v_kick_outskirts_max], detector=True)
# calculating the NTP fractions in each bin -- N cavity model
reg1_NTP_fraction_total_N_p16, reg1_NTP_fraction_total_N_median, reg1_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(1, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
reg2_NTP_fraction_total_N_p16, reg2_NTP_fraction_total_N_median, reg2_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(2, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
reg3_NTP_fraction_total_N_p16, reg3_NTP_fraction_total_N_median, reg3_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(3, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
reg4_NTP_fraction_total_N_p16, reg4_NTP_fraction_total_N_median, reg4_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(4, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
reg5_NTP_fraction_total_N_p16, reg5_NTP_fraction_total_N_median, reg5_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(5, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
reg6_NTP_fraction_total_N_p16, reg6_NTP_fraction_total_N_median, reg6_NTP_fraction_total_N_p84 = calc_profile_in_XRISM_bins(6, [v_kick_total_N_p16, v_kick_total_N_median, v_kick_total_N_p84], detector=True)
# calculating the NTP fractions in each bin -- S cavity model
reg1_NTP_fraction_total_S_p16, reg1_NTP_fraction_total_S_median, reg1_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(1, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
reg2_NTP_fraction_total_S_p16, reg2_NTP_fraction_total_S_median, reg2_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(2, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
reg3_NTP_fraction_total_S_p16, reg3_NTP_fraction_total_S_median, reg3_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(3, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
reg4_NTP_fraction_total_S_p16, reg4_NTP_fraction_total_S_median, reg4_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(4, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
reg5_NTP_fraction_total_S_p16, reg5_NTP_fraction_total_S_median, reg5_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(5, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
reg6_NTP_fraction_total_S_p16, reg6_NTP_fraction_total_S_median, reg6_NTP_fraction_total_S_p84 = calc_profile_in_XRISM_bins(6, [v_kick_total_S_p16, v_kick_total_S_median, v_kick_total_S_p84], detector=True)
# # calculating the NTP fractions in each bin -- CCA distribution
# reg1_NTP_fraction_total_CCA_dist_p16, reg1_NTP_fraction_total_CCA_dist_median, reg1_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(1, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# reg2_NTP_fraction_total_CCA_dist_p16, reg2_NTP_fraction_total_CCA_dist_median, reg2_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(2, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# reg3_NTP_fraction_total_CCA_dist_p16, reg3_NTP_fraction_total_CCA_dist_median, reg3_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(3, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# reg4_NTP_fraction_total_CCA_dist_p16, reg4_NTP_fraction_total_CCA_dist_median, reg4_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(4, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# reg5_NTP_fraction_total_CCA_dist_p16, reg5_NTP_fraction_total_CCA_dist_median, reg5_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(5, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# reg6_NTP_fraction_total_CCA_dist_p16, reg6_NTP_fraction_total_CCA_dist_median, reg6_NTP_fraction_total_CCA_dist_p84 = calc_profile_in_XRISM_bins(6, [v_kick_total_CCA_dist_p16, v_kick_total_CCA_dist_median, v_kick_total_CCA_dist_p84], detector=True)
# # calculating the NTP fractions in the AGN region for hi/low jet powers and ages
# reg1_NTP_fraction_hi_Qjet, reg1_NTP_fraction_lo_Qjet = calc_profile_in_XRISM_bins(1, [v_kick_hi_Qjet, v_kick_lo_Qjet], detector=True)
# reg2_NTP_fraction_hi_Qjet, reg2_NTP_fraction_lo_Qjet = calc_profile_in_XRISM_bins(2, [v_kick_hi_Qjet, v_kick_lo_Qjet], detector=True)
# reg1_NTP_fraction_hi_active_age, reg1_NTP_fraction_lo_active_age = calc_profile_in_XRISM_bins(1, [v_kick_hi_active_age, v_kick_lo_active_age], detector=True)
# reg2_NTP_fraction_hi_active_age, reg2_NTP_fraction_lo_active_age = calc_profile_in_XRISM_bins(2, [v_kick_hi_active_age, v_kick_lo_active_age], detector=True)




# the figures

## outskirts model
fig = plt.figure(figsize=[6, 4.2])
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.4])

plt0 = plt.subplot(gs[0])
# important radii
plt0.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# outskirts model
plt0.fill_between(s, NTP_fraction_outskirts_min, NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4, label='Predicted range')
plt0.plot(s, NTP_fraction_outskirts, color='steelblue', lw=2, ls='-.', zorder=12, label='Median profile')
# calibration
plt0.scatter([1, r200/r500], NTP_fraction_fgas_constraint_median, color='steelblue')
plt0.errorbar(1, NTP_fraction_fgas_constraint_median[0], yerr=NTP_fraction_fgas_constraint_uncertainty[0], color='steelblue', elinewidth=2, capsize=3, label='Gas fraction constraints')
plt0.errorbar(r200/r500, NTP_fraction_fgas_constraint_median[1], yerr=NTP_fraction_fgas_constraint_uncertainty[1], color='steelblue', elinewidth=2, capsize=3)
plt0.plot(s, Perseus_like_NTP_fraction_profile, ls=':', lw=1.5, zorder=11, color='black', label='Perseus-like idealisation from Sullivan et al. 2024')
plt0.set_xscale('log')
plt0.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt0.set_xlim(3*10**(-4), 0.2*10**1)
plt0.set_ylabel('$\\mathcal{F} \, \\equiv \, p_\\mathrm{nt}/p$', fontsize = 18)
plt0.set_ylim(0, 0.84)
plt0.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt0.legend(loc=9, ncol=1, columnspacing=1.2, framealpha=1)
plt0.set_title('No AGN contribution', fontsize = 15)

plt1 = plt.subplot(gs[1], sharex=plt0)
# important radii
plt1.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# observational comparisons
plt1.fill_between(reg1, reg1**0*NTP_XRISM_Perseus_lower_lims[0], reg1**0*NTP_XRISM_Perseus_upper_lims[0], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg2, reg2**0*NTP_XRISM_Perseus_lower_lims[1], reg2**0*NTP_XRISM_Perseus_upper_lims[1], color='hotpink', alpha=0.3, zorder=10, label='XRISM Collaboration et al. 2025')
plt1.fill_between(reg3, reg3**0*NTP_XRISM_Perseus_lower_lims[2], reg3**0*NTP_XRISM_Perseus_upper_lims[2], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg4, reg4**0*NTP_XRISM_Perseus_lower_lims[3], reg4**0*NTP_XRISM_Perseus_upper_lims[3], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg5, reg5**0*NTP_XRISM_Perseus_lower_lims[4], reg5**0*NTP_XRISM_Perseus_upper_lims[4], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg6, reg6**0*NTP_XRISM_Perseus_lower_lims[5], reg6**0*NTP_XRISM_Perseus_upper_lims[5], color='hotpink', alpha=0.3, zorder=10)
# core model -- N cavity
plt1.plot(reg1, reg1**0*reg1_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg2, reg2**0*reg2_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg3, reg3**0*reg3_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg4, reg4**0*reg4_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg5, reg5**0*reg5_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg6, reg6**0*reg6_NTP_fraction_outskirts_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.fill_between(reg1, reg1_NTP_fraction_outskirts_min, reg1_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg2, reg2_NTP_fraction_outskirts_min, reg2_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg3, reg3_NTP_fraction_outskirts_min, reg3_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg4, reg4_NTP_fraction_outskirts_min, reg4_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg5, reg5_NTP_fraction_outskirts_min, reg5_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg6, reg6_NTP_fraction_outskirts_min, reg6_NTP_fraction_outskirts_max, color='lightsteelblue', alpha=0.4)
plt1.set_xscale('log')
plt1.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt1.set_xlim(3*10**(-4), 0.2*10**1)
plt1.set_ylabel('$\\left<\\mathcal{F}\\right>$', fontsize = 18)
plt1.set_ylim(0, 0.075)
plt1.set_yticks([0, 0.02, 0.04, 0.06])
plt1.legend(loc=1, framealpha=1)

plt.subplots_adjust(hspace=.0, wspace=.0)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.savefig('Perseus_NTP_outskirts.png', dpi=350, bbox_inches='tight')
plt.close()


print('Outskirts NTP fractions: reg1 = ', np.round([100*reg1_NTP_fraction_outskirts_min, 100*reg1_NTP_fraction_outskirts_max], 2), '%, reg2 = ', np.round([100*reg2_NTP_fraction_outskirts_min, 100*reg2_NTP_fraction_outskirts_max], 2), '%, reg3 = ', np.round([100*reg3_NTP_fraction_outskirts_min, 100*reg3_NTP_fraction_outskirts_max], 2), '%, reg4 = ', np.round([100*reg4_NTP_fraction_outskirts_min, 100*reg4_NTP_fraction_outskirts_max], 2), '%, reg5 = ', np.round([100*reg5_NTP_fraction_outskirts_min, 100*reg5_NTP_fraction_outskirts_max], 2), '%, reg 6 = ', np.round([100*reg6_NTP_fraction_outskirts_min, 100*reg6_NTP_fraction_outskirts_max], 2), '%')

## N, S cavity AGN model
fig = plt.figure(figsize=[12, 4.2])
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.4])

plt0 = plt.subplot(gs[0, 0])
# important radii
plt0.vlines(R_cocoon_N_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
plt0.vlines(R_shock_N_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
plt0.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# core model -- N cavity
plt0.fill_between(s, NTP_fraction_total_N_p16, NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4, label='Predicted range')
plt0.plot(s, NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=12, label='Median profile')
# calibration
plt0.scatter([1, r200/r500], NTP_fraction_fgas_constraint_median, color='steelblue')
plt0.errorbar(1, NTP_fraction_fgas_constraint_median[0], yerr=NTP_fraction_fgas_constraint_uncertainty[0], color='steelblue', elinewidth=2, capsize=3, label='Gas fraction constraints')
plt0.errorbar(r200/r500, NTP_fraction_fgas_constraint_median[1], yerr=NTP_fraction_fgas_constraint_uncertainty[1], color='steelblue', elinewidth=2, capsize=3)
plt0.set_xscale('log')
plt0.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt0.set_xlim(3*10**(-4), 0.2*10**1)
plt0.set_ylabel('$\\mathcal{F} \, \\equiv \, p_\\mathrm{nt}/p$', fontsize = 18)
plt0.set_ylim(0, 0.84)
plt0.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt0.legend(loc=9, ncol=1, columnspacing=1.2, framealpha=1, bbox_to_anchor=(0.64, 1))
plt0.set_title('AGN simulating Perseus\' $\\boldsymbol{N}$ inner cavity', fontsize = 15)

plt1 = plt.subplot(gs[1, 0], sharex=plt0)
# important radii
plt1.vlines(R_cocoon_N_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
plt1.vlines(R_shock_N_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
plt1.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# observational comparisons
plt1.fill_between(reg1, reg1**0*NTP_XRISM_Perseus_lower_lims[0], reg1**0*NTP_XRISM_Perseus_upper_lims[0], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg2, reg2**0*NTP_XRISM_Perseus_lower_lims[1], reg2**0*NTP_XRISM_Perseus_upper_lims[1], color='hotpink', alpha=0.3, zorder=10, label='XRISM Collaboration et al. 2025')
plt1.fill_between(reg3, reg3**0*NTP_XRISM_Perseus_lower_lims[2], reg3**0*NTP_XRISM_Perseus_upper_lims[2], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg4, reg4**0*NTP_XRISM_Perseus_lower_lims[3], reg4**0*NTP_XRISM_Perseus_upper_lims[3], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg5, reg5**0*NTP_XRISM_Perseus_lower_lims[4], reg5**0*NTP_XRISM_Perseus_upper_lims[4], color='hotpink', alpha=0.3, zorder=10)
plt1.fill_between(reg6, reg6**0*NTP_XRISM_Perseus_lower_lims[5], reg6**0*NTP_XRISM_Perseus_upper_lims[5], color='hotpink', alpha=0.3, zorder=10)
# core model -- N cavity
plt1.plot(reg1, reg1**0*reg1_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg2, reg2**0*reg2_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg3, reg3**0*reg3_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg4, reg4**0*reg4_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg5, reg5**0*reg5_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.plot(reg6, reg6**0*reg6_NTP_fraction_total_N_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt1.fill_between(reg1, reg1_NTP_fraction_total_N_p16, reg1_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg2, reg2_NTP_fraction_total_N_p16, reg2_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg3, reg3_NTP_fraction_total_N_p16, reg3_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg4, reg4_NTP_fraction_total_N_p16, reg4_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg5, reg5_NTP_fraction_total_N_p16, reg5_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.fill_between(reg6, reg6_NTP_fraction_total_N_p16, reg6_NTP_fraction_total_N_p84, color='lightsteelblue', alpha=0.4)
plt1.set_xscale('log')
plt1.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt1.set_xlim(3*10**(-4), 0.2*10**1)
plt1.set_ylabel('$\\left<\\mathcal{F}\\right>$', fontsize = 18)
plt1.set_ylim(0, 0.075)
plt1.set_yticks([0, 0.02, 0.04, 0.06])
plt1.legend(loc=1, framealpha=1)

plt2 = plt.subplot(gs[0, 1])
# important radii
plt2.vlines(R_cocoon_S_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
plt2.vlines(R_shock_S_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
plt2.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# core model -- S cavity
plt2.fill_between(s, NTP_fraction_total_S_p16, NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4, label='Predicted range')
plt2.plot(s, NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=12, label='Median profile')
# calibration
plt2.scatter([1, r200/r500], NTP_fraction_fgas_constraint_median, color='steelblue')
plt2.errorbar(1, NTP_fraction_fgas_constraint_median[0], yerr=NTP_fraction_fgas_constraint_uncertainty[0], color='steelblue', elinewidth=2, capsize=3, label='Gas fraction constraints')
plt2.errorbar(r200/r500, NTP_fraction_fgas_constraint_median[1], yerr=NTP_fraction_fgas_constraint_uncertainty[1], color='steelblue', elinewidth=2, capsize=3)
plt2.set_xscale('log')
plt2.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt2.set_xlim(3*10**(-4), 0.2*10**1)
plt2.set_ylim(0, 0.84)
plt.setp(plt2.get_yticklabels(), visible=False)
plt2.legend(loc=9, ncol=1, columnspacing=1.2, framealpha=1, bbox_to_anchor=(0.64, 1))
plt2.set_title('AGN simulating Perseus\' $\\boldsymbol{S}$ inner cavity', fontsize = 15)

plt3 = plt.subplot(gs[1, 1], sharex=plt2)
# important radii
plt3.vlines(R_cocoon_S_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
plt3.vlines(R_shock_S_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
plt3.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# observational comparisons
plt3.fill_between(reg1, reg1**0*NTP_XRISM_Perseus_lower_lims[0], reg1**0*NTP_XRISM_Perseus_upper_lims[0], color='hotpink', alpha=0.3, zorder=10)
plt3.fill_between(reg2, reg2**0*NTP_XRISM_Perseus_lower_lims[1], reg2**0*NTP_XRISM_Perseus_upper_lims[1], color='hotpink', alpha=0.3, zorder=10, label='XRISM Collaboration et al. 2025')
plt3.fill_between(reg3, reg3**0*NTP_XRISM_Perseus_lower_lims[2], reg3**0*NTP_XRISM_Perseus_upper_lims[2], color='hotpink', alpha=0.3, zorder=10)
plt3.fill_between(reg4, reg4**0*NTP_XRISM_Perseus_lower_lims[3], reg4**0*NTP_XRISM_Perseus_upper_lims[3], color='hotpink', alpha=0.3, zorder=10)
plt3.fill_between(reg5, reg5**0*NTP_XRISM_Perseus_lower_lims[4], reg5**0*NTP_XRISM_Perseus_upper_lims[4], color='hotpink', alpha=0.3, zorder=10)
plt3.fill_between(reg6, reg6**0*NTP_XRISM_Perseus_lower_lims[5], reg6**0*NTP_XRISM_Perseus_upper_lims[5], color='hotpink', alpha=0.3, zorder=10)
# core model -- S cavity
plt3.plot(reg1, reg1**0*reg1_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.plot(reg2, reg2**0*reg2_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.plot(reg3, reg3**0*reg3_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.plot(reg4, reg4**0*reg4_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.plot(reg5, reg5**0*reg5_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.plot(reg6, reg6**0*reg6_NTP_fraction_total_S_median, color='steelblue', lw=2, ls='-.', zorder=11)
plt3.fill_between(reg1, reg1_NTP_fraction_total_S_p16, reg1_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.fill_between(reg2, reg2_NTP_fraction_total_S_p16, reg2_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.fill_between(reg3, reg3_NTP_fraction_total_S_p16, reg3_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.fill_between(reg4, reg4_NTP_fraction_total_S_p16, reg4_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.fill_between(reg5, reg5_NTP_fraction_total_S_p16, reg5_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.fill_between(reg6, reg6_NTP_fraction_total_S_p16, reg6_NTP_fraction_total_S_p84, color='lightsteelblue', alpha=0.4)
plt3.set_xscale('log')
plt3.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
plt3.set_xlim(3*10**(-4), 0.2*10**1)
plt3.set_ylim(0, 0.075)
plt.setp(plt3.get_yticklabels(), visible=False)
plt3.legend(loc=1, framealpha=1)

plt.subplots_adjust(hspace=.0, wspace=.0)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.savefig('Perseus_NTP_inner_cavities.png', dpi=350, bbox_inches='tight')
plt.close()


print('N cavity NTP fractions: reg1 = ', np.round([100*reg1_NTP_fraction_total_N_p16, 100*reg1_NTP_fraction_total_N_p84], 2), '%, reg2 = ', np.round([100*reg2_NTP_fraction_total_N_p16, 100*reg2_NTP_fraction_total_N_p84], 2), '%, reg3 = ', np.round([100*reg3_NTP_fraction_total_N_p16, 100*reg3_NTP_fraction_total_N_p84], 2), '%, reg4 = ', np.round([100*reg4_NTP_fraction_total_N_p16, 100*reg4_NTP_fraction_total_N_p84], 2), '%, reg5 = ', np.round([100*reg5_NTP_fraction_total_N_p16, 100*reg5_NTP_fraction_total_N_p84], 2), '%, reg 6 = ', np.round([100*reg6_NTP_fraction_total_N_p16, 100*reg6_NTP_fraction_total_N_p84], 2), '%')
print('S cavity NTP fractions: reg1 = ', np.round([100*reg1_NTP_fraction_total_S_p16, 100*reg1_NTP_fraction_total_S_p84], 2), '%, reg2 = ', np.round([100*reg2_NTP_fraction_total_S_p16, 100*reg2_NTP_fraction_total_S_p84], 2), '%, reg3 = ', np.round([100*reg3_NTP_fraction_total_S_p16, 100*reg3_NTP_fraction_total_S_p84], 2), '%, reg4 = ', np.round([100*reg4_NTP_fraction_total_S_p16, 100*reg4_NTP_fraction_total_S_p84], 2), '%, reg5 = ', np.round([100*reg5_NTP_fraction_total_S_p16, 100*reg5_NTP_fraction_total_S_p84], 2), '%, reg 6 = ', np.round([100*reg6_NTP_fraction_total_S_p16, 100*reg6_NTP_fraction_total_S_p84], 2), '%')



# ## recent past outbursts prediction
# fig = plt.figure(figsize=[6, 4.2])
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.4])
#
# plt0 = plt.subplot(gs[0])
# # important radii
# plt0.vlines(R_cocoon_CCA_dist_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
# plt0.vlines(R_shock_CCA_dist_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
# plt0.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# # CCA distribution
# plt0.fill_between(s, NTP_fraction_total_CCA_dist_p16, NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4, zorder=1, label='Predicted range')
# plt0.plot(s, NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11, label='Median profile')
# # calibrations
# plt0.scatter([1, r200/r500], NTP_fraction_fgas_constraint_median, color='steelblue', zorder=12)
# plt0.errorbar(1, NTP_fraction_fgas_constraint_median[0], yerr=NTP_fraction_fgas_constraint_uncertainty[0], color='steelblue', elinewidth=2, capsize=3, label='Gas fraction constraints')
# plt0.errorbar(r200/r500, NTP_fraction_fgas_constraint_median[1], yerr=NTP_fraction_fgas_constraint_uncertainty[1], color='steelblue', elinewidth=2, capsize=3)
# # hi/lo jet powers
# plt0.plot(s[halo_radius<=r2500], NTP_fraction_hi_Qjet[halo_radius<=r2500], color='darkslateblue', lw=1.2, ls='-', alpha=0.4, label='$\\log Q_\\mathrm{jet} = \\mu_{\\log Q_\\mathrm{jet}} \\pm \\sigma_{\\log Q_\\mathrm{jet}}, \,\, \\log t_\\mathrm{on} = \\mu_{\\log t_\\mathrm{on}}$')
# plt0.plot(s[halo_radius<=r2500], NTP_fraction_lo_Qjet[halo_radius<=r2500], color='darkslateblue', lw=1.2, ls='-', alpha=0.4)
# # hi/lo active ages
# plt0.plot(s[halo_radius<=r2500], NTP_fraction_hi_active_age[halo_radius<=r2500], color='teal', lw=1.2, ls='-', alpha=0.4, label='$\\log Q_\\mathrm{jet} = \\mu_{\\log Q_\\mathrm{jet}}, \,\, \\log t_\\mathrm{on} = \\mu_{\\log t_\\mathrm{on}} \\pm \\sigma_{\\log t_\\mathrm{on}}$')
# plt0.plot(s[halo_radius<=r2500], NTP_fraction_lo_active_age[halo_radius<=r2500], color='teal', lw=1.2, ls='-', alpha=0.4)
# plt0.set_xscale('log')
# plt0.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
# plt0.set_xlim(3*10**(-4), 0.2*10**1)
# plt0.set_ylabel('$\\mathcal{F} \, \\equiv \, p_\\mathrm{nt}/p$', fontsize = 18)
# plt0.set_ylim(0, 0.84)
# plt0.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# plt0.legend(loc=9, ncol=1, columnspacing=1.2, framealpha=1, bbox_to_anchor=(0.64, 1))
# plt0.set_title('AGN simulating distributions of jet powers and ages', fontsize = 15)
#
# plt1 = plt.subplot(gs[1], sharex=plt0)
# # important radii
# plt1.vlines(R_cocoon_CCA_dist_median/r500, -0.01, 1, ls='--', color='steelblue', alpha=0.4, zorder=-10)
# plt1.vlines(R_shock_CCA_dist_median/r500, -0.01, 1, ls=':', color='steelblue', alpha=0.4, zorder=-10)
# plt1.vlines(r200/r500, -0.01, 1, ls='-.', color='gray', alpha=0.6, zorder=-10)
# # observational comparisons
# plt1.fill_between(reg1, reg1**0*NTP_XRISM_Perseus_lower_lims[0], reg1**0*NTP_XRISM_Perseus_upper_lims[0], color='hotpink', alpha=0.3, zorder=10)
# plt1.fill_between(reg2, reg2**0*NTP_XRISM_Perseus_lower_lims[1], reg2**0*NTP_XRISM_Perseus_upper_lims[1], color='hotpink', alpha=0.3, zorder=10, label='XRISM Collaboration et al. 2025')
# plt1.fill_between(reg3, reg3**0*NTP_XRISM_Perseus_lower_lims[2], reg3**0*NTP_XRISM_Perseus_upper_lims[2], color='hotpink', alpha=0.3, zorder=10)
# plt1.fill_between(reg4, reg4**0*NTP_XRISM_Perseus_lower_lims[3], reg4**0*NTP_XRISM_Perseus_upper_lims[3], color='hotpink', alpha=0.3, zorder=10)
# plt1.fill_between(reg5, reg5**0*NTP_XRISM_Perseus_lower_lims[4], reg5**0*NTP_XRISM_Perseus_upper_lims[4], color='hotpink', alpha=0.3, zorder=10)
# plt1.fill_between(reg6, reg6**0*NTP_XRISM_Perseus_lower_lims[5], reg6**0*NTP_XRISM_Perseus_upper_lims[5], color='hotpink', alpha=0.3, zorder=10)
# # core model -- CCA distribution
# plt1.plot(reg1, reg1**0*reg1_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.plot(reg2, reg2**0*reg2_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.plot(reg3, reg3**0*reg3_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.plot(reg4, reg4**0*reg4_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.plot(reg5, reg5**0*reg5_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.plot(reg6, reg6**0*reg6_NTP_fraction_total_CCA_dist_median, color='steelblue', lw=2, ls='-.', zorder=11)
# plt1.fill_between(reg1, reg1_NTP_fraction_total_CCA_dist_p16, reg1_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# plt1.fill_between(reg2, reg2_NTP_fraction_total_CCA_dist_p16, reg2_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# plt1.fill_between(reg3, reg3_NTP_fraction_total_CCA_dist_p16, reg3_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# plt1.fill_between(reg4, reg4_NTP_fraction_total_CCA_dist_p16, reg4_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# plt1.fill_between(reg5, reg5_NTP_fraction_total_CCA_dist_p16, reg5_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# plt1.fill_between(reg6, reg6_NTP_fraction_total_CCA_dist_p16, reg6_NTP_fraction_total_CCA_dist_p84, color='lightsteelblue', alpha=0.4)
# # hi/lo jet powers
# plt1.plot(reg1, reg1**0*reg1_NTP_fraction_lo_Qjet, color='darkslateblue', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg1, reg1**0*reg1_NTP_fraction_hi_Qjet, color='darkslateblue', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg2, reg2**0*reg2_NTP_fraction_lo_Qjet, color='darkslateblue', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg2, reg2**0*reg2_NTP_fraction_hi_Qjet, color='darkslateblue', lw=1.2, ls='-', alpha=0.4)
# # hi/lo active ages
# plt1.plot(reg1, reg1**0*reg1_NTP_fraction_lo_active_age, color='teal', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg1, reg1**0*reg1_NTP_fraction_hi_active_age, color='teal', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg2, reg2**0*reg2_NTP_fraction_lo_active_age, color='teal', lw=1.2, ls='-', alpha=0.4)
# plt1.plot(reg2, reg2**0*reg2_NTP_fraction_hi_active_age, color='teal', lw=1.2, ls='-', alpha=0.4)
# plt1.set_xscale('log')
# plt1.set_xlabel('$r/r_\mathrm{500}$', fontsize = 18)
# plt1.set_xlim(3*10**(-4), 0.2*10**1)
# plt1.set_ylabel('$\\left<\\mathcal{F}\\right>$', fontsize = 18)
# plt1.set_ylim(0, 0.075)
# plt1.set_yticks([0, 0.02, 0.04, 0.06])
# plt1.legend(loc=1, framealpha=1)
#
# plt.subplots_adjust(hspace=.0, wspace=.0)
# fig.add_subplot(111, frame_on=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#
# plt.savefig('Perseus_NTP_recent_outbursts.png', dpi=350, bbox_inches='tight')
# plt.close()
