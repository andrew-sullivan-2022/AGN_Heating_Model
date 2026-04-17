import builtins
import h5py
import numpy as np
import pandas as pd
from pydl.pydlutils.cooling import read_ds_cooling
from numba import njit, prange
from RAiSEHD import RAiSE_run

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
    # thermal pressure scale height at the edge of the lodge
    th_pressure_scale_heights_at_lobe = np.empty((power_res, age_res))
    for i in range(power_res):
        for j in range(age_res):
            th_pressure_scale_heights_at_lobe[i, j] = -th_pressure_profile[np.argmin(np.abs(halo_radius - R_cocoon[i, j]))]/th_pressure_derivative[np.argmin(np.abs(halo_radius - R_cocoon[i, j]))]
    # setting the upper limit of the bubble propagation ~ 2.5 thermal pressure scale heights
    @njit(parallel=True)
    def set_bubble_propagation_upper_limit(bubble_filling_factors_matrix, th_pressure_scale_heights_at_lobe):
        for i in prange(power_res):
            for j in range(age_res):
                limit = 3*th_pressure_scale_heights_at_lobe[i, j]
                for k in range(radius_bins):
                    for m in range(prop_bins):
                        if prop_axis_bubble[m] > limit:
                            bubble_filling_factors_matrix[i, j, k, m] = 0
        return bubble_filling_factors_matrix
    bubble_filling_factors_matrix = set_bubble_propagation_upper_limit(bubble_filling_factors_matrix, th_pressure_scale_heights_at_lobe)
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
    for i in prange(power_res):
        for j in range(age_res):
            for k in range(prop_bins):
                th_pressure_profile_along_prop_axis[i, j, k] = th_pressure_profile[np.argmin(np.abs(halo_radius - prop_axis_for_bubble_properties[i, j, k]))]
                th_pressure_derivative_along_prop_axis[i, j, k] = th_pressure_derivative[np.argmin(np.abs(halo_radius - prop_axis_for_bubble_properties[i, j, k]))]
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
    save_nested_arrays("Q_eff_" + descriptor + ".txt", effective_energy_injection_rate)
    save_nested_arrays("v_kick_" + descriptor + ".txt", velocity_kick_in_core)
    save_nested_arrays("NTP_fraction_" + descriptor + ".txt", NTP_fraction_in_core)
