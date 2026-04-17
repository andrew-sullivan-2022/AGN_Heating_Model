**Function for calculating AGN heating**

**calculate_AGN_heating**(_log_Qjet_vals, log_active_age_vals, duty_cycle, redshift, gas_density_profile, temperature_profile, halo_radius, log_dt_)

_Repuired parameters:_

 _log_Qjet_vals_ : float or array-like
Logarithmic jet power [log W]

 _log_active_age_vals_ : float or array-like
Logarithmic active age [log yr]

 _duty_cycle_ : float
Duty cycle of the AGN [percent]

 _redshift_ : float
Redshift

 _gas_density_profile_ : array-like
Gas density [kg/m^3] of the environment, with values corresponding to _halo_radius_

 _temperature_profile_ : array-like
Temperature [K] of the environment, with values correspond to _halo_radius_

 _halo_radius_ : array-like
Radial component [m] of _gas_density_profile_ and _temperature_profile_

_Optional parameters:_

 _log_dt_ : float
Logarithmic time spacing [log yr] to evolve the source in RAiSE

_Returns:_ 
Creates array files (.txt) for:

 _Q_eff_ : array-like
Effective radially-averaged volumetric power [W/m^3] of the AGN

 _v_kick_ : array-like
Velocity kick [m/s] imparted on the gas

 _NTP_fraction_ : array-like
Fraction of non-thermal pressure to total pressure [percent] 



This code uses an minor modification of RAiSEHD.py from github.com/rossjturner/RAiSEHD. 
RAiSE_run outputs are changed to:

      df['Time (yrs)'] = 10 ** np.asarray(source_age).astype(np.float_)
      df['Lobe lengths (kpc)'] = list(2 * lobe_lengths.T / const.kpc.value)
      df['Shock lengths (kpc)'] = list(2 * shock_lengths.T / const.kpc.value)
      df['Pressure (Pa)'] = shock_pressures[0, :]
      df['Axis Ratio'] = lobe_lengths[0, :] / np.max(lobe_lengths[1:, :] * np.sin(theta[1:, None]) + 1e-256, axis=0)
