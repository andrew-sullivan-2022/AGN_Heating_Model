**Function for calculating AGN heating:**
-----------------------------------------

**calculate_AGN_heating**(**log_Qjet_vals**,   **log_active_age_vals**,   **duty_cycle**,   **redshift**,   **gas_density_profile**,   **temperature_profile**,   **halo_radius**,   **log_dt**=_0.01_)

________________________
__Parameters:__


**_log_Qjet_vals_ : float or array-like**

  Logarithmic jet power [log W]


**_log_active_age_vals_ : float or array-like**

  Logarithmic active age [log yr]


**_duty_cycle_ : float**

  Duty cycle of the AGN [decimal percentage]


**_redshift_ : float**

  Redshift


**_gas_density_profile_ : array-like**

  Gas density [kg/m^3] of the environment, with values corresponding to _halo_radius_


**_temperature_profile_ : array-like**

  Temperature [K] of the environment, with values correspond to _halo_radius_


**_halo_radius_ : array-like**

  Radial component [m] of _gas_density_profile_ and _temperature_profile_


**_log_dt_ : float, optional**

  Logarithmic time spacing [log yr] to evolve the source in RAiSE

________________________
__Returns:__


Creates files (.txt) for:

**_Q_eff_ : array-like**

  Radially-averaged profile for the volumetric power profile of the AGN [W/m^3]


**_v_kick_ : array-like**

  Radially-averaged profile of the velocity kick profile imparted on the gas [m/s]


**_NTP_fraction_ : array-like**

  Radially-averaged profile of the fraction of non-thermal pressure to total pressure of the gas [decimal percentage]

**_coupling_efficiency_ : float**

  Outburst-averaged efficiency of mechanical jet energy converted into kinetic energy of the gas [decimal percentage]

______________________________________________________________________________________

This code uses an minor modification of RAiSEHD.py from github.com/rossjturner/RAiSEHD. 


RAiSE_run outputs are changed to:

      df['Time (yrs)'] = 10 ** np.asarray(source_age).astype(np.float_)
      df['Lobe lengths (kpc)'] = list(2 * lobe_lengths.T / const.kpc.value)
      df['Shock lengths (kpc)'] = list(2 * shock_lengths.T / const.kpc.value)
      df['Pressure (Pa)'] = shock_pressures[0, :]
      df['Axis Ratio'] = lobe_lengths[0, :] / np.max(lobe_lengths[1:, :] * np.sin(theta[1:, None]) + 1e-256, axis=0)

RAiSE angular resolution is also set to:

      nangles = 64
