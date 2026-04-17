This code uses an minor modification of RAiSEHD.py from github.com/rossjturner/RAiSEHD. Specifically, we create the output files:

      df['Time (yrs)'] = 10 ** np.asarray(source_age).astype(np.float_)
      df['Lobe lengths (kpc)'] = list(2 * lobe_lengths.T / const.kpc.value)
      df['Shock lengths (kpc)'] = list(2 * shock_lengths.T / const.kpc.value)
      df['Pressure (Pa)'] = shock_pressures[0, :]
      df['Axis Ratio'] = lobe_lengths[0, :] / np.max(lobe_lengths[1:, :] * np.sin(theta[1:, None]) + 1e-256, axis=0)
