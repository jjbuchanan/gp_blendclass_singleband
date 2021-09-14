import numpy as np
import pandas as pd

'''
All values taken from
https://github.com/lsst/throughputs/tree/master/baseline

Columns in each table:
lambd(nm)  throughput(0-1)
'''
throughput_table = lambda throughput_type, band: pd.read_csv(
                            f'lsst_throughputs/{throughput_type}_{band}.dat',
                            delim_whitespace=True)

'''
Input parameters
f : Band name
throughput_type : hardware_throughput or total_throughput; generally, a dictionary
                  mapping filters to pd.DataFrame instances where each DataFrame
                  has columns 'lambd' and 'throughput'

Returns
An estimate of the integral, over all wavelengths, of throughput divided by wavelength
'''
def integrated_throughput_in_band(band, throughput_type):
    L = throughput_table(throughput_type, band).lambd.to_numpy()
    lambda_ratio = L[1:] / L[:-1]
    lambda_weights = np.log(lambda_ratio)
    
    throughputs = throughput_table(throughput_type, band).throughput.to_numpy()[:-1]
    
    weighted_throughput = lambda_weights * throughputs
    integrated_throughput = np.sum(weighted_throughput)
    
    return integrated_throughput


h_erg = 6.62607015e-27 # erg*s
c_nm_s = 299792458000000000 # nm/s

# Convert an AB magnitude to erg/s/cm2/Hz
f_cgs_from_mag = lambda mag : 10**(-(mag + 48.6)/2.5)

# Convert a flux in erg/s/cm2/Hz to AB magnitude
# This should be exactly the inverse operation of f_cgs_from_mag
mag_from_f_cgs = lambda f_cgs : -2.5*np.log10(f_cgs) - 48.6

'''
Input parameters
mag : AB magnitude
lambda_<low/high> : Bandpass lower/upper bound (wavelength)

Returns
Flux density of photons (photons/s/cm2) emitted by a source within the given bandpass,
assuming the AB magnitude is constant as a function of frequency throughout the bandpass.
'''
def flux_photons_from_mag(mag, lambda_low, lambda_high):
    # Convert an AB magnitude to erg/s/cm2/Hz,
    # which is the differential (per Hz) energy flux density (ergs per s per cm2).
    f_cgs = f_cgs_from_mag(mag)
    
    '''
    Energy per photon in erg, based on frequency in Hz:
    phot_energy_erg = h_erg * freq
    
    Differential (per Hz) photon flux density (number of photons per s per cm2):
    pho_flux_density_per_Hz = f_cgs / phot_energy_erg
                            = (f_cgs / h_erg) * (1 / freq)
    
    Assuming the AB magnitude, and hence f_cgs, is constant as a function of
    frequency within a given band, compute the flux density of photons integrated
    over the entire band:
    f_photons = integral(pho_flux_density_per_Hz, low=freq_low, high=freq_hi)
              = (f_cgs / h_erg) * (np.log(freq_hi) - np.log(freq_lo))
              = (f_cgs / h_erg) * np.log(freq_hi / freq_lo)
    Since
      freq_low = c / lambda_high
      freq_high = c / lambda_low
    we have:
    '''
    f_photons = (f_cgs / h_erg) * np.log(lambda_high / lambda_low)
    
    # Notice that lambda_high and lambda_low can have any length units,
    # as long as they're both the same.
    
    return f_photons

'''
Input parameters
mag : AB magnitude
lambda_weights : How much each lambda bin contributes to the integrated flux density
throughputs : Array of values between 0 and 1 representing the fraction of photons
                within a given lambda bin which, having originated from a source of
                magnitude mag, go on to pass all filters and photoconvert in the CCDs.

Returns
f_filtered_photons : Expected flux density of detected photons (photons/s/cm2)
                        that have passed through all filters.
'''
def flux_filtered_photons_from_mag(mag, integrated_throughput):
    '''
    Effectively, the code should do this:
    
    f_filtered_photons = 0
    for idx in range(len(throughput_array.index)):
        lambda_low = throughput_array.lambda[idx]
        lambda_high = throughput_array.lambda[idx+1]
        f_photons = flux_photons_from_mag(mag, lambda_low, lambda_high)
        throughput = throughput_array.throughput[idx]
        f_filtered_photons += f_photons * throughput
    return f_filtered_photons
    
    Notice that there's
        a) a determination of f_cgs from mag, and
        b) a multiplication by (f_cgs / h_erg)
    in every single call of flux_photons_from_mag, which is a ton of redundant
    computation. The following is a streamlined rewrite.
    '''
    f_cgs = f_cgs_from_mag(mag)
    
    f_filtered_photons = (f_cgs / h_erg) * integrated_throughput
    
    return f_filtered_photons

def npho_from_mag(mag, integrated_throughput, exposure_time, area):
    f_photons = flux_filtered_photons_from_mag(mag, integrated_throughput)
    return f_photons * exposure_time * area

# For fixed values of integrated_throughput, exposure_time, and area,
# this should be exactly the inverse operation of npho_from_mag.
def mag_from_npho(npho, integrated_throughput, exposure_time, area):
    f_cgs = (npho * h_erg) / (integrated_throughput * exposure_time * area)
    return mag_from_f_cgs(f_cgs)

'''
The effective sky magnitude is computed as if the sky itself were an
independent source of light. Any of its own light it happens to absorb is
"folded in" to this effective magnitude, so the only thing that can further
reduce the overall photon throughput is the telescope hardware.

Sky magnitudes are given per square arcsec, so the number of photons estimated
from sky magnitudes should be multiplied by the pixel scale^2 (in arcsec^2) to
get the number of expected photons in a single pixel.

Sky magnitudes taken from
https://smtn-002.lsst.io/#sky-counts
'''
sky_mag = {'u': 22.95, 'g': 22.24, 'r': 21.20, 'i': 20.47, 'z': 19.60, 'y': 18.63}

# # The values in sky_mag above agree within 0.04 of the values obtained via:
# def sky_mag(band):
#     throughputs = throughput_table('hardware', band).throughput
#     darksky = pd.read_csv('lsst_throughputs/darksky.dat', delim_whitespace=True)
#     flam_weights = darksky.lambd[:8501] * throughputs
#     weighted_diff_flam = darksky.flam_cgs[:8501] * flam_weights
#     integrated_throughput = integrated_throughput_in_band(band, 'hardware')
#     f_cgs = 0.1 * np.sum(weighted_diff_flam) / (c_nm_s * integrated_throughput)
#     return mag_from_f_cgs(f_cgs)

def npho_dark_sky(band, exposure_time, area, scale):
    integrated_throughput = integrated_throughput_in_band(band, 'hardware')
    npho_per_arcsec2 = npho_from_mag(sky_mag[band], integrated_throughput,
                                    exposure_time, area)
    npho_per_pixel = npho_per_arcsec2 * scale**2
    return npho_per_pixel