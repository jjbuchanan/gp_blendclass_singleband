import numpy as np
import pandas as pd

import galsim
from galsim import degrees, arcsec

from flux_conversion import npho_from_mag, integrated_throughput_in_band


'''
Given a table of galaxies, select those within a rectangular RA-Dec patch
spanning a given angular width on each side, centered at a given RA,Dec.
'''
def select_gals(center, halfwidth, gal_table):
    # Define a window around the image center
    # Be as generous with the RA boundary as possible
    ra_up = center.deproject(-halfwidth, -halfwidth).ra
    dec_up = center.deproject(0*arcsec, halfwidth).dec
    ra_down = center.ra - (ra_up - center.ra)
    dec_down = center.dec - (dec_up - center.dec)

    # Select the galaxies in this window
    ra_filter = gal_table.ra > ra_down/degrees
    ra_filter &= gal_table.ra < ra_up/degrees
    dec_filter = gal_table.dec > dec_down/degrees
    dec_filter &= gal_table.dec < dec_up/degrees
    return gal_table[ra_filter & dec_filter]


'''
Construct the light model for a single galaxy
'''
def draw_gal(gal, npho, dx, dy):
    '''
    First, construct the undistorted galaxy light profile:
    Model the disk and bulge separately, then add together.
    From the galsim docs: "The shear() method precisely preserves the area."
    So start with a radius (_eff_rad) of a circle with the same area as the final desired ellipse,
    and then apply an area-preserving shear to reach the desired ellipticities.
    '''
    # Bulge
    bulge_eff_rad = np.sqrt(gal.size_bulge_true * gal.size_minor_bulge_true)
    bulge = galsim.Sersic(n=gal.sersic_bulge, half_light_radius=bulge_eff_rad)
    bulge = bulge.shear(g1=gal.ellipticity_1_bulge_true, g2=gal.ellipticity_2_bulge_true)
    bulge *= gal.bulge_to_total_ratio_i / bulge.flux
    # Disk
    disk_eff_rad = np.sqrt(gal.size_disk_true * gal.size_minor_disk_true)
    disk = galsim.Sersic(n=gal.sersic_disk, half_light_radius=disk_eff_rad)
    disk = disk.shear(g1=gal.ellipticity_1_disk_true, g2=gal.ellipticity_2_disk_true)
    disk *= (1.-gal.bulge_to_total_ratio_i) / disk.flux
    # And finally sum
    lightmodel = disk + bulge
    
    '''
    Apply weak lensing distortions, now that all "true" (unlensed) quantities have been set.
    Need to first convert from "theoretical" shear and convergence to "reduced" shear and magnification.
    Using the same transformations as implemented here:
    http://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/lensing_ps.html#theoryToObserved
    '''
    g1 = gal.shear_1/(1-gal.convergence)
    g2 = gal.shear_2/(1-gal.convergence)
    mu = 1/((1-gal.convergence)**2 - (gal.shear_1**2 + gal.shear_2**2))
    lightmodel = lightmodel.lens(g1, g2, mu)
    
    '''
    Overall flux normalization. To be compatible with the photon shooting procedure
    specified below, this should be the expected number of photons reaching the camera.
    '''
    lightmodel *= npho / lightmodel.flux
    
    lightmodel = lightmodel.shift(dx, dy)

    return lightmodel


'''
Combined galaxy light profile.
In expectation, i.e. without random exposure-dependent fluctuations.

Mean atmospheric throughput is modeled here, but the atmospheric PSF (since it will change from exposure to exposure) is not.
Similarly, the mean number of photons per galaxy is modeled here, but the actual number collected in any given exposure (which will be Poisson-distributed) is not.
Finally, mean galactic positions on the image are modeled here, but random shifts due to telescope dithering are not.
'''
def build_galaxy_composite(gals, image_center, params, verbose=False):
    # Truth information for all drawn galaxies
    true_pos = pd.DataFrame(np.zeros((len(gals.index), 2)), columns=['x','y'])
    npho = pd.DataFrame(data=np.zeros((len(gals.index), len(params.filters))), columns=params.filters)

    # Combined galaxy light profile
    # One for every filter
    gal_composite_drawer = [galsim.Sersic(n=1, scale_radius=1, flux=0.0) # Empty profile
                            for _ in range(len(params.filters))]

    integrated_total_throughput = {band: integrated_throughput_in_band(band, 'total')
                                    for band in params.filters}

    for g_iloc in range(len(gals.index)):
        if verbose:
            if (g_iloc+1)%500 == 0:
                print(g_iloc+1, '/', len(gals.index))
        
        gal = gals.iloc[g_iloc]

        '''
        The values returned by project() should have an orientation that is exactly
        compatible with shift(). However, project() returns Angle instances, while
        shift() expects bare numbers that are compatible with the pixel scale.
        Docs:
        http://galsim-developers.github.io/GalSim/_build/html/gsobject.html?highlight=drawimage#galsim.GSObject.shift
        http://galsim-developers.github.io/GalSim/_build/html/wcs.html?highlight=project#galsim.CelestialCoord.project
        '''
        gal_coord = galsim.CelestialCoord(gal.ra*degrees, gal.dec*degrees)
        dx, dy = image_center.project(gal_coord)
        # Since the pixel scale is given in arcsec, give dx and dy in arcsec as well
        dx, dy = dx/arcsec, dy/arcsec
        
        # The true position of this galaxy, in grid coordinates
        true_pos.x[g_iloc] = params.ngrid_x/2 - 1 + (1/params.scale)*dx
        true_pos.y[g_iloc] = params.ngrid_y/2 - 1 + (1/params.scale)*dy

        for band_idx, band in enumerate(params.filters):        
            mag = gal[f'mag_{band}_noMW']
            # The expected number of photons collected from this galaxy in this band
            npho_gal = npho_from_mag(mag, integrated_total_throughput[band],
                                     params.exposure_time, params.area)
            npho[band][g_iloc] = npho_gal
            
            gal_drawer = draw_gal(gal, npho_gal, dx, dy)
            gal_composite_drawer[band_idx] += gal_drawer

    return gal_composite_drawer, true_pos, npho