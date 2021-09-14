import numpy as np
import pandas as pd
from collections import namedtuple

import galsim
from galsim import degrees, arcsec

from flux_conversion import npho_dark_sky
from galaxy_drawing import select_gals, build_galaxy_composite


def draw_exposure(gal_composite_drawer, mean_npho_sky, seed, params):
    sky_conditions = params.kolmogorov_fwhm
    npho_sky = mean_npho_sky
    # Introduce some variability between exposures
    rng = np.random.default_rng()
    sky_conditions *= (1 + params.kolmogorov_variability)**rng.standard_normal()
    npho_sky *= (1 + params.skycount_variability)**rng.standard_normal()

    # Sky background, including random per-pixel fluctuations
    image = galsim.ImageF(params.ngrid_x, params.ngrid_y, init_value=npho_sky)
    seed += 1
    sky_noise = galsim.PoissonNoise(rng=galsim.BaseDeviate(seed))
    image.addNoise(sky_noise)
    image *= params.gain
    # Digitize
    image = galsim.ImageI(image)
    # Record the mean sky level actually applied to this image
    npho_sky = np.mean(image.array)

    # Convolve the galaxy light profile with the atmospheric PSF
    psf = galsim.Kolmogorov(fwhm=sky_conditions)
    smeared_gal_drawer = galsim.Convolve(psf, gal_composite_drawer)

    # Simulate telescope dither, which (after final image corrections) is roughly 
    # equivalent to a random sub-pixel-scale shift of the image coordinates.
    offset = rng.uniform(low=-1, high=1, size=2)

    # Compute the final image pixel values
    seed += 2
    image = smeared_gal_drawer.drawImage(
        image=image, add_to_image=True,
        scale=params.scale, gain=params.gain, offset=offset,
        method='phot', rng=galsim.BaseDeviate(seed-1),
        sensor=galsim.SiliconSensor(name='lsst_itl_32',
                                    rng=galsim.BaseDeviate(seed))
    )

    pixels = image.array.astype(float)

    return pixels, offset, seed, npho_sky


def dither_correction(pixels, offset):
    dx, dy = offset
    sx, sy = int(np.sign(dx)), int(np.sign(dy))
    dx, dy = abs(dx), abs(dy)
    '''To invert the dither we need to roll the image array in the
    opposite direction of the original offset.
    In the convention used by both galsim and matplotlib, the
    x direction corresponds to numpy array axis 0, and y to 1.'''
    translated_pixels = (1-dx)*(1-dy)*pixels
    translated_pixels += dy*(1-dx)*np.roll(pixels, -sy, axis=0)
    translated_pixels += dx*(1-dy)*np.roll(pixels, -sx, axis=1)
    translated_pixels += dx*dy*np.roll(np.roll(pixels, -sy, axis=0), -sx, axis=1)

    return translated_pixels


# Simulate multiple, coadded exposures
def draw_coadd(band_idx, band, gal_composite_drawer, params, verbose=False):
    rng = np.random.default_rng()
    seed = rng.integers(100000000)

    coadded_image_pixels = np.zeros((params.ngrid_x, params.ngrid_y), dtype=int)
    mean_npho_sky = npho_dark_sky(band, params.exposure_time, params.area, params.scale)

    sum_applied_npho_sky = 0

    if verbose:
        print('Exposure', end='')
    for exposure in range(params.n_exposures):
        if verbose:
            print('', exposure+1, end='')
            if (exposure+1)%10 == 0:
                print()
        
        # Draw this single exposure
        pixels, offset, seed, applied_npho_sky = draw_exposure(gal_composite_drawer[band_idx],
            mean_npho_sky, seed, params)
        sum_applied_npho_sky += applied_npho_sky
        
        # Coadd
        translated_pixels = dither_correction(pixels, offset)
        coadded_image_pixels += translated_pixels.round().astype(int)

    return coadded_image_pixels, sum_applied_npho_sky


def draw_postage_stamp(window, params, image_center=None, label=None):
    if image_center is None:
    # Pick some RA,Dec center for the scene
    # # Random center
    #     center_ra = rng.uniform(window.ra.min()+p.image_halfwidth/degrees,
    #                            window.ra.max()-p.image_halfwidth/degrees)*degrees
    #     center_dec = rng.uniform(window.dec.min()+p.image_halfwidth/degrees,
    #                             window.dec.max()-p.image_halfwidth/degrees)*degrees
    # Center of window
        center_ra = (window.ra.min() + (window.ra.max()-window.ra.min())/2)*degrees
        center_dec = (window.dec.min() + (window.dec.max()-window.dec.min())/2)*degrees
        image_center = galsim.CelestialCoord(center_ra, center_dec)

    # Given the image center and width, select all the galaxies
    # from the source table inside that box.
    gals = select_gals(image_center, params.image_halfwidth, window)

    # Work out how to draw each galaxy in relation to the image center
    gal_composite_drawer, true_pos, npho = build_galaxy_composite(gals,
                                                        image_center, params)

    # Simulate a coadded image in each band
    for band_idx, band in enumerate(params.filters):
        coadded_image_pixels, mean_npho_sky = draw_coadd(band_idx, band,
                                                gal_composite_drawer, params)
        # Save the image pixel data
        fname = f'pixelCounts_{band}-band'
        if label is not None:
            fname += '_' + label
        np.savetxt(fname + '.csv', coadded_image_pixels, delimiter=",")
        # Save other image data
        pdict = params._asdict()
        for pname in ['image_halfwidth', 'area']:
            del(pdict[pname])
        pdict['mean_npho_sky'] = mean_npho_sky
        imdata = pd.Series(pdict)
        fname = 'imdata'
        if label is not None:
            fname += '_' + label
        imdata.to_csv(fname + '.csv')

    # Some drawn galaxies have centers that lie outside the image frame,
    # so select just those that lie inside.
    # Exclude a 1-pixel band around the edge of the image.
    in_image_x_filter = (true_pos.x >= 1) & (true_pos.x < params.ngrid_x - 1)
    in_image_y_filter = (true_pos.y >= 1) & (true_pos.y < params.ngrid_y - 1)
    # Give columns more informative names
    npho = npho.rename(columns={c: 'meanNphoPerExposure_'+c for c in npho.columns})
    true_pos = true_pos.rename(columns={c: 'pixIdx_'+c for c in true_pos.columns})
    gals = gals.rename(columns={'shear_1': 'gamma1', 'shear_2': 'gamma2', 'convergence': 'kappa'})
    gals = gals.rename(columns={f'mag_{band}_noMW': f'mag_{band}' for band in params.filters})
    gals = gals[['gamma1', 'gamma2', 'kappa', 'redshift'] + 
                [f'mag_{band}' for band in params.filters]]
    # Merge galaxy truth information
    gals = gals.reset_index(drop=True) # Use same index as other tables
    galinfo = pd.concat([gals, true_pos, npho], axis=1)
    galinfo_in_image = galinfo[in_image_x_filter & in_image_y_filter]
    # Save galaxy truth information
    fname = 'galinfo'
    if label is not None:
        fname += '_' + label
    galinfo_in_image.to_csv(fname + '.csv', index=False)


def draw_large_scene(window, params, image_center=None, label=None):
    if label is not None:
        print('Drawing', label)

    if image_center is None:
    # Pick some RA,Dec center for the scene
        # Center of window
        center_ra = (window.ra.min() + (window.ra.max()-window.ra.min())/2)*degrees
        center_dec = (window.dec.min() + (window.dec.max()-window.dec.min())/2)*degrees
        image_center = galsim.CelestialCoord(center_ra, center_dec)
        # # Random center
        # center_ra = rng.uniform(window.ra.min()+p.image_halfwidth/degrees,
        #                        window.ra.max()-p.image_halfwidth/degrees)*degrees
        # center_dec = rng.uniform(window.dec.min()+p.image_halfwidth/degrees,
        #                         window.dec.max()-p.image_halfwidth/degrees)*degrees

    # Given the image center and width, select all the galaxies from the source
    # table inside that box.
    gals = select_gals(image_center, params.image_halfwidth, window)

    # Work out how to draw each galaxy in relation to the image center
    gal_composite_drawer, true_pos, npho = build_galaxy_composite(gals,
                                            image_center, params, verbose=True)

    # Save all galaxy info
    fname = f'{params.ngrid_x}x{params.ngrid_y}_gal_info.csv'
    if label is not None:
        fname = f'gal_info_{label}.csv'
    gals.to_csv(fname)
    # Save true galaxy positions, in image pixel grid coordinates
    fname = f'{params.ngrid_x}x{params.ngrid_y}_true_pos.csv'
    if label is not None:
        fname = f'true_pos_{label}.csv'
    true_pos.to_csv(fname)
    # Save mean total npho per exposure, in each filter band, for each galaxy
    fname = f'{params.ngrid_x}x{params.ngrid_y}_true_npho_allbands.csv'
    if label is not None:
        fname = f'true_npho_allbands_{label}.csv'
    npho.to_csv(fname)

    # Simulate a coadded image in each filter band
    for band_idx, band in enumerate(params.filters):
        print(band, 'band')
        coadded_image_pixels, mean_npho_sky = draw_coadd(band_idx, band,
                                    gal_composite_drawer, params, verbose=True)
        # Save image pixel values
        fname = f'{params.ngrid_x}x{params.ngrid_y}-pix_{band}-band.csv'
        if label is not None:
            fname = f'{band}-band_{label}.csv'
        np.savetxt(fname, coadded_image_pixels, delimiter=',')


def draw_specific_tiles(draw_method, window, params, image_indices):
    dec_span_degrees = window['dec'].max() - window['dec'].min()
    ra_span_degrees_raw = window['ra'].max() - window['ra'].min()
    '''Convert RA span into gnomonic (TAN) span.
    For a fixed raw RA span, the gnomonic span decreases as Dec approaches +/-90 degrees.
    Thus to define a gnomonic span with the widest possible validity
    (not going beyond the bounds of the dataset),
    use the most extreme Dec value attested in the dataset.'''
    ra_span_degrees_min = ra_span_degrees_raw * np.cos(np.abs(window['dec']).max() * np.pi/180)
    dec_span_arcsec = dec_span_degrees*degrees / arcsec
    ra_span_arcsec_min = ra_span_degrees_min*degrees / arcsec
    dec_span_pixels = dec_span_arcsec / params.scale
    ra_span_pixels_min = ra_span_arcsec_min / params.scale
    # Without checking I don't know which of RA vs. Dec corresponds to x vs. y,
    # but for square images, that difference doesn't matter here.
    dec_span_tiles = dec_span_pixels / params.ngrid_y
    ra_span_tiles_min = ra_span_pixels_min / params.ngrid_x

    print('ra_span_degrees_min =', ra_span_degrees_min)
    print('dec_span_degrees =', dec_span_degrees)

    # For every desired image, draw a coadded exposure at that location.
    for dec_tile_idx, ra_tile_idx in image_indices:
        print('dec_tile_idx, ra_tile_idx =', dec_tile_idx, ra_tile_idx)

        # Specify the RA,Dec corresponding to the selected image location
        center_ra = (window['ra'].min() + 
                ra_span_degrees_min * (ra_tile_idx/ra_span_tiles_min))*degrees
        center_dec = (window['dec'].min() +
                    dec_span_degrees * (dec_tile_idx/dec_span_tiles))*degrees
        print('center_ra =', center_ra)
        print('center_dec =', center_dec)
        image_center = galsim.CelestialCoord(center_ra, center_dec)

        # Draw the image at that location
        imlabel = f'image-decidx_{dec_tile_idx}-raidx_{ra_tile_idx}'
        draw_method(window, params, image_center, label=imlabel)
    print('image indices drawn:', image_indices)


'''
For any given image drawing method, pick out out multiple nonoverlapping image
regions to draw.
Parameters:
draw_method - A function that takes window, params, image_center, label as
              input and draws an image
window - Galaxy table with rectangular boundaries in RA--Dec space
params - Image drawing parameters ('Params' namedtuple)
n_images - Number of distinct images to draw
'''
def draw_random_tiles(draw_method, window, params, n_images, ignore_indices=None):
    dec_span_degrees = window['dec'].max() - window['dec'].min()
    ra_span_degrees_raw = window['ra'].max() - window['ra'].min()
    '''Convert RA span into gnomonic (TAN) span.
    For a fixed raw RA span, the gnomonic span decreases as Dec approaches +/-90 degrees.
    Thus to define a gnomonic span with the widest possible validity
    (not going beyond the bounds of the dataset),
    use the most extreme Dec value attested in the dataset.'''
    ra_span_degrees_min = ra_span_degrees_raw * np.cos(np.abs(window['dec']).max() * np.pi/180)
    dec_span_arcsec = dec_span_degrees*degrees / arcsec
    ra_span_arcsec_min = ra_span_degrees_min*degrees / arcsec
    dec_span_pixels = dec_span_arcsec / params.scale
    ra_span_pixels_min = ra_span_arcsec_min / params.scale
    # Without checking I don't know which of RA vs. Dec corresponds to x vs. y,
    # but for square images, that difference doesn't matter here.
    dec_span_tiles = dec_span_pixels / params.ngrid_y
    ra_span_tiles_min = ra_span_pixels_min / params.ngrid_x

    print('ra_span_degrees_min =', ra_span_degrees_min)
    print('dec_span_degrees =', dec_span_degrees)

    # For every desired image, pick out a random unique location in the sky,
    # and draw a coadded exposure at that location.
    rng = np.random.default_rng()
    selected_indices = set() if ignore_indices is None else ignore_indices
    report_interval = max(1, n_images//100)
    for i in range(n_images):
        # Pick out at random an image location that hasn't yet been selected
        dec_tile_idx = rng.integers(low=1, high=dec_span_tiles)
        ra_tile_idx = rng.integers(low=1, high=ra_span_tiles_min)
        while (dec_tile_idx, ra_tile_idx) in selected_indices:
            dec_tile_idx = rng.integers(low=1, high=dec_span_tiles)
            ra_tile_idx = rng.integers(low=1, high=ra_span_tiles_min)
        # Remember locations that have already been selected, to avoid repeats
        print('dec_tile_idx, ra_tile_idx =', dec_tile_idx, ra_tile_idx)
        selected_indices.add((dec_tile_idx, ra_tile_idx))

        # Specify the RA,Dec corresponding to the selected image location
        center_ra = (window['ra'].min() + 
                ra_span_degrees_min * (ra_tile_idx/ra_span_tiles_min))*degrees
        center_dec = (window['dec'].min() +
                    dec_span_degrees * (dec_tile_idx/dec_span_tiles))*degrees
        print('center_ra =', center_ra)
        print('center_dec =', center_dec)
        image_center = galsim.CelestialCoord(center_ra, center_dec)

        # Draw the image at that location
        draw_method(window, params, image_center, label='image-'+str(i))
        if (i+1)%report_interval == 0:
            print(i+1, 'images drawn')
    print('selected_indices:', selected_indices)

# Thin wrapper for draw_random_tiles
def draw_random_postage_stamps(window, params, n_images):
    draw_random_tiles(draw_postage_stamp, window, params, n_images)

# Thin wrapper for draw_random_tiles
def draw_random_scenes(window, params, n_images, ignore_indices):
    draw_random_tiles(draw_large_scene, window, params, n_images, ignore_indices)

# Thin wrapper for draw_specific_tiles
def draw_specific_scenes(window, params, image_indices):
    draw_specific_tiles(draw_large_scene, window, params, image_indices)

paramnames = ['filters', 'n_exposures', 'gain',
    'exposure_time', 'area', 'scale', 'kolmogorov_fwhm',
    'kolmogorov_variability', 'skycount_variability', 'ngrid_x', 'ngrid_y',
    'image_halfwidth']
Params = namedtuple('Params', paramnames, defaults=[None]*len(paramnames))

def main():
    # Read galaxy table from parquet file
    window = pd.read_parquet('window.parquet')

    params = dict()

    # params.filters = ['u','g','r','i','z','y']
    params['filters'] = ['i']
    # params['n_exposures'] = 100
    params['n_exposures'] = 1

    params['gain'] = 1
    params['exposure_time'] = 30 # s
    d_eff = 642.3 # cm
    params['area'] = np.pi * (d_eff/2)**2
    params['scale'] = 0.2 # arcsec/pixel
    params['kolmogorov_fwhm'] = 0.7
    # Fractional variability from exposure to exposure
    params['kolmogorov_variability'] = 0.1
    params['skycount_variability'] = 0.05

    # Large scene
    params['ngrid_x'] = 2048
    params['ngrid_y'] = 2048
    # Select large enough region to cover pixel grid and then some
    params['image_halfwidth'] = 224*arcsec

    # # Postage stamp
    # params['ngrid_x'] = 32
    # params['ngrid_y'] = 32
    # # Select large enough region to cover pixel grid and then some
    # params['image_halfwidth'] = 3.5*arcsec

    # Using a namedtuple because it's just nicer to call p.name instead of p['name'],
    # and also because (named)tuples are immutable.
    params_ = Params(**params)

    # draw_large_scene(window, params_)
    # N_IMAGES = 1
    # draw_random_postage_stamps(window, params_, N_IMAGES)
    ignore_indices = {(6, 2), (1, 2), (5, 5), (2, 4), (6, 5), (6, 1), (6, 4),
        (4, 5), (3, 6), (4, 1), (5, 6), (2, 1), (3, 3), (1, 4), (1, 5)}
    # draw_random_scenes(window, params_, N_IMAGES, ignore_indices)
    image_indices = ignore_indices
    draw_specific_scenes(window, params_, image_indices)

if __name__ == "__main__":
    main()