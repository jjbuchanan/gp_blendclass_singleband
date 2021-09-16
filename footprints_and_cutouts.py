import os
import numpy as np
import pandas as pd
from progressbar import progressbar, ProgressBar
from scipy.ndimage import gaussian_filter
from image_simulation.flux_conversion import integrated_throughput_in_band, npho_from_mag
# Importing LSST Pipeline segmaps
from astropy.io import fits
# Segmap extraction
import sep


def discrete_gaussian_kernel(fwhm, pixel_scale, halfwidth=5, integer_valued=False):
    tophat = np.zeros((2*halfwidth+1, 2*halfwidth+1))
    tophat[halfwidth, halfwidth] = 1

    # Recall that a Gaussian FWHM = 2.355*sigma
    sigma = fwhm / 2.355
    sigma_pixels = sigma / pixel_scale

    kernel = gaussian_filter(tophat, sigma=sigma_pixels, order=0,
                                   mode='constant', cval=0.0,
                                   truncate=halfwidth/sigma_pixels)

    if integer_valued:
        kernel = np.round(kernel / np.min(kernel)).astype(int)
    
    return kernel


def effective_area(psf_fwhm, pixel_scale):
    discrete_psf = discrete_gaussian_kernel(psf_fwhm, pixel_scale)
    A = np.sum(discrete_psf**2)
    return A


def expand_footprints(footprints, segmap, n_sigma, fwhm, pixel_scale):
    # npix = int(n_sigma * fwhm / (2.355 * pixel_scale)) # Old
    hw = 1 + int(n_sigma * fwhm / (2.355 * pixel_scale))

    # Expand x--y bounds stored in the footprint features,
    # while making sure to stay within the overall image borders.
    footprints_expanded = np.copy(footprints)
    for i in range(len(footprints)):
        footprints_expanded['xmin'][i] = max(0, footprints['xmin'][i]-1)
        footprints_expanded['xmax'][i] = min(segmap.shape[1]-1, footprints['xmax'][i]+1)
        footprints_expanded['ymin'][i] = max(0, footprints['ymin'][i]-1)
        footprints_expanded['ymax'][i] = min(segmap.shape[0]-1, footprints['ymax'][i]+1)
    
    # Add some padding to avoid going out of bounds when looking at neighboring pixel indices
    # segmap_withBorder = np.zeros((segmap.shape[0]+2*npix, segmap.shape[1]+2*npix)).astype(int) # Old
    segmap_withBorder = np.zeros((segmap.shape[0]+2*hw, segmap.shape[1]+2*hw)).astype(int)
    # Expand footprint segmentation maps
    for ix in range(segmap.shape[0]):
        for iy in range(segmap.shape[1]):
            center_val = segmap[ix, iy]
            if center_val != 0:
                # expansion_kernel = np.ones((2*npix+1,2*npix+1)).astype(int) # Old
                # segmap_withBorder[ix:ix+2*npix+1, iy:iy+2*npix+1] = center_val * expansion_kernel # Old
                kernel = discrete_gaussian_kernel(n_sigma*fwhm, pixel_scale, halfwidth=hw)
                expansion_kernel = (kernel/kernel.max() >= 0.5).astype(int)
                segmap_withBorder[ix:ix+2*hw+1, iy:iy+2*hw+1] *= 1 - expansion_kernel
                segmap_withBorder[ix:ix+2*hw+1, iy:iy+2*hw+1] += center_val * expansion_kernel

    # Remove padding
    # segmap_expanded = segmap_withBorder[npix:segmap_withBorder.shape[0]-npix,
    #                                     npix:segmap_withBorder.shape[1]-npix] # Old
    segmap_expanded = segmap_withBorder[hw:segmap_withBorder.shape[0]-hw,
                                        hw:segmap_withBorder.shape[1]-hw]
    
    return footprints_expanded, segmap_expanded


def merge_footprints(footprints, segmap):
    # Add some padding to avoid going out of bounds when looking at neighboring pixel indices
    segmap_withBorder = np.zeros((segmap.shape[0]+2, segmap.shape[1]+2)).astype(int)
    # Make a copy of segmap to avoid altering the original
    segmap_withBorder[1:segmap.shape[0]+1, 1:segmap.shape[1]+1] = np.copy(segmap)

    # Footprints that haven't yet been merged into others
    remaining_footprints = list(range(len(footprints)))
    
    for ix in range(segmap.shape[0]):
        for iy in range(segmap.shape[1]):
            center_val = segmap_withBorder[ix+1, iy+1]
            if center_val != 0:
                neighbors = segmap_withBorder[ix:ix+3, iy:iy+3]
                segmaps_to_merge = set(neighbors.reshape(-1)) - {0, center_val}
                # Merge all neighboring footprints into footprint center_val-1
                for i in segmaps_to_merge:
                    # Reweight barycenter
                    f_i, f_cent = footprints['flux'][i-1], footprints['flux'][center_val-1]
                    x_i, x_cent = footprints['x'][i-1], footprints['x'][center_val-1]
                    y_i, y_cent = footprints['y'][i-1], footprints['y'][center_val-1]
                    footprints['x'][center_val-1] = (f_i*x_i + f_cent*x_cent) / (f_i + f_cent)
                    footprints['y'][center_val-1] = (f_i*y_i + f_cent*y_cent) / (f_i + f_cent)
                    footprints['flux'][center_val-1] = f_i + f_cent
                    # Adjust x--y bounds
                    footprints['xmin'][center_val-1] = min(footprints['xmin'][i-1], footprints['xmin'][center_val-1])
                    footprints['xmax'][center_val-1] = max(footprints['xmax'][i-1], footprints['xmax'][center_val-1])
                    footprints['ymin'][center_val-1] = min(footprints['ymin'][i-1], footprints['ymin'][center_val-1])
                    footprints['ymax'][center_val-1] = max(footprints['ymax'][i-1], footprints['ymax'][center_val-1])
                    # Merge footprint segmentation maps
                    template = segmap_withBorder == i
                    segmap_withBorder += template * (center_val - i)
                    # Now footprint i-1 has been completely merged into footprint center_val-1
                    remaining_footprints.remove(i-1)
    
    # Remove padding
    segmap_merged = segmap_withBorder[1:segmap.shape[0]+1, 1:segmap.shape[1]+1]
    
    # Select just those footprints that haven't been merged into others
    footprints_merged = footprints[remaining_footprints]
    # For each merged footprint, set the corresponding values in the merged segmap to be
    # = 1 + the index of that footprint in footprints_merged
    for reindexed_i, orig_i in enumerate(remaining_footprints):
        # Save time by only looking at the region containing the footprint
        x_lo, x_hi = footprints_merged['xmin'][reindexed_i], footprints_merged['xmax'][reindexed_i]
        y_lo, y_hi = footprints_merged['ymin'][reindexed_i], footprints_merged['ymax'][reindexed_i]
        # Recall that SEP's x/y correpond to numpy indices 1/0, respectively.
        template = segmap_merged[y_lo:y_hi+1, x_lo:x_hi+1] == orig_i+1
        segmap_merged[y_lo:y_hi+1, x_lo:x_hi+1] += template * (reindexed_i - orig_i)
    
    return footprints_merged, segmap_merged


def make_footprints(scene, background, footprint_threshold, psf_fwhm,
                    pixel_scale, n_sigma_expand=1, background_subtracted=False):
    A = effective_area(psf_fwhm, pixel_scale)
    sqrt_A = np.sqrt(A)

    convolution_kernel = discrete_gaussian_kernel(psf_fwhm, pixel_scale,
                                                    integer_valued=True)

    sub = 0 if background_subtracted else background
    footprints, segmap = sep.extract(scene-sub, thresh=footprint_threshold,
                                        err=np.sqrt(background)*sqrt_A,
                                        deblend_cont=1.0, minarea=1,
                                        filter_kernel=convolution_kernel,
                                        filter_type='conv',
                                        segmentation_map=True)

    footprints, segmap = expand_footprints(footprints, segmap, n_sigma_expand,
                                            psf_fwhm, pixel_scale)

    footprints, segmap = merge_footprints(footprints, segmap)

    return footprints, segmap


def load_fits_array(filename):
    fits_array = None
    with fits.open(filename) as hdul:
        fits_array = hdul[0].data
    return fits_array


def summarize_footprints(image, segmap):
    # Initialize footprint_info,
    # which will contain summary values for each footprint
    n_footprints = segmap.max()
    fields = [('xmin', '<i4'), ('xmax', '<i4'), ('ymin', '<i4'),
                    ('ymax', '<i4'), ('x', '<f8'), ('y', '<f8')]
    footprint_info = np.zeros((n_footprints), dtype=fields)

    # Initialize running values for each footprint
    xmin = segmap.shape[1] * np.ones(n_footprints)
    xmax = -1 * np.ones(n_footprints)
    ymin = segmap.shape[0] * np.ones(n_footprints)
    ymax = -1 * np.ones(n_footprints)
    total_flux = np.zeros(n_footprints)
    flux_moment_x = np.zeros(n_footprints)
    flux_moment_y = np.zeros(n_footprints)

    # For each image pixel,
    # update the running values for its corresponding footprint
    for y in range(segmap.shape[0]):
        for x in range(segmap.shape[1]):
            # Recall: For SEP segmaps,
            # x/y correspond to numpy indices 1/0, respectively
            # footprint number = segmap value - 1
            i = segmap[y,x] - 1
            if i >= 0:
                xmin[i] = min(xmin[i], x)
                xmax[i] = max(xmax[i], x)
                ymin[i] = min(ymin[i], y)
                ymax[i] = max(ymax[i], y)
                total_flux[i] += image[y,x]
                flux_moment_x[i] += x * image[y,x]
                flux_moment_y[i] += y * image[y,x]

    # Store summary values for each footprint
    footprint_info['xmin'] = xmin
    footprint_info['xmax'] = xmax
    footprint_info['ymin'] = ymin
    footprint_info['ymax'] = ymax
    footprint_info['x'] = flux_moment_x / total_flux
    footprint_info['y'] = flux_moment_y / total_flux

    return footprint_info


def find_mag_coords(mag, image, true_pos, true_npho, band, n_exposures, exposure_time, area):
    # Some galaxies lie outside the image frame altogether,
    # so identify those that lie inside.
    # Exclude a 1-pixel band around the edge of the image.
    in_image_x_filter = (true_pos.x >= 1)&(true_pos.x < image.shape[1]-1)
    in_image_y_filter = (true_pos.y >= 1)&(true_pos.y < image.shape[0]-1)

    # Select only those galaxies with an AB magnitude between mag and mag-1
    # Since we've only stored the mean expected number of photons arriving
    # from each galaxy, the relevant mag must first be converted into a
    # corresponding number of photons.
    integrated_throughput = integrated_throughput_in_band(band, 'total')
    npho_mag = npho_from_mag(mag, integrated_throughput, exposure_time, area)
    npho_magMinusOne = npho_from_mag(mag-1, integrated_throughput, exposure_time, area)
    mag_filter = (true_npho[band] >= npho_mag)&(true_npho[band] < npho_magMinusOne)

    # Select just those galaxies of appropriate magnitude that lie in the image
    mag_gals_in_image = true_pos[mag_filter &
                                    in_image_x_filter & in_image_y_filter]
    # Record the pixel coordinates of those galaxies
    mag_coords = {(int(g.x), int(g.y)) for g in mag_gals_in_image.itertuples()}

    return mag_coords


def find_bright_coords(image, background, true_pos, true_npho, band, psf_fwhm,
                        pixel_scale, n_exposures, select_gold=False):
    # Some galaxies lie outside the image frame altogether,
    # so identify those that lie inside.
    # Exclude a 1-pixel band around the edge of the image.
    in_image_x_filter = (true_pos.x >= 1)&(true_pos.x < image.shape[1]-1)
    in_image_y_filter = (true_pos.y >= 1)&(true_pos.y < image.shape[0]-1)

    # See if S/N > cutoff
    # where S/N = sqrt(A_psf) * npho / sqrt(background)
    # Standard S/N cutoff is 5
    # "Gold" sample is defined by S/N > 20
    #   (https://doi.org/10.3847/1538-4357/ab042c, sec. 4.1)
    A = effective_area(psf_fwhm, pixel_scale)
    sqrt_A = np.sqrt(A)
    n_sigma = 5 if not select_gold else 20
    SoN = sqrt_A * true_npho[band]*n_exposures / np.sqrt(background)
    bright_filter = SoN >= n_sigma

    # Select just those bright galaxies that lie in the image
    bright_gals_in_image = true_pos[bright_filter &
                                    in_image_x_filter & in_image_y_filter]
    # Record the pixel coordinates of those galaxies,
    # along with their brightness (expected total npho per exposure)
    bright_coords = {(int(g.x), int(g.y)) : true_npho[band][idx]
                        for idx, g in bright_gals_in_image.iterrows()}

    return bright_coords


'''
Parameters
footprints, segmap: The output of make_footprints (i.e. sep.extract)
bright_coords: The output of find_bright_coords
mag_coords: The output of find_mag_coords

Returns
unblended_filter: List of 0s and 1s, one value for each unblended footprint,
    such that unblended footprint i has value 1 if and only if its galaxy is in
    the desired mag range.
blended_filter: List of 0s and 1s, one value for each blended footprint,
    such that blended footprint i has value 1 if and only if any of its
    galaxies are in the desired mag range.
'''
def filter_mag_footprints_any(footprints, segmap, bright_coords, mag_coords):
    unblended_filter = []
    blended_filter = []

    for i in range(len(footprints)):
        ngals_in_footprint = 0
        mag_gal_found = False
        
        x_lo, x_hi = footprints['xmin'][i], footprints['xmax'][i]
        y_lo, y_hi = footprints['ymin'][i], footprints['ymax'][i]
        
        # For every pixel in the footprint region, check if there's a bright galaxy center at that location
        # (Save time by not checking any of the pixels outside of this footprint's maximum x--y extent)
        for x in range(x_lo, x_hi+1):
            for y in range(y_lo, y_hi+1):
                # SEP's "x" direction is the second index of segmap, and "y" is the first
                # Pixels belonging to the i-th footprint (e.g., footprints[i]) have value i+1
                if segmap[y,x] == i+1:
                    if (x,y) in bright_coords:
                        ngals_in_footprint += 1
                    if (x,y) in mag_coords:
                        mag_gal_found = True

        if ngals_in_footprint == 1:
            unblended_filter.append(mag_gal_found)
        elif ngals_in_footprint > 1:
            blended_filter.append(mag_gal_found)

    return unblended_filter, blended_filter


'''
Parameters
footprints, segmap: The output of make_footprints (i.e. sep.extract)
bright_coords: The output of find_bright_coords
mag_coords: The output of find_mag_coords
rank: Integer specifying the brightness order a galaxy in the desired magnitude
    range should appear in a blended footprint. For example, rank=0 means that
    a blended footprint must have its brightest galaxy in the desired mag range
    in order to be included in blended_filter. rank=-1 means that a blended
    footprint must have its dimmest galaxy in the desired mag range in order to
    be included in blended_filter.
    If rank == 'any', call filter_mag_footprints_any() instead.

Returns
If rank == 'any', the return output matches filter_mag_footprints_any().
Otherwise,
unblended_filter: List of 0s and 1s, one value for each unblended footprint,
    such that unblended footprint i has value 1 if and only if its galaxy is in
    the desired mag range.
blended_filter: List of 0s and 1s, one value for each blended footprint,
    such that blended footprint i has value 1 if and only if its 'rank'
    brightest galaxy is in the desired mag range.
'''
def filter_mag_footprints_byRank(footprints, segmap, bright_coords, mag_coords, rank):
    if rank == 'any':
        return filter_mag_footprints_any(footprints, segmap, bright_coords, mag_coords)

    unblended_filter = []
    blended_filter = []

    for i in range(len(footprints)):
        footprint_gal_coords = []
        
        x_lo, x_hi = footprints['xmin'][i], footprints['xmax'][i]
        y_lo, y_hi = footprints['ymin'][i], footprints['ymax'][i]
        
        # For every pixel in the footprint region, check if there's a bright galaxy center at that location
        # (Save time by not checking any of the pixels outside of this footprint's maximum x--y extent)
        for x in range(x_lo, x_hi+1):
            for y in range(y_lo, y_hi+1):
                # SEP's "x" direction is the second index of segmap, and "y" is the first
                # Pixels belonging to the i-th footprint (e.g., footprints[i]) have value i+1
                if segmap[y,x] == i+1:
                    if (x,y) in bright_coords:
                        footprint_gal_coords.append((x,y))

        if len(footprint_gal_coords) == 1:
            mag_gal_found = footprint_gal_coords[0] in mag_coords
            unblended_filter.append(mag_gal_found)
        elif len(footprint_gal_coords) > 1:
            npho_from_coord = lambda coord : bright_coords[coord]
            footprint_gal_coords.sort(key=npho_from_coord, reverse=True)
            mag_gal_found = footprint_gal_coords[rank] in mag_coords
            blended_filter.append(mag_gal_found)

    return unblended_filter, blended_filter


'''
Determine which footprints are unblended, blended, or empty ("false positive").

Parameters
footprints, segmap: The output of make_footprints (i.e. sep.extract)
bright_coords: The output of find_bright_coords

Returns
false_positives, unblended_i, blended_i: The indices (in footprints) of those
    footprints which are empty, unblended, and blended, respectively.
'''
def label_footprints(footprints, segmap, bright_coords):
    # ngals_in_all_footprints = 0
    # Footprints containing no bright galaxy centers
    false_positives = []
    # Footprints containing exactly 1 bright galaxy center
    unblended_i = []
    # Footprints containing more than 1 bright galaxy center
    blended_i = []

    for i in range(len(footprints)):
        ngals_in_footprint = 0
        
        x_lo, x_hi = footprints['xmin'][i], footprints['xmax'][i]
        y_lo, y_hi = footprints['ymin'][i], footprints['ymax'][i]
        
        # For every pixel in the footprint region, check if there's a bright galaxy center at that location
        # (Save some time by not checking any of the pixels outside of this footprint's maximum x--y extent)
        for x in range(x_lo, x_hi+1):
            for y in range(y_lo, y_hi+1):
                # SEP's "x" direction is the second index of segmap, and "y" is the first
                # This agrees with the galsim/matplotlib convention
                # Pixels belonging to the i-th footprint (e.g., footprints[i]) have value i+1
                if segmap[y,x] == i+1:
                    if (x,y) in bright_coords:
                        ngals_in_footprint += 1
                        # ngals_in_all_footprints += 1
        
        if ngals_in_footprint == 0:
            false_positives.append(i)
        if ngals_in_footprint == 1:
            unblended_i.append(i)
        elif ngals_in_footprint > 1:
            blended_i.append(i)

    return false_positives, unblended_i, blended_i


def mag_filters(footprint_i, mag_i):
    mag_i = set(mag_i)

    footprint_has_mag = [i in mag_i for i in footprint_i]

    return np.array(footprint_has_mag)


'''
Parameters
scene: A 2D numpy array representing an image
footprints, segmap: The output of make_footprints (i.e. of sep.extract)
footprint_indices: A 1D numpy array containing indices of specific
    footprints to make cutouts of
halfwidth: A positive integer; width of resulting cutouts is 2*halfwidth + 1
'''
def make_cutouts(footprint_indices, scene, footprints, segmap, halfwidth, center=None):
    ngrid = scene.shape[0]
    
    cutouts = np.zeros((len(footprint_indices), 2*halfwidth+1, 2*halfwidth+1))
    
    for cutout_i, footprint_i in enumerate(footprint_indices):
        if center is None or center == 'barycenter':
            # Use the footprint intensity barycenter as the cutout center
            center_x = int(footprints['x'][footprint_i])
            center_y = int(footprints['y'][footprint_i])
        elif center == 'midrange':
            center_x = int((footprints['xmax'][footprint_i]+footprints['xmin'][footprint_i])/2)
            center_y = int((footprints['ymax'][footprint_i]+footprints['ymin'][footprint_i])/2)

        lo_x, hi_x = center_x - halfwidth, center_x + halfwidth
        lo_y, hi_y = center_y - halfwidth, center_y + halfwidth
        
        # Adjust the cutout to fit within the scene borders (0..ngrid-1), if need be
        # Exclude a 1-pixel band around the edges of the scene
        if hi_x > ngrid - 2:
            lo_x, hi_x = ngrid - 2 - 2*halfwidth, ngrid - 2
        if lo_x < 1:
            lo_x, hi_x = 1, 1 + 2*halfwidth
        if hi_y > ngrid - 2:
            lo_y, hi_y = ngrid - 2 - 2*halfwidth, ngrid - 2
        if lo_y < 1:
            lo_y, hi_y = 1, 1 + 2*halfwidth
        
        '''Zero out all pixels that don't belong to footprint i
        Reminders:
        Pixels belonging to the i-th footprint (e.g., footprints[i]) have value i+1 in segmap
        SEP's "x" direction is the second index of the pixel array, and "y" is the first'''
        footprint_filter = segmap[lo_y:hi_y+1, lo_x:hi_x+1] == footprint_i+1
        cutout = scene[lo_y:hi_y+1, lo_x:hi_x+1] * footprint_filter
        
        cutouts[cutout_i] = cutout
    
    return cutouts


def smooth(image, fwhm, pixel_scale):
    sigma_arcsec = fwhm / 2.355
    sigma_pixels = sigma_arcsec / pixel_scale
    smoothed_image = gaussian_filter(image, sigma=sigma_pixels, order=0,
                                    mode='constant', cval=0.0, truncate=5.0)
    return smoothed_image


'''
From simulated scenes and matching truth info, make footprints and return
cutouts for each of them, organized by whether the footprint is blended or
unblended.

Parameters
brightness_rank: integer or 'any', Which galaxy (ordered by brightness) in a
    blend to consider when plotting blended accuracy vs. magnitude
select_gold: boolean, Whether to only count galaxies with S/N > 20 (as opposed to 5)
cutout_center: string or None, How to center the cutouts (None->barycenter)
use_pipeline_segmap: boolean, Whether to use the DM stack's footprints
footprint_threshold: float, 5.7 roughly matches the DM stack footprints
n_sigma_expand: float, 2.3 roughly matches the DM stack footprints
hw: integer, "Half-width" (in pixels) of image cutouts, such that width = 2*hw + 1

Returns
cutouts_blended_allScenes: numpy array of shape (n_blended_allScenes, 2*hw+1, 2*hw+1),
    containing the cutouts of all blended footprints across all scenes
cutouts_unblended_allScenes: numpy array of shape (n_unblended_allScenes, 2*hw+1, 2*hw+1),
    containing the cutouts of all unblended footprints across all scenes
blended_mag_filters_allScenes: dict containing, for every mag in mags, a numpy
    array of length n_blended_allScenes, containing a 1 for every blended
    footprint with a galaxy at that mag, and 0s for all the other blended footprints
unblended_mag_filters_allScenes: dict containing, for every mag in mags, a numpy
    array of length n_unblended_allScenes, containing a 1 for every unblended
    footprint with a galaxy at that mag, and 0s for all the other unblended footprints
'''
def preprocess_scenes(brightness_rank=0, select_gold=False,
    cutout_center=None, use_pipeline_segmap=True, footprint_threshold=5.7,
    n_sigma_expand=2.3, hw=11, train=True):
    mags = (21.5,22.5,23.5,24.5,25.5,26.5)

    psf_fwhm = 0.7 # arcsec
    pixel_scale = 0.2 # arcsec/pixel
    n_exposures = 100
    exposure_time = 30
    d_eff = 642.3
    area = (np.pi/4) * d_eff**2
    # Only use a single filter band for now
    band = 'i'
    if train:
        train_images = range(10)
        IMAGE_FILENAMES = [os.path.join('simulated_scenes',f'{band}-band_image-{i}.csv') for i in train_images]
        TRUE_POS_FILENAMES = [os.path.join('simulated_scenes',f'true_pos_image-{i}.csv') for i in train_images]
        TRUE_NPHO_FILENAMES = [os.path.join('simulated_scenes',f'true_npho_allbands_image-{i}.csv') for i in train_images]
        PIPELINE_SEGMAP_FILENAMES = [os.path.join('simulated_scenes',f'pipeline_segmap_{i}.fits') for i in train_images]
    else:
        test_images = (11,12,13,15,16,19,21,23,24,25)
        IMAGE_FILENAMES = [os.path.join('simulated_scenes',f'{band}-band_image-{i}.csv') for i in test_images]
        TRUE_POS_FILENAMES = [os.path.join('simulated_scenes',f'true_pos_image-{i}.csv') for i in test_images]
        TRUE_NPHO_FILENAMES = [os.path.join('simulated_scenes',f'true_npho_allbands_image-{i}.csv') for i in test_images]
        PIPELINE_SEGMAP_FILENAMES = [os.path.join('simulated_scenes',f'pipeline_segmap_{i}.fits') for i in test_images]

    # Make footprints at this threshold
    if use_pipeline_segmap:
        print('Using pipeline segmaps')
    else:
        print('footprint_threshold:', footprint_threshold)
    footprints_allScenes = []
    segmap_allScenes = []
    unblended_i_allScenes = []
    blended_i_allScenes = []
    unblended_mag_filters_allScenes = {mag : [] for mag in mags}
    blended_mag_filters_allScenes = {mag : [] for mag in mags}
    n_unblended_allScenes = 0
    n_blended_allScenes = 0
    print('Making footprints on scenes')
    print(len(IMAGE_FILENAMES))
    for scene_idx in progressbar(range(len(IMAGE_FILENAMES))):
        # Load up the files for this scene
        image_filename = IMAGE_FILENAMES[scene_idx]
        true_pos_filename = TRUE_POS_FILENAMES[scene_idx]
        true_npho_filename = TRUE_NPHO_FILENAMES[scene_idx]
        scene = np.loadtxt(image_filename, delimiter=',')
        background = np.median(scene)
        # Extract footprints
        if use_pipeline_segmap:
            segmap = load_fits_array(PIPELINE_SEGMAP_FILENAMES[scene_idx])
            footprints = summarize_footprints(scene, segmap)
        else:
            footprints, segmap = make_footprints(scene, background,
                                footprint_threshold, psf_fwhm, pixel_scale,
                                n_sigma_expand=n_sigma_expand)
        # Establish the true positions of galaxies in the scene
        true_pos = pd.read_csv(true_pos_filename)
        true_npho = pd.read_csv(true_npho_filename)
        bright_coords = find_bright_coords(scene, background, true_pos,
        true_npho, band, psf_fwhm, pixel_scale, n_exposures, select_gold)
        # Label the footprints by how many galaxies they contain
        false_positives, unblended_i, blended_i = label_footprints(
                                        footprints, segmap, bright_coords)
        # Store data for use below
        footprints_allScenes.append(footprints)
        segmap_allScenes.append(segmap)
        unblended_i_allScenes.append(unblended_i)
        blended_i_allScenes.append(blended_i)
        n_unblended_allScenes += len(unblended_i)
        n_blended_allScenes += len(blended_i)

        for mag in mags:
            mag_coords = find_mag_coords(mag, scene, true_pos, true_npho,
                band, n_exposures, exposure_time, area)
            unblended_mag_filter, blended_mag_filter = filter_mag_footprints_byRank(
                footprints, segmap, bright_coords, mag_coords, brightness_rank)
            unblended_mag_filters_allScenes[mag] += unblended_mag_filter
            blended_mag_filters_allScenes[mag] += blended_mag_filter

    # Convert mag filters from lists to numpy arrays
    unblended_mag_filters_allScenes = {mag : np.array(unblended_mag_filters_allScenes[mag])
                                        for mag in mags}
    blended_mag_filters_allScenes = {mag : np.array(blended_mag_filters_allScenes[mag])
                                        for mag in mags}

    # Print some summary values
    print('n_unblended_allScenes:', n_unblended_allScenes,
        'n_blended_allScenes:', n_blended_allScenes)
    for mag in mags:
        print('mag:', mag)
        print('  n_unblended:', np.sum(unblended_mag_filters_allScenes[mag]),
            'n_blended:', np.sum(blended_mag_filters_allScenes[mag]))

    # Make cutouts for this image width
    print('Image width:', 2*hw+1)
    cutouts_unblended_allScenes = np.zeros((n_unblended_allScenes,
                                                2*hw+1, 2*hw+1))
    cutouts_blended_allScenes = np.zeros((n_blended_allScenes,
                                                2*hw+1, 2*hw+1))
    # Indices used for storing cutouts in cutouts_*blended_allScenes
    unblended_lo, unblended_hi = 0, 0
    blended_lo, blended_hi = 0, 0
    print('Making cutouts on scenes')
    for scene_idx in progressbar(range(len(blended_i_allScenes))):
        # Update *blended_hi
        unblended_hi += len(unblended_i_allScenes[scene_idx])
        blended_hi += len(blended_i_allScenes[scene_idx])
        # Load data for this scene
        image_filename = IMAGE_FILENAMES[scene_idx]
        scene = np.loadtxt(image_filename, delimiter=',')
        footprints = footprints_allScenes[scene_idx]
        segmap = segmap_allScenes[scene_idx]
        unblended_i = unblended_i_allScenes[scene_idx]
        blended_i = blended_i_allScenes[scene_idx]
        # Make and store cutouts
        cutouts_unblended_allScenes[unblended_lo:unblended_hi] = make_cutouts(
            unblended_i, scene, footprints, segmap, hw, cutout_center)
        cutouts_blended_allScenes[blended_lo:blended_hi] = make_cutouts(
            blended_i, scene, footprints, segmap, hw, cutout_center)
        # Update *blended_lo
        unblended_lo, blended_lo = unblended_hi, blended_hi

    return (cutouts_blended_allScenes, cutouts_unblended_allScenes,
        blended_mag_filters_allScenes, unblended_mag_filters_allScenes)
