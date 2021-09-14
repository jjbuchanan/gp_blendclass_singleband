import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar, ProgressBar
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix
from image_simulation.flux_conversion import integrated_throughput_in_band, npho_from_mag
# Importing LSST Pipeline segmaps
from astropy.io import fits
# Segmap extraction
import sep
# GP model
from MuyGPyS.examples.classify import do_classify
from MuyGPyS.examples.regress import do_regress
# Alternative model: Logistic regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA


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


def label_footprints_and_compare_peaks_mag(footprints, segmap, bright_coords,
    mag_coords, rank, peaks_by_footprint):
    unblended_i_mag = []
    blended_i_mag = []

    # Check the relationship to peaks
    n_unblended_onepeak_mag = 0
    n_blended_multipeak_mag = 0

    for i in range(len(footprints)):
        footprint_gal_coords = []
        
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
                        footprint_gal_coords.append((x,y))
        
        npeaks = len(peaks_by_footprint[i])
        if len(footprint_gal_coords) == 1:
            if footprint_gal_coords[0] in mag_coords:
                unblended_i_mag.append(i)
                if npeaks == 1:
                    n_unblended_onepeak_mag += 1
        elif len(footprint_gal_coords) > 1:
            npho_from_coord = lambda coord : bright_coords[coord]
            footprint_gal_coords.sort(key=npho_from_coord, reverse=True)
            if footprint_gal_coords[rank] in mag_coords:
                blended_i_mag.append(i)
                if npeaks > 1:
                    n_blended_multipeak_mag += 1
    
    return (unblended_i_mag, blended_i_mag, n_unblended_onepeak_mag,
        n_blended_multipeak_mag)


'''
In addition to labeling the footprints (just as in label_footprints), check to
see if the number of peaks in each footprint corresponds to its classification.

Parameters
All those of label_footprints, plus
peaks_by_footprint: The output of find_peaks

Returns
All items returned by label_footprints, plus
ngals_in_all_footprints: The total number of galaxies contained in any
    footprint.
n_unblended_onepeak: The number of unblended footprints with exactly one peak.
n_blended_multipeak: The number of blended footprints with more than one peak.
'''
def label_footprints_and_compare_peaks(footprints, segmap, bright_coords,
                                        peaks_by_footprint):
    # Footprints containing no bright galaxy centers
    false_positives = []
    # Footprints containing exactly 1 bright galaxy center
    unblended_i = []
    # Footprints containing more than 1 bright galaxy center
    blended_i = []

    # Check the relationship to peaks
    n_unblended_onepeak = 0
    n_blended_multipeak = 0

    # Count the total number of galaxies contained in all footprints
    ngals_in_all_footprints = 0

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
                        ngals_in_all_footprints += 1
        
        npeaks = len(peaks_by_footprint[i])
        if ngals_in_footprint == 0:
            false_positives.append(i)
        if ngals_in_footprint == 1:
            unblended_i.append(i)
            if npeaks == 1:
                n_unblended_onepeak += 1
        elif ngals_in_footprint > 1:
            blended_i.append(i)
            if npeaks > 1:
                n_blended_multipeak += 1
        
    return (false_positives, unblended_i, blended_i, ngals_in_all_footprints,
            n_unblended_onepeak, n_blended_multipeak)


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

'''
Parameters
labels: a 1D numpy array containing integer values in the range
(0..n_classes-1)
n_classes: if None, this will be inferred from the largest value in labels

Returns:
A 2D numpy array with a one-hot representation of each class label.
Specifically, label 0 -> [1., -1., ...], 1 -> [-1., 1., ...], etc.
'''
def onehot_from_class_labels(labels, n_classes=None):
    if n_classes is None:
        n_classes = np.max(labels) + 1
    return 2 * np.eye(n_classes)[labels] - 1.

'''
Alias for onehot_from_class_labels for the case when n_classes=2
Convert 0 to [1., -1.], 1 to [-1., 1.]

Parameters
labels: a 1D numpy array containing only 0s and 1s
'''
def onehot_from_binary(labels):
    return onehot_from_class_labels(labels, n_classes=2)


'''
Convert 0 to [-1], 1 to [1]

Parameters
labels: a 1D numpy array containing only 0s and 1s

Returns
A 2D numpy array of shape (len(labels), 1)
'''
def regress_target_from_binary(labels):
    target = 2 * labels - 1
    return target.reshape(-1, 1)

'''
Convert  [1., -1., ...] to 0, [-1., 1., ...] to 1, etc.

Parameters
onehot: a 2D numpy array containing integer values in the range
(0..n_classes-1)
'''
def labels_from_onehot(onehot):
    return np.argmax(onehot, axis=1)


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


def mag_filters(footprint_i, mag_i):
    mag_i = set(mag_i)

    footprint_has_mag = [i in mag_i for i in footprint_i]

    return np.array(footprint_has_mag)


'''
Generate train and test datasets in a format that MuyGPyS can handle.

Parameters
pos: Numpy array of positive data examples, one row per example
neg: Numpy array of negative data examples, one row per example
n_folds: Split the data into this many equally-sized "folds"; for each fold,
    generate a test set consisting of that fold and a training set consisting
    of all the other data.
max_train_size: If not None, cap the training set size at max_train_size
truncate: Cap the feature values at the specified quantile (w.r.t. the
    training data), before normalization (if any).
norm_order: If not None, normalize the data according to the specified
    strategy. 'max_val' means divide all features in the dataset by the
    largest feature value in the training set. A positive integer means
    to divide all features in the dataset by the largest row norm in the
    training set, where the row norm is computed for each example using
    sum(abs(features)**norm_order)**(1./norm_order)
balance: If True, use all the examples in the smaller of (pos, neg),
    and select a random subset of the larger, so that the returned sets are
    equally-sized.
'''
def binaryclass_muygpys_kfold_datasets(pos, neg, mag_filters_pos, mag_filters_neg,
            n_folds=5, max_train_size=None, truncate=1.0, norm_order='max_val',
            balance=True, regress=False):
    if balance:
        rng = np.random.default_rng()
        if len(pos) < len(neg):
            idx = rng.choice(np.arange(len(neg)), size=len(pos), replace=False)
            neg = neg[idx]
            mag_filters_neg = {mag : mag_filters_neg[mag][idx] for mag in mag_filters_neg}
        elif len(neg) < len(pos):
            idx = rng.choice(np.arange(len(pos)), size=len(neg), replace=False)
            pos = pos[idx]
            mag_filters_pos = {mag : mag_filters_pos[mag][idx] for mag in mag_filters_pos}
    
    # Merge positive and negative examples into a common dataset
    X = np.concatenate((pos, neg))
    mag_filters = {mag : np.concatenate((mag_filters_pos[mag], mag_filters_neg[mag]))
                    for mag in mag_filters_pos}
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))).astype(int)
    output = regress_target_from_binary(y) if regress else onehot_from_binary(y)
    # Flatten
    X = X.reshape((X.shape[0], -1))
    
    if n_folds == 'loo':
        cv_strategy = LeaveOneOut()
    else:
        cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for train_idx, test_idx in cv_strategy.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        output_train, output_test = output[train_idx], output[test_idx]
        test_mag_filter = {mag : mag_filters[mag][test_idx] for mag in mag_filters}

        if max_train_size is not None:
            X_train = X_train[:max_train_size]
            y_train = y_train[:max_train_size]
            output_train = output_train[:max_train_size]
        
        # Truncate pixel values
        if 1 > truncate > 0:
            max_X_val = np.quantile(X_train, truncate)
            X_train = np.maximum(X_train, max_X_val)
            X_test = np.maximum(X_test, max_X_val)

        # Normalize pixel values
        norm = 1.0
        if norm_order == 'max_val':
            norm = np.max(X_train)
        elif norm_order is not None:
            norm = np.max(np.linalg.norm(X_train, ord=norm_order, axis=1))
        X_train /= norm
        X_test /= norm
        
        train = {'input': X_train, 'output': output_train, 'lookup': y_train}
        test = {'input': X_test, 'output': output_test, 'lookup': y_test}
        
        yield train, test, test_mag_filter


'''
Create train and test datasets in a format that MuyGPyS can handle.

Parameters
pos: Numpy array of positive data examples, one row per example
neg: Numpy array of negative data examples, one row per example
max_train_size: If not None, cap the training set size at max_train_size
truncate: Cap the feature values at the specified quantile (w.r.t. the
    training data), before normalization (if any).
norm_order: If not None, normalize the data according to the specified
    strategy. 'max_val' means divide all features in the dataset by the
    largest feature value in the training set. A positive integer means
    to divide all features in the dataset by the largest row norm in the
    training set, where the row norm is computed for each example using
    sum(abs(features)**norm_order)**(1./norm_order)
balance: If True, use all the examples in the smaller of (pos, neg),
    and select a random subset of the larger, so that the returned sets are
    equally-sized.
'''
def binaryclass_muygpys_datasets(pos, neg,
    pos_test, neg_test, mag_filters_pos_test, mag_filters_neg_test,
    max_train_size=None, truncate=1.0, norm_order='max_val',
    balance=True, regress=False):
    if balance:
        rng = np.random.default_rng()
        if len(pos) < len(neg):
            neg = rng.choice(neg, size=len(pos), replace=False)
        elif len(neg) < len(pos):
            pos = rng.choice(pos, size=len(neg), replace=False)
    
    # Merge positive and negative examples into a common dataset
    X_train = np.concatenate((pos, neg))
    y_train = np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))).astype(int)
    onehot_train = regress_target_from_binary(y_train) if regress else onehot_from_binary(y_train)
    
    X_test = np.concatenate((pos_test, neg_test))
    y_test = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test)))).astype(int)
    onehot_test = regress_target_from_binary(y_test) if regress else onehot_from_binary(y_test)
    mag_filters_test = {mag : np.concatenate((mag_filters_pos_test[mag], mag_filters_neg_test[mag]))
                    for mag in mag_filters_pos_test}
    
    # Flatten
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    if max_train_size is not None:
        X_train = X_train[:max_train_size]
        y_train = y_train[:max_train_size]
        onehot_train = onehot_train[:max_train_size]
    
    # Truncate pixel values
    if 1 > truncate > 0:
        max_X_val = np.quantile(X_train, truncate)
        X_train = np.maximum(X_train, max_X_val)
        X_test = np.maximum(X_test, max_X_val)

    # Normalize pixel values
    norm = 1.0
    if norm_order == 'max_val':
        norm = np.max(X_train)
    elif norm_order is not None:
        norm = np.max(np.linalg.norm(X_train, ord=norm_order, axis=1))
    X_train /= norm
    X_test /= norm
    
    train = {'input': X_train, 'output': onehot_train, 'lookup': y_train}
    test = {'input': X_test, 'output': onehot_test, 'lookup': y_test}
    
    return train, test, mag_filters_test


def find_peaks(footprints, image, segmap, allow_neighbors=False):
    peaks_by_footprint = dict()
    
    for i in range(len(footprints)):
        # Save time by only looking at the region containing the footprint
        x_lo, x_hi = footprints['xmin'][i], footprints['xmax'][i]
        y_lo, y_hi = footprints['ymin'][i], footprints['ymax'][i]
        # Recall that SEP's x/y correpond to numpy indices 1/0, respectively.
        footprint_filter = segmap[y_lo:y_hi+1, x_lo:x_hi+1] == i+1
        highlight = image[y_lo:y_hi+1, x_lo:x_hi+1] * footprint_filter
        
        # Add some padding to avoid going out of bounds when looking at neighboring pixel indices
        highlight_withBorder = np.zeros((highlight.shape[0]+2, highlight.shape[1]+2)).astype(int)
        highlight_withBorder[1:highlight.shape[0]+1, 1:highlight.shape[1]+1] = highlight
        
        peaks = []
        # For every pixel in the footprint, check to see if it's a peak
        for i0 in range(1, highlight.shape[0]+1):
            for i1 in range(1, highlight.shape[1]+1):
                # Avoid calling something a "peak" if it's just a zero (not in the footprint) surrounded by zeros
                if highlight_withBorder[i0, i1] != 0:
                    # Check to see if this pixel is at least as intense as its 8 nearest neighbors
                    if highlight_withBorder[i0, i1] == np.max(highlight_withBorder[i0-1:i0+2, i1-1:i1+2]):
                        # For every peak, record its image pixel array indices
                        # Index in original image
                        #  = index in highlight_withBorder + offset of highlight_withBorder w.r.t. image
                        #  = index in highlight_withBorder + offset of highlight w.r.t. image - 1
                        peak = (y_lo-1+i1, x_lo-1+i0)
                        # If desired:
                        # Check to see if an immediately adjacent peak as already been recorded,
                        # and if so skip this peak.
                        add_this_peak = True
                        if not allow_neighbors:
                            neighbors = [(peak[0]+1, peak[1]), (peak[0]+1, peak[1]+1), (peak[0]+1, peak[1]-1),
                                        (peak[0]-1, peak[1]), (peak[0]-1, peak[1]+1), (peak[0]-1, peak[1]-1),
                                        (peak[0], peak[1]+1), (peak[0], peak[1]-1)]
                            neighbor_already_added = False
                            for neighbor in neighbors:
                                if neighbor in peaks:
                                    neighbor_already_added = True
                                    break
                            if neighbor_already_added:
                                add_this_peak = False
                        if add_this_peak:
                            peaks.append((y_lo-1+i1, x_lo-1+i0))
        peaks_by_footprint[i] = peaks
    
    return peaks_by_footprint


def smooth(image, fwhm, pixel_scale):
    sigma_arcsec = fwhm / 2.355
    sigma_pixels = sigma_arcsec / pixel_scale
    smoothed_image = gaussian_filter(image, sigma=sigma_pixels, order=0,
                                    mode='constant', cval=0.0, truncate=5.0)
    return smoothed_image


def include_flips(data, shuffle=True):
    # Assume input images started off square and have been flattened
    image_width = int(np.sqrt(data['input'].shape[1]))
    # Un-flatten the input images, for easier manipulation
    orig = data['input'].reshape((data['input'].shape[0], image_width, image_width))
    
    # Apply flips
    horiz = orig[:, :, ::-1]
    vert = orig[:, ::-1, :]
    r180 = orig[:, ::-1, ::-1]
    trans = np.transpose(orig, (0, 2, 1))
    r90 = trans[:, ::-1, :]
    r270 = trans[:, :, ::-1]
    ntrans = trans[:, ::-1, ::-1]
    
    all_flips = [orig, horiz, vert, r180, trans, r90, r270, ntrans]
    
    # Merge into a single dataset
    augmented = dict()
    augmented['input'] = np.concatenate(all_flips)
    augmented['input'] = augmented['input'].reshape((augmented['input'].shape[0], -1))
    augmented['output'] = np.concatenate([data['output']]*len(all_flips))
    augmented['lookup'] = np.concatenate([data['lookup']]*len(all_flips))
    
    # Shuffle
    if shuffle:
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(np.arange(len(augmented['lookup'])))
        for item in ('input', 'output', 'lookup'):
            augmented[item] = augmented[item][shuffled_indices]
    
    return augmented


def main_validate():
    model_name = 'Gaussian process'
    # model_name = 'Logistic regression'
    regress = True
    print(model_name)
    print()

    sigma_sq_grid = (1e6, 1.2e6, 1.4e6, 1.7e6, 2e6, 2.3e6, 2.7e6, 3e6, 3.3e6, 3.7e6, 4e6, 5e6, 6e6, 7e6)

    use_pipeline_segmap = True
    footprint_thresholds = (5,)
    # footprint_thresholds = (5.7,) # Roughly matches Pipeline footprints
    n_sigma_expand = 1
    # n_sigma_expand = 2.3 # Roughly matches Pipeline footprints

    n_trials = 5

    n_folds = 400 # Fine for GP classification
    # n_folds = 50 # Better for logistic regression classification
    # n_folds = 'loo'

    mags = (21.5,22.5,23.5,24.5,25.5,26.5)
    brightness_rank = 0
    # brightness_rank = 'any'

    # Whether to only count galaxies with S/N > 20 (as opposed to 5)
    select_gold = False
    # Whether to augment the training data by including flipped images
    augment = False
    # How to center the cutouts (None->barycenter)
    cutout_center = None


    # embed_dims = (8,)
    # embed_dims = (8,15,40)
    # embed_dims = (6,7,8,9,10,11)
    image_hws = (11,) # pixels
    # image_hws = (9,11,13,15) # pixels
    # image_hws = (10,11,12,13,14,15) # pixels

    eps_values = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    length_scale_values = (2e0, 6e0, 2e1, 6e1, 2e2, 6e2, 2e3)
    nu_values = (3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1)

    # # Best values for GP classifier, max_val normalization, footprint_threshold=5
    embed_dim = 8
    hw = 11
    # eps = 3e-6
    # length_scale = 2e1
    # nu = 1e1

    psf_fwhm = 0.7 # arcsec
    pixel_scale = 0.2 # arcsec/pixel
    n_exposures = 100
    exposure_time = 30
    d_eff = 642.3
    area = (np.pi/4) * d_eff**2
    # Only use a single filter band for now
    band = 'i'

    PATH = os.path.join('data','simulated_scenes')
    train_images = range(10)
    IMAGE_FILENAMES = [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in train_images]
    TRUE_POS_FILENAMES = [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in train_images]
    TRUE_NPHO_FILENAMES = [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in train_images]
    PIPELINE_SEGMAP_FILENAMES = [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in train_images]
    # test_images = (11,12,13,15,16,19,21,23,24,25)
    # TEST_IMAGE_FILENAMES = [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in test_images]
    # TEST_TRUE_POS_FILENAMES = [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in test_images]
    # TEST_TRUE_NPHO_FILENAMES = [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in test_images]
    # TEST_PIPELINE_SEGMAP_FILENAMES = [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in test_images]
    # OUTFILE_NAME = 'confusion_matrices'

    confusion_matrices_byThreshold = dict()

    for footprint_threshold in footprint_thresholds:
    # for eps in eps_values:
        if use_pipeline_segmap:
            print('Using pipeline segmaps')
        else:
            print('footprint_threshold:', footprint_threshold)
        # print('eps:', eps)

        # Make footprints at this threshold
        footprints_allScenes = []
        segmap_allScenes = []
        unblended_i_allScenes = []
        blended_i_allScenes = []
        unblended_mag_filters_allScenes = {mag : [] for mag in mags}
        blended_mag_filters_allScenes = {mag : [] for mag in mags}
        n_unblended_allScenes = 0
        n_blended_allScenes = 0
        print('Making footprints on scenes')
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

        # Evaluate classifier for different combinations of hyperparameters
        confusion_matrices_byImageHW = dict()
        for hw in image_hws:
        # for length_scale in length_scale_values:
            print('Image width:', 2*hw+1)
            # print('length_scale:', length_scale)

            # Make cutouts for this image width
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

            confusion_matrices_byPCAdim = dict()

            cross_entropy = {sigma_sq : 0 for sigma_sq in sigma_sq_grid}
            for trial in range(n_trials):
            # for embed_dim in embed_dims:
            # for nu in nu_values:
                print()
                print('Trial', trial+1, '/', n_trials)
                # print('embed_dim:', embed_dim)
                # print('nu:', nu)
                embed_method = None if embed_dim is None else 'pca'

                # Create n_folds many train--test splits and evaluate
                # classifier on each.
                total_confusion_matrix = np.zeros((2,2), dtype=int)
                total_confusion_matrix_mag = {mag : np.zeros((2,2), dtype=int) for mag in mags}
                print('K-fold cross validation')
                bar = ProgressBar(max_value=n_folds, redirect_stdout=True)
                for train, test, test_mag_filter in binaryclass_muygpys_kfold_datasets(
                    cutouts_blended_allScenes, cutouts_unblended_allScenes,
                    blended_mag_filters_allScenes, unblended_mag_filters_allScenes,
                    n_folds=n_folds, regress=regress):
                    if augment:
                        train = include_flips(train)

                    predicted_labels = np.zeros(test['lookup'].shape)
                    # Logistic regression
                    if model_name == 'Logistic regression':
                        # First need to reduce dimensionality to make tractable
                        pca = PCA(n_components=embed_dim)
                        train['input'] = pca.fit_transform(train['input'])
                        test['input'] = pca.transform(test['input'])
                        lr = LogisticRegressionCV(cv=5, penalty='l2', scoring='accuracy', max_iter=10000, n_jobs=-1)
                        lr = lr.fit(train['input'], train['lookup'])
                        predicted_labels = lr.predict(test['input'])
                    # GP classification
                    if model_name == 'Gaussian process':
                        get_posterior = do_regress if regress else do_classify
                        kwargs = {'variance_mode': 'diagonal'} if regress else {'opt_batch_size': 500}
                        posterior = get_posterior(train, test, nn_count=50, 
                            # hyper_dict={'eps': 1e-5, 'length_scale': 5e1, 'nu': 1e0},
                            # hyper_dict={'eps': 3e-6, 'length_scale': 2e1, 'nu': 1e1},
                            hyper_dict={'eps': eps, 'length_scale': length_scale, 'nu': nu},
                            # optim_bounds={'eps': (1e-8, 3e-3), 'length_scale': (5e-1, 1e2), 'nu': (1e-2, 1e1)},
                            # optim_bounds={'eps': (1e-8, 1e-2), 'length_scale': (5e-1, 20)},
                            # opt_batch_size=500,
                            kern='matern',
                            embed_method=embed_method, embed_dim=embed_dim,
                            loss_method='log', nn_kwargs={'nn_method': 'exact',
                            'p': 2}, verbose=False,
                            **kwargs)
                        predicted_labels = (posterior[0] > 0).reshape(-1) if regress else np.argmax(posterior, axis=1)
                        if regress:
                            y = test['lookup']
                            mean = posterior[0].reshape(-1)
                            for sigma_sq in sigma_sq_grid:
                                std = np.sqrt(posterior[1] * sigma_sq)
                                p = norm.sf(0, mean, std)
                                cross_entropy[sigma_sq] -= np.log(p[y == 1]).sum() + np.log(1 - p[y == 0]).sum()

                    c = confusion_matrix(test['lookup'], predicted_labels)
                    if c.shape == (1,1):
                        c = c * np.eye(2, dtype=int) # Now c.shape == (2,2)
                        # Zero out whichever c entry doesn't match this test example
                        clear_idx = 1 - test['lookup'][0]
                        c[clear_idx, clear_idx] = 0
                    if c.shape == (2,2):
                        total_confusion_matrix += c

                    for mag in mags:
                        c = confusion_matrix(test['lookup'][test_mag_filter[mag]],
                                predicted_labels[test_mag_filter[mag]])
                        if c.shape == (1,1):
                            c = c * np.eye(2, dtype=int) # Now c.shape == (2,2)
                            # Zero out whichever c entry doesn't match this test example
                            clear_idx = 1 - test['lookup'][test_mag_filter[mag]][0]
                            c[clear_idx, clear_idx] = 0
                        if c.shape == (2,2):
                            total_confusion_matrix_mag[mag] += c

                    bar.update(bar.value + 1)

                print()
                # Report results
                tn, fp, fn, tp = total_confusion_matrix.ravel()
                acc, posacc, negacc = (tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tn/(tn+fp)
                accerr = np.sqrt(acc*(1-acc) / (tp+tn+fp+fn))
                posaccerr = np.sqrt(posacc*(1-posacc) / (tp+fn))
                negaccerr = np.sqrt(negacc*(1-negacc) / (tn+fp))
                print('tn, fp =', tn, fp)
                print('fn, tp =', fn, tp)
                print(f'Acc: {acc:.5f}, Posacc: {posacc:.5f}, Negacc: {negacc:.5f}')
                print(f'Accerr: {accerr:.5f}, Posaccerr: {posaccerr:.5f}, Negaccerr: {negaccerr:.5f}')

                acc_mags, accerr_mags = [], []
                posacc_mags, posaccerr_mags = [], []
                negacc_mags, negaccerr_mags = [], []
                tn_mags, fp_mags, fn_mags, tp_mags = [], [], [], []
                for mag in mags:
                    tn, fp, fn, tp = total_confusion_matrix_mag[mag].ravel()
                    tn_mags.append(tn)
                    fp_mags.append(fp)
                    fn_mags.append(fn)
                    tp_mags.append(tp)
                    acc_mags.append((tp+tn)/(tp+tn+fp+fn))
                    accerr_mags.append(np.sqrt(acc_mags[-1]*(1-acc_mags[-1]) / (tp+tn+fp+fn)))
                    posacc_mags.append(tp/(tp+fn))
                    posaccerr_mags.append(np.sqrt(posacc_mags[-1]*(1-posacc_mags[-1]) / (tp+fn)))
                    negacc_mags.append(tn/(tn+fp))
                    negaccerr_mags.append(np.sqrt(negacc_mags[-1]*(1-negacc_mags[-1]) / (tn+fp)))
                print('mags =', mags)
                print('Blend brightness rank for mags:', brightness_rank)
                print('tn =', tn_mags)
                print('fp =', fp_mags)
                print('fn =', fn_mags)
                print('tp =', tp_mags)
                print('posacc =', posacc_mags)
                print('posaccerr =', posaccerr_mags)
                print('negacc =', negacc_mags)
                print('negaccerr =', negaccerr_mags)

                # print('Total confusion matrix:')
                # print(total_confusion_matrix)
                # Store confusion matrix
                confusion_matrices_byPCAdim[embed_dim] = total_confusion_matrix
            if regress:
                print()
                print('cross entropy loss')
                for sigma_sq in sigma_sq_grid:
                    print('sigma_sq =', sigma_sq, 'cross_entropy =', cross_entropy[sigma_sq]/n_trials)
            confusion_matrices_byImageHW[hw] = confusion_matrices_byPCAdim
        confusion_matrices_byThreshold[footprint_threshold] = confusion_matrices_byPCAdim

    # # Write confusion matrices to a file
    # with open(OUTFILE_NAME, 'w') as out:
    #     pickle.dump(confusion_matrices_byThreshold, out)


def main_test():
    model_name = 'Gaussian process'
    regress = True
    # model_name = 'Logistic regression'

    n_trials = 100

    footprint_type = 'pipeline'
    footprint_params = {'pipeline': (True, None, None, 2300000.0),
    'custom': (False, 5.7, 2.3, 3300000.0),
    'variant': (False, 5, 1, 2700000.0)}
    use_pipeline_segmap, footprint_threshold, n_sigma_expand, sigma_sq = (
        footprint_params[footprint_type])

    p_stepsize = 0.1
    p_lo_bins = np.arange(start=0, stop=1, step=p_stepsize)

    mags = (21.5,22.5,23.5,24.5,25.5,26.5)
    brightness_rank = 0
    # brightness_rank = 'any'

    # Whether to only count galaxies with S/N > 20 (as opposed to 5)
    select_gold = False
    # Whether to augment the training data by including flipped images
    augment = False
    # How to center the cutouts (None->barycenter)
    cutout_center = None

    # # Best values for GP classifier, max_val normalization, footprint_threshold=5
    embed_dim = 8
    hw = 11
    eps = 3e-6
    length_scale = 2e1
    nu = 1e1

    psf_fwhm = 0.7 # arcsec
    pixel_scale = 0.2 # arcsec/pixel
    n_exposures = 100
    exposure_time = 30
    d_eff = 642.3
    area = (np.pi/4) * d_eff**2
    # Only use a single filter band for now
    band = 'i'

    PATH = os.path.join('data','simulated_scenes')
    train_images = range(10)
    IMAGE_FILENAMES = [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in train_images]
    TRUE_POS_FILENAMES = [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in train_images]
    TRUE_NPHO_FILENAMES = [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in train_images]
    PIPELINE_SEGMAP_FILENAMES = [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in train_images]
    test_images = (11,12,13,15,16,19,21,23,24,25)
    TEST_IMAGE_FILENAMES = [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in test_images]
    TEST_TRUE_POS_FILENAMES = [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in test_images]
    TEST_TRUE_NPHO_FILENAMES = [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in test_images]
    TEST_PIPELINE_SEGMAP_FILENAMES = [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in test_images]

    if use_pipeline_segmap:
        print('Using pipeline segmaps')
    else:
        print('footprint_threshold:', footprint_threshold)
    print()

    # Make training footprints at this threshold
    footprints_allScenes = []
    segmap_allScenes = []
    unblended_i_allScenes = []
    blended_i_allScenes = []
    n_unblended_allScenes = 0
    n_blended_allScenes = 0
    unblended_mag_filters_allScenes = {mag : [] for mag in mags}
    blended_mag_filters_allScenes = {mag : [] for mag in mags}
    print('Making footprints on training scenes')
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

    # Make test footprints at this threshold
    footprints_test_allScenes = []
    segmap_test_allScenes = []
    unblended_i_test_allScenes = []
    blended_i_test_allScenes = []
    n_unblended_test_allScenes = 0
    n_blended_test_allScenes = 0
    unblended_mag_filters_test_allScenes = {mag : [] for mag in mags}
    blended_mag_filters_test_allScenes = {mag : [] for mag in mags}
    print('Making footprints on test scenes')
    for scene_idx in progressbar(range(len(TEST_IMAGE_FILENAMES))):
        # Load up the files for this scene
        image_filename = TEST_IMAGE_FILENAMES[scene_idx]
        true_pos_filename = TEST_TRUE_POS_FILENAMES[scene_idx]
        true_npho_filename = TEST_TRUE_NPHO_FILENAMES[scene_idx]
        scene = np.loadtxt(image_filename, delimiter=',')
        background = np.median(scene)
        # Extract footprints
        if use_pipeline_segmap:
            segmap = load_fits_array(TEST_PIPELINE_SEGMAP_FILENAMES[scene_idx])
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
        footprints_test_allScenes.append(footprints)
        segmap_test_allScenes.append(segmap)
        unblended_i_test_allScenes.append(unblended_i)
        blended_i_test_allScenes.append(blended_i)
        n_unblended_test_allScenes += len(unblended_i)
        n_blended_test_allScenes += len(blended_i)

        for mag in mags:
            mag_coords = find_mag_coords(mag, scene, true_pos, true_npho,
                band, n_exposures, exposure_time, area)
            unblended_mag_filter, blended_mag_filter = filter_mag_footprints_byRank(
                footprints, segmap, bright_coords, mag_coords, brightness_rank)
            unblended_mag_filters_test_allScenes[mag] += unblended_mag_filter
            blended_mag_filters_test_allScenes[mag] += blended_mag_filter

    # Convert mag filters from lists to numpy arrays
    unblended_mag_filters_allScenes = {mag : np.array(unblended_mag_filters_allScenes[mag])
                                        for mag in mags}
    blended_mag_filters_allScenes = {mag : np.array(blended_mag_filters_allScenes[mag])
                                        for mag in mags}
    unblended_mag_filters_test_allScenes = {mag : np.array(unblended_mag_filters_test_allScenes[mag])
                                        for mag in mags}
    blended_mag_filters_test_allScenes = {mag : np.array(blended_mag_filters_test_allScenes[mag])
                                        for mag in mags}

    # Print some summary values
    print()
    print('TRAINING SET')
    print('n_unblended_allScenes:', n_unblended_allScenes,
        'n_blended_allScenes:', n_blended_allScenes)
    for mag in mags:
        print('mag:', mag)
        print('  n_unblended:', np.sum(unblended_mag_filters_allScenes[mag]),
            'n_blended:', np.sum(blended_mag_filters_allScenes[mag]))
    print()
    print('TEST SET')
    print('n_unblended_test_allScenes:', n_unblended_test_allScenes,
        'n_blended_test_allScenes:', n_blended_test_allScenes)
    for mag in mags:
        print('mag:', mag)
        print('  n_unblended_test:', np.sum(unblended_mag_filters_test_allScenes[mag]),
            'n_blended_test:', np.sum(blended_mag_filters_test_allScenes[mag]))
    print()

    # Make training cutouts
    cutouts_unblended_allScenes = np.zeros((n_unblended_allScenes,
                                                2*hw+1, 2*hw+1))
    cutouts_blended_allScenes = np.zeros((n_blended_allScenes,
                                                2*hw+1, 2*hw+1))
    # Indices used for storing cutouts in cutouts_*blended_allScenes
    unblended_lo, unblended_hi = 0, 0
    blended_lo, blended_hi = 0, 0
    print('Making cutouts on training scenes')
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

    # Make test cutouts
    cutouts_unblended_test_allScenes = np.zeros((n_unblended_test_allScenes,
                                                2*hw+1, 2*hw+1))
    cutouts_blended_test_allScenes = np.zeros((n_blended_test_allScenes,
                                                2*hw+1, 2*hw+1))
    # Indices used for storing cutouts in cutouts_*blended_allScenes
    unblended_lo, unblended_hi = 0, 0
    blended_lo, blended_hi = 0, 0
    print('Making cutouts on test scenes')
    for scene_idx in progressbar(range(len(blended_i_test_allScenes))):
        # Update *blended_hi
        unblended_hi += len(unblended_i_test_allScenes[scene_idx])
        blended_hi += len(blended_i_test_allScenes[scene_idx])
        # Load data for this scene
        image_filename = TEST_IMAGE_FILENAMES[scene_idx]
        scene = np.loadtxt(image_filename, delimiter=',')
        footprints = footprints_test_allScenes[scene_idx]
        segmap = segmap_test_allScenes[scene_idx]
        unblended_i = unblended_i_test_allScenes[scene_idx]
        blended_i = blended_i_test_allScenes[scene_idx]
        # Make and store cutouts
        cutouts_unblended_test_allScenes[unblended_lo:unblended_hi] = make_cutouts(
            unblended_i, scene, footprints, segmap, hw, cutout_center)
        cutouts_blended_test_allScenes[blended_lo:blended_hi] = make_cutouts(
            blended_i, scene, footprints, segmap, hw, cutout_center)
        # Update *blended_lo
        unblended_lo, blended_lo = unblended_hi, blended_hi

    tn_mags_allTrials, fp_mags_allTrials = np.zeros((n_trials, len(mags))), np.zeros((n_trials, len(mags)))
    fn_mags_allTrials, tp_mags_allTrials = np.zeros((n_trials, len(mags))), np.zeros((n_trials, len(mags)))
    posacc_allTrials, posaccerr_allTrials = np.zeros(n_trials), np.zeros(n_trials)
    negacc_allTrials, negaccerr_allTrials = np.zeros(n_trials), np.zeros(n_trials)
    balacc_allTrials, balaccerr_allTrials = np.zeros(n_trials), np.zeros(n_trials)
    posacc_mags_allTrials, posaccerr_mags_allTrials = np.zeros((n_trials, len(mags))), np.zeros((n_trials, len(mags)))
    negacc_mags_allTrials, negaccerr_mags_allTrials = np.zeros((n_trials, len(mags))), np.zeros((n_trials, len(mags)))
    balacc_mags_allTrials, balaccerr_mags_allTrials = np.zeros((n_trials, len(mags))), np.zeros((n_trials, len(mags)))

    n_nearp_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    n_nearp_err_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    p_median_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    p_16percentile_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    p_84percentile_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    calibration_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    calibrationerr_allTrials = np.zeros((n_trials, len(p_lo_bins)))
    mean_cross_entropy_allTrials = np.zeros(n_trials)
    stderr_mean_cross_entropy_allTrials = np.zeros(n_trials)

    for trial in progressbar(range(n_trials), redirect_stdout=True):
        print()

        # Construct train and test sets
        embed_method = None if embed_dim is None else 'pca'
        train, test, test_mag_filter = binaryclass_muygpys_datasets(
            cutouts_blended_allScenes, cutouts_unblended_allScenes,
            cutouts_blended_test_allScenes, cutouts_unblended_test_allScenes,
            blended_mag_filters_test_allScenes, unblended_mag_filters_test_allScenes,
            regress=regress)
        if augment:
            train = include_flips(train)

        print('Training set composition')
        print('n_all =', len(train['lookup']))
        print('n_blended =', sum(train['lookup'] == 1))
        print('n_unblended =', sum(train['lookup'] == 0))
        print('Test set composition')
        print('n_all =', len(test['lookup']))
        print('n_blended =', sum(test['lookup'] == 1))
        print('n_unblended =', sum(test['lookup'] == 0))

        predicted_labels = np.zeros(test['lookup'].shape, dtype=int)
        # Logistic regression
        if model_name == 'Logistic regression':
            # First need to reduce dimensionality to make tractable
            pca = PCA(n_components=embed_dim)
            train['input'] = pca.fit_transform(train['input'])
            test['input'] = pca.transform(test['input'])
            lr = LogisticRegressionCV(cv=5, penalty='l2', scoring='accuracy',
                                        max_iter=10000, n_jobs=-1)
            lr = lr.fit(train['input'], train['lookup'])
            predicted_labels = lr.predict(test['input'])
            y, p = test['lookup'], lr.predict_proba(test['input'])[:,1]
        # GP classification
        if model_name == 'Gaussian process':
            get_posterior = do_regress if regress else do_classify
            kwargs = {'variance_mode': 'diagonal'} if regress else {'opt_batch_size': 500}
            posterior = get_posterior(train, test, nn_count=50, 
                # hyper_dict={'eps': 1e-5, 'length_scale': 5e1, 'nu': 1e0},
                # hyper_dict={'eps': 3e-6, 'length_scale': 2e1, 'nu': 1e1},
                hyper_dict={'eps': eps, 'length_scale': length_scale, 'nu': nu},
                # optim_bounds={'eps': (1e-8, 3e-3), 'length_scale': (5e-1, 1e2), 'nu': (1e-2, 1e1)},
                # optim_bounds={'eps': (1e-8, 1e-2), 'length_scale': (5e-1, 20)},
                kern='matern',
                embed_method=embed_method, embed_dim=embed_dim,
                loss_method='log', nn_kwargs={'nn_method': 'exact',
                'p': 2}, verbose=False,
                **kwargs)
            predicted_labels = (posterior[0] > 0).reshape(-1) if regress else np.argmax(posterior, axis=1)
            if regress:
                mean = posterior[0].reshape(-1)
                std = np.sqrt(posterior[1] * sigma_sq)
                y, p = test['lookup'], norm.sf(0, mean, std)

        c = np.zeros((2,2), dtype=int)
        c_ = confusion_matrix(test['lookup'], predicted_labels)
        if c_.shape == (1,1):
            c_ = c_ * np.eye(2, dtype=int) # Now c_.shape == (2,2)
            # Zero out whichever c_ entry doesn't match this test example
            clear_idx = 1 - test['lookup'][0]
            c_[clear_idx, clear_idx] = 0
        if c_.shape == (2,2):
            c += c_

        c_mags = {mag : np.zeros((2,2), dtype=int)
                                        for mag in mags}
        for mag in mags:
            c_ = confusion_matrix(test['lookup'][test_mag_filter[mag]],
                    predicted_labels[test_mag_filter[mag]])
            if c_.shape == (1,1):
                c_ = c_ * np.eye(2, dtype=int) # Now c_.shape == (2,2)
                # Zero out whichever c entry doesn't match this test example
                clear_idx = 1 - test['lookup'][test_mag_filter[mag]][0]
                c_[clear_idx, clear_idx] = 0
            if c_.shape == (2,2):
                c_mags[mag] += c_

        # Report results
        print()
        # Overall accuracy
        tn, fp, fn, tp = c.ravel()
        n_blended, n_unblended = tp + fn, tn + fp
        posacc, negacc = tp/n_blended, tn/n_unblended
        posaccerr = np.sqrt(posacc*(1-posacc) / n_blended)
        negaccerr = np.sqrt(negacc*(1-negacc) / n_unblended)
        balacc = (posacc + negacc) / 2
        balaccerr = np.sqrt((posacc*(1-posacc) / n_blended)
                            + (negacc*(1-negacc) / n_unblended)) / 2
        print('tn, fp =', tn, fp)
        print('fn, tp =', fn, tp)
        print(f'Balacc: {balacc:.5f}, Posacc: {posacc:.5f}, Negacc: {negacc:.5f}')
        print(f'Balaccerr: {balaccerr:.5f}, Posaccerr: {posaccerr:.5f}, Negaccerr: {negaccerr:.5f}')
        balacc_allTrials[trial] = balacc
        posacc_allTrials[trial] = posacc
        negacc_allTrials[trial] = negacc
        balaccerr_allTrials[trial] = balaccerr
        posaccerr_allTrials[trial] = posaccerr
        negaccerr_allTrials[trial] = negaccerr
        # Accuracy vs. mag
        balacc_mags, balaccerr_mags = np.zeros(len(mags)), np.zeros(len(mags))
        posacc_mags, posaccerr_mags = np.zeros(len(mags)), np.zeros(len(mags))
        negacc_mags, negaccerr_mags = np.zeros(len(mags)), np.zeros(len(mags))
        tn_mags, fp_mags = np.zeros(len(mags)), np.zeros(len(mags))
        fn_mags, tp_mags = np.zeros(len(mags)), np.zeros(len(mags))
        for idx, mag in enumerate(mags):
            tn, fp, fn, tp = c_mags[mag].ravel()
            tn_mags[idx] = tn
            fp_mags[idx] = fp
            fn_mags[idx] = fn
            tp_mags[idx] = tp
        n_blended_mags, n_unblended_mags = tp_mags + fn_mags, tn_mags + fp_mags
        posacc_mags = tp_mags / n_blended_mags
        posaccerr_mags = np.sqrt(posacc_mags*(1-posacc_mags) / n_blended_mags)
        negacc_mags = tn_mags / n_unblended_mags
        negaccerr_mags = np.sqrt(negacc_mags*(1-negacc_mags) / n_unblended_mags)
        balacc_mags = (posacc_mags + negacc_mags) / 2
        balaccerr_mags = np.sqrt((np.sqrt(posacc_mags*(1-posacc_mags) / n_blended_mags))
                                 + (np.sqrt(negacc_mags*(1-negacc_mags) / n_unblended_mags))) / 2
        print('mags =', mags)
        print('Blend brightness rank for mags:', brightness_rank)
        print('tn =', tn_mags)
        print('fp =', fp_mags)
        print('fn =', fn_mags)
        print('tp =', tp_mags)
        print('posacc =', posacc_mags)
        print('posaccerr =', posaccerr_mags)
        print('negacc =', negacc_mags)
        print('negaccerr =', negaccerr_mags)
        balacc_mags_allTrials[trial], balaccerr_mags_allTrials[trial] = balacc_mags, balaccerr_mags
        posacc_mags_allTrials[trial], posaccerr_mags_allTrials[trial] = posacc_mags, posaccerr_mags
        negacc_mags_allTrials[trial], negaccerr_mags_allTrials[trial] = negacc_mags, negaccerr_mags
        tn_mags_allTrials[trial], fp_mags_allTrials[trial] = tn_mags, fp_mags
        fn_mags_allTrials[trial], tp_mags_allTrials[trial] = fn_mags, tp_mags
        # Calibration
        if model_name == 'Logistic regression' or (model_name == 'Gaussian process' and regress):
            p_median = np.zeros(len(p_lo_bins))
            p_16percentile = np.zeros(len(p_lo_bins))
            p_84percentile = np.zeros(len(p_lo_bins))
            n_nearp = np.zeros(len(p_lo_bins))
            n_blended_nearp = np.zeros(len(p_lo_bins))
            for idx, p_lo in enumerate(p_lo_bins):
                p_indices = (p >= p_lo) & (p < p_lo+p_stepsize)
                if idx == len(p_lo_bins) - 1:
                    p_indices |= p == 1
                n_nearp[idx] = p_indices.sum()
                n_blended_nearp[idx] = y[p_indices].sum()
                p_median[idx] = np.median(p[p_indices]) if n_nearp[idx] > 0 else np.nan
                p_16percentile[idx] = np.percentile(p[p_indices], 16) if n_nearp[idx] > 0 else np.nan
                p_84percentile[idx] = np.percentile(p[p_indices], 84) if n_nearp[idx] > 0 else np.nan
            n_nearp_allTrials[trial] = n_nearp
            n_nearp_err = np.sqrt(n_nearp * (n_nearp / len(p)) * (1 - n_nearp / len(p)))
            n_nearp_err_allTrials[trial] = n_nearp_err
            calibration = n_blended_nearp / n_nearp
            calibrationerr = np.sqrt(calibration*(1-calibration) / n_nearp)
            p_median_allTrials[trial] = p_median
            p_16percentile_allTrials[trial] = p_16percentile
            p_84percentile_allTrials[trial] = p_84percentile
            calibration_allTrials[trial] = calibration
            calibrationerr_allTrials[trial] = calibrationerr
            sum_cross_entropy = -(np.log(p[y == 1]).sum() + np.log(1 - p[y == 0]).sum())
            mean_cross_entropy = sum_cross_entropy / len(p)
            sum_sqdev_cross_entropy = np.sum((-np.log(p[y == 1]) - mean_cross_entropy)**2)
            sum_sqdev_cross_entropy += np.sum((-np.log(1 - p[y == 0]) - mean_cross_entropy)**2)
            std_dev_cross_entropy = np.sqrt(sum_sqdev_cross_entropy / (len(p) - 1))
            stderr_mean_cross_entropy = std_dev_cross_entropy / np.sqrt(len(p))
            mean_cross_entropy_allTrials[trial] = mean_cross_entropy
            stderr_mean_cross_entropy_allTrials[trial] = stderr_mean_cross_entropy
            print('n_nearp =', n_nearp)
            print('n_nearp_err =', n_nearp_err)
            print('p_median =', p_median)
            print('calibration =', calibration)
            print('calibrationerr =', calibrationerr)
            print('mean_cross_entropy =', mean_cross_entropy)
            print('stderr_mean_cross_entropy =', stderr_mean_cross_entropy)

    print()
    print('SUMMARY')
    print(model_name)
    print(footprint_type, 'footprints')
    print('Negacc:', f'{np.mean(negacc_allTrials):.4f}',
            '+/-', f'{np.mean(negaccerr_allTrials):.4f}', '(stat)',
            '+/-', f'{np.std(negacc_allTrials, ddof=1):.4f}', '(model)')
    print('Posacc:', f'{np.mean(posacc_allTrials):.4f}',
            '+/-', f'{np.mean(posaccerr_allTrials):.4f}', '(stat)',
            '+/-', f'{np.std(posacc_allTrials, ddof=1):.4f}', '(model)')
    print('Balacc:', f'{np.mean(balacc_allTrials):.4f}',
            '+/-', f'{np.mean(balaccerr_allTrials):.4f}', '(stat)',
            '+/-', f'{np.std(balacc_allTrials, ddof=1):.4f}', '(model)')
    print(f'balacc_{footprint_type} =', list(np.mean(balacc_mags_allTrials, axis=0)))
    print(f'balaccerr_{footprint_type} =', list(np.mean(balaccerr_mags_allTrials, axis=0)))
    print(f'balaccerr_model_{footprint_type} = ', list(np.std(balacc_mags_allTrials, axis=0, ddof=1)))
    print(f'posacc_{footprint_type} =', list(np.mean(posacc_mags_allTrials, axis=0)))
    print(f'posaccerr_{footprint_type} =', list(np.mean(posaccerr_mags_allTrials, axis=0)))
    print(f'posaccerr_model_{footprint_type} =', list(np.std(posacc_mags_allTrials, axis=0, ddof=1)))
    print(f'negacc_{footprint_type} =', list(np.mean(negacc_mags_allTrials, axis=0)))
    print(f'negaccerr_{footprint_type} =', list(np.mean(negaccerr_mags_allTrials, axis=0)))
    print(f'negaccerr_model_{footprint_type} =', list(np.std(negacc_mags_allTrials, axis=0, ddof=1)))
    if regress:
        print()
        print('mean cross entropy loss:', f'{np.mean(mean_cross_entropy_allTrials):.4f}',
                '+/-', f'{np.mean(stderr_mean_cross_entropy_allTrials):.4f}', '(stat)',
                '+/-', f'{np.std(mean_cross_entropy_allTrials, ddof=1):.4f}', '(model)')
        print(f'n_nearp_{footprint_type} =', list(np.mean(n_nearp_allTrials, axis=0)))
        print(f'n_nearp_err_{footprint_type} =', list(np.mean(n_nearp_err_allTrials, axis=0)))
        print(f'n_nearp_err_model_{footprint_type} =', list(np.std(n_nearp_allTrials, axis=0, ddof=1)))
        p_median, p_16percentile, p_84percentile = [0]*len(p_lo_bins), [0]*len(p_lo_bins), [0]*len(p_lo_bins)
        calibration, calibrationerr = [0]*len(p_lo_bins), [0]*len(p_lo_bins)
        p_median_err_model, calibrationerr_model = [0]*len(p_lo_bins), [0]*len(p_lo_bins)
        for col_idx in range(len(p_lo_bins)):
            for average, allTrials in [(p_median, p_median_allTrials),
            (p_16percentile, p_16percentile_allTrials),
            (p_84percentile, p_84percentile_allTrials),
            (calibration, calibration_allTrials),
            (calibrationerr, calibrationerr_allTrials)]:
                col = allTrials[:,col_idx]
                colmean = np.mean(col[~np.isnan(col)])
                average[col_idx] = colmean
            for spread, allTrials in [(p_median_err_model, p_median_allTrials),
            (calibrationerr_model, calibration_allTrials)]:
                col = allTrials[:,col_idx]
                colstd = np.std(col[~np.isnan(col)], ddof=1)
                spread[col_idx] = colstd
        print(f'p_median_{footprint_type} =', p_median)
        print(f'p_16percentile_{footprint_type} =', p_16percentile)
        print(f'p_84percentile_{footprint_type} =', p_84percentile)
        print(f'p_median_err_model_{footprint_type} =', p_median_err_model)
        print(f'calibration_{footprint_type} =', calibration)
        print(f'calibrationerr_{footprint_type} =', calibrationerr)
        print(f'calibrationerr_model_{footprint_type} =', calibrationerr_model)


def main_peak_counting(include_train=True, include_test=False):
    footprint_type = 'variant'
    footprint_params = {'pipeline': (True, None, None),
    'custom': (False, 5.7, 2.3),
    'variant': (False, 5, 1)}
    use_pipeline_segmap, footprint_threshold, n_sigma_expand = footprint_params[footprint_type]

    mags = (21.5,22.5,23.5,24.5,25.5,26.5)
    brightness_rank = 0

    # Whether to only count galaxies with S/N > 20 (as opposed to 5)
    select_gold = False

    psf_fwhm = 0.7 # arcsec
    pixel_scale = 0.2 # arcsec/pixel
    n_exposures = 100
    exposure_time = 30
    d_eff = 642.3
    area = (np.pi/4) * d_eff**2
    # Only use a single filter band for now
    band = 'i'

    PATH = os.path.join('data','simulated_scenes')
    IMAGE_FILENAMES = []
    TRUE_POS_FILENAMES = []
    TRUE_NPHO_FILENAMES = []
    PIPELINE_SEGMAP_FILENAMES = []
    if include_train:
        print('Including training data')
        train_images = range(10)
        IMAGE_FILENAMES += [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in train_images]
        TRUE_POS_FILENAMES += [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in train_images]
        TRUE_NPHO_FILENAMES += [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in train_images]
        PIPELINE_SEGMAP_FILENAMES += [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in train_images]
    if include_test:
        print('Including test data')
        test_images = (11,12,13,15,16,19,21,23,24,25)
        IMAGE_FILENAMES += [os.path.join(PATH,f'{band}-band_image-{i}.csv') for i in test_images]
        TRUE_POS_FILENAMES += [os.path.join(PATH,f'true_pos_image-{i}.csv') for i in test_images]
        TRUE_NPHO_FILENAMES += [os.path.join(PATH,f'true_npho_allbands_image-{i}.csv') for i in test_images]
        PIPELINE_SEGMAP_FILENAMES += [os.path.join(PATH,f'pipeline_segmap_{i}.fits') for i in test_images]
    print()

    if use_pipeline_segmap:
        print('Using pipeline segmaps')
    else:
        print('footprint_threshold:', footprint_threshold)
    print()
    # Make footprints at this threshold

    total_ngals_allScenes = 0
    ngals_in_all_footprints_allScenes = 0
    total_n_footprints_allScenes = 0
    n_unblended_onepeak_allScenes = 0
    n_blended_multipeak_allScenes = 0
    n_false_positives_allScenes = 0
    n_unblended_allScenes = 0
    n_blended_allScenes = 0
    n_unblended_onepeak_mag_allScenes = {mag : 0 for mag in mags}
    n_blended_multipeak_mag_allScenes = {mag : 0 for mag in mags}
    n_unblended_mag_allScenes = {mag : 0 for mag in mags}
    n_blended_mag_allScenes = {mag : 0 for mag in mags}
    print('Making footprints on scenes')
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
        for mag in mags:
            mag_coords = find_mag_coords(mag, scene, true_pos, true_npho, band, n_exposures,
                exposure_time, area)

        # Count the peaks in each footprint
        scene = smooth(scene, psf_fwhm, pixel_scale)
        peaks_by_footprint = find_peaks(footprints, scene, segmap)
        # Label the footprints by how many galaxies they contain.
        # At the same time, evaluate the effectiveness of peak counting for
        # blend identification.
        (false_positives, unblended_i, blended_i, ngals_in_all_footprints,
            n_unblended_onepeak, n_blended_multipeak
            ) = label_footprints_and_compare_peaks(footprints, segmap,
            bright_coords, peaks_by_footprint)
        # Store data for use below
        total_ngals_allScenes += len(bright_coords)
        ngals_in_all_footprints_allScenes += ngals_in_all_footprints
        total_n_footprints_allScenes += len(footprints)
        n_unblended_onepeak_allScenes += n_unblended_onepeak
        n_blended_multipeak_allScenes += n_blended_multipeak
        n_false_positives_allScenes += len(false_positives)
        n_unblended_allScenes += len(unblended_i)
        n_blended_allScenes += len(blended_i)

        for mag in mags:
            mag_coords = find_mag_coords(mag, scene, true_pos, true_npho,
                band, n_exposures, exposure_time, area)
            (unblended_i_mag, blended_i_mag, n_unblended_onepeak_mag,
                n_blended_multipeak_mag) = label_footprints_and_compare_peaks_mag(
                footprints, segmap, bright_coords, mag_coords,
                brightness_rank, peaks_by_footprint)
            n_unblended_mag_allScenes[mag] += len(unblended_i_mag)
            n_blended_mag_allScenes[mag] += len(blended_i_mag)
            n_unblended_onepeak_mag_allScenes[mag] += n_unblended_onepeak_mag
            n_blended_multipeak_mag_allScenes[mag] += n_blended_multipeak_mag

    print('SUMMARY')
    print(footprint_type, 'footprints')
    print()
    print('total_ngals_allScenes:', total_ngals_allScenes)
    print('ngals_in_all_footprints_allScenes:', ngals_in_all_footprints_allScenes)
    print('total_n_footprints_allScenes:', total_n_footprints_allScenes)
    print('n_false_positives_allScenes:', n_false_positives_allScenes)
    print('n_unblended_allScenes:', n_unblended_allScenes)
    print('n_unblended_onepeak_allScenes:', n_unblended_onepeak_allScenes)
    print('n_blended_allScenes:', n_blended_allScenes)
    print('n_blended_multipeak_allScenes:', n_blended_multipeak_allScenes)
    print()
    galfrac = ngals_in_all_footprints_allScenes / total_ngals_allScenes
    galfracerr = np.sqrt(galfrac * (1-galfrac) / total_ngals_allScenes)
    blendfrac = n_blended_allScenes / total_n_footprints_allScenes
    blendfracerr = np.sqrt(blendfrac * (1-blendfrac) / total_n_footprints_allScenes)
    unblendfrac = n_unblended_allScenes / total_n_footprints_allScenes
    unblendfracerr = np.sqrt(unblendfrac * (1-unblendfrac) / total_n_footprints_allScenes)
    emptyfrac = n_false_positives_allScenes / total_n_footprints_allScenes
    emptyfracerr = np.sqrt(emptyfrac * (1-emptyfrac) / total_n_footprints_allScenes)
    print('frac gals in footprints:', f'{galfrac:.4f}', '+/-', f'{galfracerr:.4f}')
    print('frac unblended footprints:', f'{unblendfrac:.4f}', '+/-', f'{unblendfracerr:.4f}')
    print('frac blended footprints:', f'{blendfrac:.4f}', '+/-', f'{blendfracerr:.4f}')
    print('frac empty footprints:', f'{emptyfrac:.4f}', '+/-', f'{emptyfracerr:.4f}')
    print()
    posacc = n_blended_multipeak_allScenes / n_blended_allScenes
    posaccerr = np.sqrt(posacc*(1-posacc) / n_blended_allScenes)
    negacc = n_unblended_onepeak_allScenes / n_unblended_allScenes
    negaccerr = np.sqrt(negacc*(1-negacc) / n_unblended_allScenes)
    print('Unblended accuracy:', f'{negacc:.4f}', '+/-', f'{negaccerr:.4f}')
    print('Blended accuracy:', f'{posacc:.4f}', '+/-', f'{posaccerr:.4}')
    balacc = (posacc + negacc) / 2
    balaccerr = np.sqrt(posaccerr**2 + negaccerr**2) / 2
    print('Balanced accuracy:', f'{balacc:.4f}', '+/-', f'{balaccerr:.4f}')
    print()
    print('mags:', mags)
    blended_acc_mag = []
    blended_accerr_mag = []
    unblended_acc_mag = []
    unblended_accerr_mag = []
    for mag in mags:
        tp, fn = n_blended_multipeak_mag_allScenes[mag], (n_blended_mag_allScenes[mag] - n_blended_multipeak_mag_allScenes[mag])
        tn, fp = n_unblended_onepeak_mag_allScenes[mag], (n_unblended_mag_allScenes[mag] - n_unblended_onepeak_mag_allScenes[mag])
        posacc, negacc = tp/(tp+fn), tn/(tn+fp)
        posaccerr = np.sqrt(posacc*(1-posacc) / (tp+fn))
        negaccerr = np.sqrt(negacc*(1-negacc) / (tn+fp))
        blended_acc_mag.append(posacc)
        blended_accerr_mag.append(posaccerr)
        unblended_acc_mag.append(negacc)
        unblended_accerr_mag.append(negaccerr)
    print()
    print(f'posacc_{footprint_type} =', blended_acc_mag)
    print(f'posaccerr_{footprint_type} =', blended_accerr_mag)
    print(f'negacc_{footprint_type} =', unblended_acc_mag)
    print(f'negaccerr_{footprint_type} =', unblended_accerr_mag)


if __name__ == '__main__':
    # main_validate()
    main_test()
    # main_peak_counting()
    # main_peak_counting(include_train=True, include_test=True)