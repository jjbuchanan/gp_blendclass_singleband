# Overview
This repository contains the code used for the data preprocessing, image simulation, and model training and analysis reported in the paper "Gaussian Process Classification for Galaxy Blend Identification in LSST" ([arXiv:2107.09246](https://arxiv.org/abs/2107.09246)).

# DC2 catalog preprocessing
## Basic usage
Run the notebook `dc2catalogmerge.ipynb`

Ensure that GCRCatalogs has been set up and that the catalogs `desc_dc2_run2.2i_dr6_truth` and `desc_cosmodc2` are available.  
The [LSSTDESC Data Portal](https://lsstdesc-portal.nersc.gov/) contains instructions on how to access the publicly available data from these catalogs and set up GCRCatalogs.  
To match the paper results, ensure that all the files corresponding to CosmoDC2 Healpix ID 9685 are available.

## Python dependencies
GCRCatalogs, jupyter, numpy, scipy, matplotlib (to make the plots in the notebook)  
Tested in Python 3.8.7 on Windows 10.

# Coadd image simulation
Note: The specific images used in the paper are contained in `data/simulated_scenes` in this repository.
## Basic usage
Inside `image_simulation`:  
`python coadd_simulation.py`

## Python dependencies
numpy, pandas  
galsim, which is simplest to install in a conda environment. See the galsim documentation [here](http://galsim-developers.github.io/GalSim/_build/html/index.html).  
Tested in Python 3.8.7 on Ubuntu Linux.

# Classifier models (except CNN)
## Basic usage
`python hyperparameter_scan.py`

Uncomment one and only one line in the final block, depending on whether you want to train+validate a probabilistic classifier (`main_validate`), test a probabilistic classifier (`main_test`), include only training data in the peak counting approach (`main_peak_counting`), or include both train+test data in peak counting (`main_peak_counting(include_train=True, include_test=True)`).

The main methods begin with a variety of options for examining different types of models, hyperparameters, and cross-validation strategies.

## Python dependencies
All of the following should be available via pip install:  
numpy, pandas, sep, astropy, scipy, sklearn, progressbar2, muygpys  
Tested in Python 3.8.7 on Windows 10.

See https://github.com/LLNL/MuyGPyS for more information on MuyGPyS, the Gaussian process modeling utility used in this analysis.

The code in this repository (specifically hyperparameter_scan.py) makes use of an older, pre-release version of MuyGPyS than that which is available on the MuyGPyS GitHub repository or via pip install. The underlying model implementation in that pre-release version is completely equivalent to the currently-available (14 September 2021) pip install package, but the API has changed. The corresponding changes needed in the present code are small and straightforward, but we have not implemented those changes in this repository; instead we present the actual code used for the studies reported in [arXiv:2107.09246](https://arxiv.org/abs/2107.09246).

# CNN model
## Basic usage
`python CNN_Classification.py`

This will produce outputs that can be interpreted using  
`plot_generation.ipynb`  
It makes use of the utility script  
`footprints_and_cutouts.py`

## Python dependencies
tensorflow, numpy, pandas, tensorflow_probability, matplotlib, sklearn, scipy  
Tested in Python 3.8.7 on Windows 10.
