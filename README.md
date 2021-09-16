# Classifier models (except CNN)
## Basic usage
`python hyperparameter_scan.py`

Uncomment one and only one line in the final block, depending on whether you want to train+validate a probabilistic classifier (`main_validate`), test a probabilistic classifier (`main_test`), include only training data in the peak counting approach (`main_peak_counting`), or include both train+test data in peak counting (`main_peak_counting(include_train=True, include_test=True)`).

The main methods begin with a variety of options for examining different types of models, hyperparameters, and cross-validation strategies.

This code was developed and run in Windows 10. Every effort has been made to keep file paths OS-agnostic, but some tweaking may still be needed to run on other OS's.

## Python dependencies
All of the following should be available via pip install:  
numpy, pandas, sep, astropy, scipy, sklearn, progressbar2, muygpys

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

# Coadd image simulation
## Basic usage
Inside `image_simulation`:  
`python coadd_simulation.py`

## Python dependencies
galsim, which is simplest to install in a conda environment. See the galsim documentation [here](http://galsim-developers.github.io/GalSim/_build/html/index.html).

Because galsim is not supported on Windows, the image simulation code was developed and run in Ubuntu Linux.
