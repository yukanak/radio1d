#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from itertools import combinations
from astropy.cosmology import Planck15 as cosmo
import telescope_1d
import os, sys

def plot_sky_slices_stat(npix, redundant, sky, seed, error, correlated, ndishes, path='slac', second_plot='terr'):
    if path == 'slac':
        path = os.path.join(os.environ['HOME'], 'public_html/figs/npix_{npix}_ndish_{ndishes}_redundant_{redundant}_sky_{sky}_seed_{seed}_error_{error}_correlated_{correlated}_filtered.png'.format(npix=npix, ndishes=ndishes, redundant=redundant, sky=sky, seed=seed, error=error, correlated=correlated))
        # Check if the image already exists
        if os.path.isfile(path):
            return
    # Initialize telescope
    t = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=npix, Npad=2**8,
                             minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
    if npix < 8192:
        assert(8192%npix == 0)
        fact = 8192//npix
        timag = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
    else:
        timag = t
    # Make skies
    if sky == 'uniform':
        image = timag.get_uniform_sky(high=2, seed=seed)
    elif sky == 'poisson':
        image = timag.get_poisson_sky(lam=0.01, seed=seed)
    elif sky == 'gaussian':
        image = timag.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=60, seed=seed)
    else:
        print("Bad sky type!")
        stop()
    if npix < 8192:
        image = np.append(image[:-1].reshape(npix,fact).mean(axis=1),0)
    # Observe image
    uvplane = t.observe_image(image)
    # Get observed uvplane
    uvplane_no_error = t.get_obs_uvplane(uvplane, time_error_sigma=0, filter_FG=True)
    if second_plot == 'terr':
        uvplane_with_error = t.get_obs_uvplane(uvplane, time_error_sigma=error, correlated=correlated, seed=seed, filter_FG=True)
    elif second_plot == 'fweight':
        freqs = t.freqs
        weight = freqs/freqs.mean()
        uvplane_with_error = np.copy(uvplane_no_error)/weight[:,None]
    # Power spectrum calculation and plotting
    (ps_binned_no_error, k_modes_no_error, baselines_binned_no_error) = t.get_uvplane_ps(uvplane_no_error, Nfreqchunks=4, m_baselines=1, m_freq=2, padding=1, window_fn=np.blackman)
    (ps_binned_with_error, k_modes_with_error, baselines_binned_with_error) = t.get_uvplane_ps(uvplane_with_error, Nfreqchunks=4, m_baselines=1, m_freq=2, padding=1, window_fn=np.blackman)
    uvplane_diff = uvplane_with_error - uvplane_no_error
    (difference_ps_binned, k_modes_diff, baselines_binned_diff) = t.get_uvplane_ps(uvplane_diff, Nfreqchunks=4, m_baselines=1, m_freq=2, padding=1, window_fn=np.blackman)
    fig = t.plot_uvplane_ps_slice(ps_binned_no_error, ps_binned_with_error, k_modes_no_error, baselines_binned_no_error, Nfreqchunks=4, difference_ps_binned=difference_ps_binned)
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    #return fig, rmap_no_error, rmap_with_error, t

if __name__ == '__main__': 
    # ./plot_random_sky_slices.py 4096 False 'uniform' 0 300e-12 True 32
    # Results in sys.argv[1] = 4096, etc.
    print(sys.version)
    if len(sys.argv) < 8:
        print ("Doing everything...")
        Npix_array = [2**11, 2**12, 2**13]
        redundant_array = [True, False]
        sky_array = ['uniform', 'poisson', 'gaussian']
        seed_array = [0,1,2]
        time_error_array = [1e-12, 10e-12, 100e-12, 300e-12, 1e-9]
        correlated_array = [True, False]
        ndishes_array = [32, 64, 128]
        for npix in Npix_array:
            for redundant in redundant_array:
                for sky in sky_array:
                    for seed in seed_array:
                        for error in time_error_array:
                            for correlated in correlated_array:
                                for ndishes in ndishes_array:
                                    plot_sky_slices_stat(npix, redundant, sky, seed, error, correlated, ndishes)
    else:
        npix = int(sys.argv[1])
        redundant = eval(sys.argv[2])
        sky = str(sys.argv[3])
        seed = int(sys.argv[4])
        error = eval(sys.argv[5])
        correlated = eval(sys.argv[6])
        ndishes = int(sys.argv[7])
        plot_sky_slices_stat(npix, redundant, sky, seed, error, correlated, ndishes)
