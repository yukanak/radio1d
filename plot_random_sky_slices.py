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

def plot_random_sky_slices(npix, redundant, sky, error):
    Ndishes_array = [32, 64, 128]

    for ndishes in Ndishes_array:
        # Check if the image already exists
        path = os.path.join(os.environ['HOME'], 'public_html/figs/npix_{npix}_ndish_{ndishes}_redundant_{redundant}_sky_{sky}_error_{error}.png'.format(npix=npix, ndishes=ndishes, redundant=redundant, sky=sky, error=error))
        if os.path.isfile(path):
            continue
        t = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=npix, Npad=2**8,
                                 minfreq=400, maxfreq=800, redundant=redundant)
        if sky == 'uniform':
            image = t.get_uniform_sky(high=2)
        elif sky == 'poisson':
            image = t.get_poisson_sky(lam=1, sigma_f=40)
        elif sky == 'gaussian':
            image = t.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=40)
        uvplane = t.beam_convolution(image)
        rmap_no_error = t.get_obs_rmap(uvplane, time_error_sigma=0)
        (ps_binned_no_error, k_modes_no_error, alpha_binned_no_error) = t.get_rmap_ps(rmap_no_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
        rmap_with_error = t.get_obs_rmap(uvplane, time_error_sigma=error)
        (ps_binned_with_error, k_modes_with_error, alpha_binned_with_error) = t.get_rmap_ps(rmap_with_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
        f = t.plot_rmap_ps_slice(ps_binned_no_error, ps_binned_with_error, k_modes_no_error, alpha_binned_no_error, alpha_idx_source=[], 
                                 alpha_idx_no_source=[t.Npix//2, t.Npix//2+t.Npix//400, t.Npix//2+t.Npix//150, t.Npix//2+t.Npix//80], Nfreqchunks=4)
        f.savefig(path)
        plt.close(f)

if __name__ == '__main__': 
    # ./plot_random_sky_slices.py 4086 False uniform 300e-12
    # Results in sys.argv[1] = 4086, etc.
    print(sys.version)
    if len(sys.argv) < 5:
        #print("Usage: {} npix redundant sky error".format(sys.argv[0]))
        #sys.exit(1)
        Npix_array = [2**11, 2**12, 2**13]
        redundant_array = [True, False]
        sky_array = ['uniform', 'poisson', 'gaussian']
        time_error_array = [1e-12, 10e-12, 100e-12, 300e-12, 1e-9]
        for npix in Npix_array:
            for redundant in redundant_array:
                for sky in sky_array:
                    for error in time_error_array:
                        plot_random_sky_slices(npix, redundant, sky, error)
        sys.exit(0)
    npix = int(sys.argv[1])
    redundant = eval(sys.argv[2])
    sky = str(sys.argv[3])
    error = eval(sys.argv[4])
    plot_random_sky_slices(npix, redundant, sky, error)
