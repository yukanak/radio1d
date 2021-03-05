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

def plot_random_sky_slices(npix, redundant, sky, seed, error, correlated):
    #breakpoint()
    Ndishes_array = [32, 64, 128]
    for ndishes in Ndishes_array:
        # Check if the image already exists
        path = os.path.join(os.environ['HOME'], 'public_html/figs/npix_{npix}_ndish_{ndishes}_redundant_{redundant}_sky_{sky}_seed_{seed}_error_{error}_correlated_{correlated}.png'.format(npix=npix, ndishes=ndishes, redundant=redundant, sky=sky, seed=seed, error=error, correlated=correlated))
        #path = '../20210225_random_sky_slices/figs/npix_{npix}_ndish_{ndishes}_redundant_{redundant}_sky_{sky}_seed_{seed}_error_{error}_correlated_{correlated}.png'.format(npix=npix, ndishes=ndishes, redundant=redundant, sky=sky, seed=seed, error=error, correlated=correlated)
        if os.path.isfile(path):
            continue
        t = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=npix, Npad=2**8,
                                 minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
        if sky == 'uniform':
            if npix == 8192:
                image = t.get_uniform_sky(high=2, seed=seed)
            elif npix == 4096:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_uniform_sky(high=2, seed=seed)
                image = image1[:-1].reshape(4096,2).mean(axis=1)
                image = np.append(image, image1[-1])
            elif npix == 2048:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_uniform_sky(high=2, seed=seed)
                image = image1[:-1].reshape(2048,4).mean(axis=1)
                image = np.append(image, image1[-1])
        elif sky == 'poisson':
            if npix == 8192:
                image = t.get_poisson_sky(lam=0.01, seed=seed)
            elif npix == 4096:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_poisson_sky(lam=0.01, seed=seed)
                image = image1[:-1].reshape(4096,2).mean(axis=1)
                image = np.append(image, image1[-1])
            elif npix == 2048:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_poisson_sky(lam=0.01, seed=seed)
                image = image1[:-1].reshape(2048,4).mean(axis=1)
                image = np.append(image, image1[-1])
        elif sky == 'gaussian':
            if npix == 8192:
                image = t.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=60, seed=seed)
            elif npix == 4096:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=60, seed=seed)
                image = image1[:-1].reshape(4096,2).mean(axis=1)
                image = np.append(image, image1[-1])
            elif npix == 2048:
                t2 = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=8192, Npad=2**8,
                                          minfreq=400, maxfreq=800, redundant=redundant, seed=seed)
                image1 = t2.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=60, seed=seed)
                image = image1[:-1].reshape(2048,4).mean(axis=1)
                image = np.append(image, image1[-1])
        uvplane = t.beam_convolution(image)
        rmap_no_error = t.get_obs_rmap(uvplane, time_error_sigma=0)
        (ps_binned_no_error, k_modes_no_error, alpha_binned_no_error) = t.get_rmap_ps(rmap_no_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
        rmap_with_error = t.get_obs_rmap(uvplane, time_error_sigma=error, correlated=correlated, seed=seed)
        (ps_binned_with_error, k_modes_with_error, alpha_binned_with_error) = t.get_rmap_ps(rmap_with_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
        rmap_diff = rmap_with_error - rmap_no_error
        (difference_ps_binned, k_modes_diff, alpha_binned_diff) = t.get_rmap_ps(rmap_diff, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
        f = t.plot_rmap_ps_slice(ps_binned_no_error, ps_binned_with_error, k_modes_no_error, alpha_binned_no_error, alpha_idx_source=[], 
                                 alpha_idx_no_source=[t.Npix//2, t.Npix//2+t.Npix//256, t.Npix//2+t.Npix//128, t.Npix//2+t.Npix//64], Nfreqchunks=4, difference_ps_binned=difference_ps_binned)
        f.savefig(path)
        plt.close(f)

if __name__ == '__main__': 
    # ./plot_random_sky_slices.py 4096 False 'uniform' 0 300e-12 True
    # Results in sys.argv[1] = 4096, etc.
    print(sys.version)
    if len(sys.argv) < 7:
        #print("Usage: {} npix redundant sky error".format(sys.argv[0]))
        #sys.exit(1)
        Npix_array = [2**11, 2**12, 2**13]
        redundant_array = [True, False]
        sky_array = ['uniform', 'poisson', 'gaussian']
        seed_array = [0,1,2]
        time_error_array = [1e-12, 10e-12, 100e-12, 300e-12, 1e-9]
        correlated_array = [True, False]
        for npix in Npix_array:
            for redundant in redundant_array:
                for sky in sky_array:
                    for seed in seed_array:
                        for error in time_error_array:
                            for correlated in correlated_array:
                                plot_random_sky_slices(npix, redundant, sky, seed, error, correlated)
        sys.exit(0)
    npix = int(sys.argv[1])
    redundant = eval(sys.argv[2])
    sky = str(sys.argv[3])
    seed = int(sys.argv[4])
    error = eval(sys.argv[5])
    correlated = eval(sys.argv[6])
    plot_random_sky_slices(npix, redundant, sky, seed, error, correlated)
