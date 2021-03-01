import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from itertools import combinations
from astropy.cosmology import Planck15 as cosmo
import telescope_1d

Ndishes_array = [32, 64, 128]
Npix_array = [2**11, 2**12, 2**13]
redundant_array = [True, False]
sky_array = ['uniform', 'poisson', 'gaussian']
time_error_array = [1e-12, 10e-12, 100e-12]

for npix in Npix_array:
    for ndishes in Ndishes_array:
        print(f'Doing {ndishes} dishes, {npix} pixels')
        for redundant in redundant_array:
            t = telescope_1d.Telescope1D(Nfreq=256, Ndishes=ndishes, DDish=6, Npix_fft=npix, Npad=2**8,
                                         minfreq=400, maxfreq=800, redundant=redundant)
            for sky in sky_array:
                if sky == 'uniform':
                    image = t.get_uniform_sky(high=2)
                elif sky == 'poisson':
                    image = t.get_poisson_sky(lam=1, sigma_f=40)
                elif sky == 'gaussian':
                    image = t.get_gaussian_sky(mean=1, sigma_o=0.5, sigma_f=40)
                uvplane = t.beam_convolution(image)
                rmap_no_error = t.get_obs_rmap(uvplane, time_error_sigma=0)
                (ps_binned_no_error, k_modes_no_error, alpha_binned_no_error) = t.get_rmap_ps(rmap_no_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
                for error in time_error_array:
                    rmap_with_error = t.get_obs_rmap(uvplane, time_error_sigma=error)
                    (ps_binned_with_error, k_modes_with_error, alpha_binned_with_error) = t.get_rmap_ps(rmap_with_error, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman)
                    f = t.plot_rmap_ps_slice(ps_binned_no_error, ps_binned_with_error, k_modes_no_error, alpha_binned_no_error, alpha_idx_source=[], 
                                             alpha_idx_no_source=[t.Npix//2, t.Npix//2+10, t.Npix//2+25, t.Npix//2+50], Nfreqchunks=4)
                    f.savefig(f'../20210225_random_sky_slices/figs/npix_{npix}_ndish_{ndishes}_redundant_{redundant}_sky_{sky}_error_{error}.png')
                    plt.close(f)
