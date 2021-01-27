import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft
from matplotlib.colors import LogNorm
from itertools import combinations

class Telescope1D:
    def __init__(self, Nfreq=256, Ndishes=32, DDish=6, Npix=2**12, Npad=2**8,
                 minfreq=400, maxfreq=800, redundant=False):
        '''
        DDish is the diameter of the dishes in meters.
        The minfreq, maxfreq should be in MHz.
        '''
        self.Npix = Npix
        self.Npad = Npad
        self.Nfft = Npix*Npad
        self.Nfreq = Nfreq
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.Ndishes = Ndishes
        self.DDish = DDish
        self.redundant = redundant

    @property
    def freqs(self):
        return np.linspace(self.minfreq,self.maxfreq,self.Nfreq)

    @property
    def lams(self):
        return self.freq2lam(self.freqs)

    def freq2lam(self, freq_MHz):
        '''
        Turns frequency (in MHz) into wavelength (in m).
        '''
        return 3e8/(freq_MHz*1e6)

    def image2uv(self, image):
        '''
        Convert from pixel space to uv space by FFT.
        '''
        assert(len(image)==self.Npix)
        bigimage = np.zeros(self.Npix*self.Npad)
        bigimage[:self.Npix//2] = image[-self.Npix//2:]
        bigimage[-self.Npix//2:] = image[:self.Npix//2]
        return rfft(bigimage)

    def uv2image(self, uv):
        '''
        Convert from uv space to pixel space by inverse FFT.
        '''
        assert(len(uv) == self.Nfft//2+1)
        bigimage = irfft(uv)
        image = np.hstack((bigimage[-self.Npix//2:],bigimage[:self.Npix//2]))
        return image

    @property
    def alpha(self):
        '''
        Returns sin(theta), where theta describes angular coordinates from
        -pi/2 to +pi/2 with theta = 0 pointing at the zenith.
        '''
        return (np.arange(-self.Npix//2,self.Npix//2)+0.5)*(2/self.Npix)

    def empty_uv(self):
        return np.zeros(self.Nfft//2+1,np.complex)

    def empty_image(self):
        return np.zeros(self.Npix,np.float)

    def DoL2ndx(self, DoL):
        '''
        Returns the index in the uv plane of the D/lambda.
        DoL can be a vector or array.
        Not nearest integer; this returns a float.
        '''
        return DoL*self.Npad*2+0.5

    def primary_beam_1(self, D, freq_MHz):
        '''
        Take baseline length D and frequency in MHz, return beam.
        '''
        lam = self.freq2lam(freq_MHz)
        t = self.empty_uv()
        t[:int(self.DoL2ndx(D/lam/2))] = 1.0
        return self.uv2image(t)

    @property
    def dish_locations(self):
        '''
        If self.redundant = True, the distance between consecutive dishes is
        the diameter of the dishes, self.DDish.
        Otherwise, distances between two consecutive dishes are D, 1.25*D,
        1.5*D, 1.75*D, 2*D, D, 1.25*D, ... where D is the dish diameter.
        '''
        dish_locations = np.arange(0,self.Ndishes,dtype=float)
        for ii in np.arange(1,self.Ndishes):
            if self.redundant:
                distance = self.DDish # This makes redundant array
            else:
                # Get D, 1.25*D, 1.5*D, 1.75*D, 2*D, D, ... for ii = 1, 2, ...
                distance = self.DDish*((ii-1)%5*0.25+1)
            dish_locations[ii] = dish_locations[ii-1]+distance
        return dish_locations

    @property
    def baseline_lengths(self):
        '''
        Get every baseline combination, compute baseline lengths.
        Returns a vector of all baseline lengths for each combination.
        '''
        baseline_lengths = np.array([d2-d1 for d1, d2 in
                                     combinations(self.dish_locations,2)])
        return baseline_lengths

    @property
    def DoL(self):
        '''
        Get 2D array of (N freqs x N unique baselines) containg D/lambda for
        each combination.
        '''
        unique_baseline_lengths = np.unique(self.baseline_lengths)
        return np.outer(1/self.lams, unique_baseline_lengths)

    def uv2uvplane(self, uv, indices=None):
        '''
        Instead of e.g. uvplane = uv[indices.astype(int)] (nearest
        neighbor), use linear interpolation approach to assign uv values to
        uvplane.
        '''
        if indices is None:
            indices = self.DoL2ndx(self.DoL)
        uvplane = np.zeros_like(indices, np.complex)
        if indices.ndim == 1:
            # Getting uvplane for one frequency bin
            for ii, idx_val in enumerate(indices):
                # Looping through every element of indices
                val1 = uv[np.floor(idx_val).astype(int)]
                val2 = uv[np.ceil(idx_val).astype(int)]
                # If the idx_val is 1.2, we'd do 0.2*(uv[2]-uv[1]) + uv[1]
                val = (idx_val%1) * (val2-val1) + val1
                uvplane[ii] = val
        elif indices.ndim == 2:
            for ii, idx_vals in enumerate(indices):
                for jj, idx_val in enumerate(idx_vals):
                    # Looping through every element of indices
                    val1 = uv[np.floor(idx_val).astype(int)]
                    val2 = uv[np.ceil(idx_val).astype(int)]
                    # If the idx_val is 1.2, we'd do 0.2*(uv[2]-uv[1]) + uv[1]
                    val = (idx_val%1) * (val2-val1) + val1
                    uvplane[ii,jj] = val
        return uvplane

    def get_obs_rmap(self, uvplane, time_error_sigma=10e-12):
        '''
        Get the rmap observed by the telescope array.
        For no time/phase error, do time_error_sigma = 0 seconds.
        Returns (N frequencies x N unique baselines) array.
        '''
        baseline_lengths = self.baseline_lengths
        dish_locations = self.dish_locations
        unique_baseline_lengths = np.unique(baseline_lengths)
        indices = self.DoL2ndx(self.DoL).astype(int)
        rmap_obs = []
        # Add time errors; each antenna's error sampled from Gaussian
        time_errors = np.random.normal(0,time_error_sigma,self.Ndishes)
        for i,f in enumerate(self.freqs):
            phase_errors = time_errors*f*1e6*2*np.pi
            # Loop through each unique baseline length
            # Get and average all the observed visibilities for each
            for j, baseline_len in enumerate(unique_baseline_lengths):
                redundant_baseline_idxs = np.where(baseline_lengths==baseline_len)[0]
                uvplane_obs = []
                for k in redundant_baseline_idxs:
                    dish1_loc, dish2_loc = list(combinations(dish_locations,2))[k]
                    dish1_idx = np.where(dish_locations==dish1_loc)[0][0]
                    dish2_idx = np.where(dish_locations==dish2_loc)[0][0]
                    uvplane_obs.append((np.exp(1j*(phase_errors[dish2_idx]-phase_errors[dish1_idx])))*uvplane[:,j])
                uvplane_obs = np.array(uvplane_obs, np.complex)
                uvplane[:,j] = np.mean(uvplane_obs, axis=0)
            uvi = self.empty_uv()
            uvi[indices[i,:]] = uvplane[i,:]
            rmap_obs.append(self.uv2image(uvi))
        rmap_obs = np.array(rmap_obs)
        return rmap_obs

    def plot_rmap(self, rmap):
        '''
        Plot the dirty map.
        '''
        plt.figure(figsize=(20,10))
        plt.imshow(rmap,aspect='auto',origin='lower',
                   extent=(self.alpha[0],self.alpha[-1],self.freqs[0],self.freqs[-1]))
        plt.ylabel('frequency [MHz]')
        plt.xlabel(r'sin($\theta$)')
        plt.colorbar()
        plt.show()

    def plot_wedge(self):
        '''
        Simulate various skies, and plot the wedge.
        '''
        #TODO: how does the p2fac change for a non redundant array? is the beam dependent on the baseline lengths or just dish diameter?
        #TODO cont'd: if dependent on all the unique baseline lengths, how to deal with that in the nested for loop?
        pass
