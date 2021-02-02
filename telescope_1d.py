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

        # If redundant = True, the distance between consecutive dishes is DDish
        dish_locations = np.arange(0,Ndishes,dtype=float)
        for ii in np.arange(1,Ndishes):
            if redundant:
                distance = DDish # This makes redundant array
            else:
                # Distances between two consecutive dishes are
                # DDish, 1.25*DDish, 1.5*DDish, 1.75*DDish, 2*DDish, DDish, ...
                #distance = DDish*((ii-1)%5*0.25+1)
                distance = DDish*(1+np.random.randint(0,3)/4)
            dish_locations[ii] = dish_locations[ii-1]+distance
        self.dish_locations = dish_locations
        
        # Get every baseline combination, compute baseline lengths
        self.baseline_lengths = np.array([d2-d1 for d1, d2 in
                                          combinations(dish_locations,2)])
        self.unique_baseline_lengths = np.unique(self.baseline_lengths)

        # Get 2D array of (N freqs x N unique baselines) containg D/lambda
        self.DoL = np.outer(1/self.lams, self.unique_baseline_lengths)

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
        #assert(len(image)==self.Npix)
        bigimage = np.zeros(self.Npix*self.Npad)
        # Put last half of image in beginning of bigimage
        bigimage[:self.Npix//2+1] = image[-self.Npix//2-1:]
        # Put first half of image at end of bigimage
        bigimage[-self.Npix//2:] = image[:self.Npix//2]
        return rfft(bigimage)

    def uv2image(self, uv):
        '''
        Convert from uv space to pixel space by inverse FFT.
        '''
        assert(len(uv) == self.Nfft//2+1)
        bigimage = irfft(uv)
        # Concatenate the last chunk of bigimage to first chunk of bigimage
        image = np.hstack((bigimage[-self.Npix//2:],bigimage[:self.Npix//2+1]))
        return image

    @property
    def alpha(self):
        '''
        Returns sin(theta), where theta describes angular coordinates from
        -pi/2 to +pi/2 with theta = 0 pointing at the zenith.
        '''
        #return (np.arange(-self.Npix//2,self.Npix//2)+0.5)*(2/self.Npix)
        return np.linspace(-1,1,self.Npix+1)

    def empty_uv(self):
        return np.zeros(self.Nfft//2+1,np.complex)

    def empty_image(self):
        return np.zeros(self.Npix+1,np.float)

    def DoL2ndx(self, DoL):
        '''
        Returns the index in the uv plane of the D/lambda.
        DoL can be a vector or array.
        Not nearest integer; this returns a float.
        '''
        return DoL*self.Npad*2

    def primary_beam_1(self, freq_MHz):
        '''
        Take frequency in MHz, return beam.
        '''
        lam = self.freq2lam(freq_MHz)
        t = self.empty_uv()
        t[:int(self.DoL2ndx(self.DDish/lam/2))] = 1.0
        return self.uv2image(t)

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
        indices = (self.DoL2ndx(self.DoL)+0.5).astype(int)
        rmap_obs = []
        # Add time errors; each antenna's error sampled from Gaussian
        time_errors = np.random.normal(0,time_error_sigma,self.Ndishes)
        for i, f in enumerate(self.freqs):
            phase_errors = time_errors*f*1e6*2*np.pi
            # Loop through each unique baseline length
            # Get and average all the observed visibilities for each
            uvplane_obs = np.zeros_like(uvplane, np.complex)
            for j, baseline_len in enumerate(self.unique_baseline_lengths):
                redundant_baseline_idxs = np.where(self.baseline_lengths==baseline_len)[0]
                uvplane_j = []
                for k in redundant_baseline_idxs:
                    dish1_loc, dish2_loc = list(combinations(self.dish_locations,2))[k]
                    dish1_idx = np.where(self.dish_locations==dish1_loc)[0][0]
                    dish2_idx = np.where(self.dish_locations==dish2_loc)[0][0]
                    uvplane_j.append((np.exp(1j*(phase_errors[dish2_idx]-phase_errors[dish1_idx])))*uvplane[:,j])
                uvplane_j = np.array(uvplane_j, np.complex)
                uvplane_obs[:,j] = np.mean(uvplane_j, axis=0)
            uvi = self.empty_uv()
            uvi[indices[i,:]] = uvplane_obs[i,:]
            rmap_obs.append(self.uv2image(uvi))
        rmap_obs = np.array(rmap_obs)
        return rmap_obs

    def plot_rmap(self, rmap, vmax=None, vmin=None):
        '''
        Plot the dirty map.
        '''
        plt.figure(figsize=(20,10))
        plt.imshow(rmap,aspect='auto',origin='lower', vmax=vmax, vmin=vmin,
                   extent=(self.alpha[0],self.alpha[-1],self.freqs[0],self.freqs[-1]))
        plt.ylabel('frequency [MHz]')
        plt.xlabel(r'sin($\theta$)')
        plt.colorbar()
        plt.show()

    def get_p2fac(self):
        return np.array([np.abs(self.primary_beam_1(f)**2)/np.cos(self.alpha) for f in self.freqs])

    def plot_wedge(self, Nreal=100):
        '''
        Simulate various skies, and plot the wedge.
        '''
        p2fac = self.get_p2fac()
        Nuniquebaselines = self.unique_baseline_lengths.shape[0]
        uvplane = np.zeros((self.Nfreq,Nuniquebaselines),np.complex)
        ps = np.zeros((self.Nfreq+1,Nuniquebaselines)) # (2*Nfreq/2)+1 = Nfreq+1
        for c in range(Nreal):
            # Create a random sky, this sky will be the same at all frequencies
            sky = np.random.uniform(0,1,self.Npix+1)
            # Loop over frequencies
            for i,f in enumerate(self.freqs):
                # Multiply by the beam^2/cos(alpha)
                msky = sky * p2fac[i,:]
                # FT to the uvplane and sample at indices corresponding to D/lambda
                uv = self.image2uv(msky)
                uvplane[i,:] = self.uv2uvplane(uv,indices=self.DoL2ndx(self.DoL)[i,:])
            # After uvplane is done, calculate power spectrum in the frequency direction
            # This gives delay spectrum
            for j in range(Nuniquebaselines):
                # FFT along frequency axis, get the power in the frequency domain
                # Power in the frequency domain gives us structure in the redshift direction, for 21 cm
                ps[:,j] += np.abs(rfft(np.hstack((uvplane[:,j],np.zeros(self.Nfreq))))**2)
        plt.imshow(ps[:,:],origin='lower',aspect='auto',interpolation='nearest', norm=LogNorm(), cmap='jet')
        plt.xlabel(r'Baseline Length - $k_\perp$')
        plt.ylabel('Delay - FT Along Frequency Direction')
        plt.colorbar()
        plt.show()
