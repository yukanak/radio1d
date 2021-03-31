#!/usr/bin/env python
import sys
import numpy as np
import telescope_1d
import os
ndishes = int(sys.argv[1])
npix = int(sys.argv[2])
redundant = bool(sys.argv[3]=="1")
redstr = 'red' if redundant else 'nred'



t = None


for seed in range(30):
    for sigt in 'sig point gauss unif'.split():
        sig = None
        for te in [0,0.1,1.,10.,100.]:
            outfname = f"out/{ndishes}_{npix}_{redstr}_{seed}_{sigt}_{te}.npy"
            if os.path.isfile (outfname):
                print (f"{outfname} exists.")
                continue
            if t is None:
                telescope_1d.Telescope1D(Ndishes=ndishes, Npix_fft=npix, redundant=redundant, seed=22)
            if sig is None:
                if sigt == 'sig':
                    sig = t.get_signal(seed)
                elif sigt == 'point':
                    sig =  t.get_point_source_sky(seed=seed)
                elif sigt =='gauss':
                    sig = t.get_gaussian_sky(seed=seed)
                elif sigt =='unif':
                    sig = t.get_uniform_sky(seed=seed)
                else:
                    print ("Shit!")
                    stop
                uvsig = t.observe_image(sig)

            print (f"Working {outfname}")
            uvplane, uvplane_f, uvplane_1 = t.get_obs_uvplane(uvsig, time_error_sigma=te*1e-12, filter_FG=True)

            np.save(outfname,(uvplane, uvplane_f, uvplane_1))




