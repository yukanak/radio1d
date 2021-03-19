# 1D Radio Interferometry
The `Telescope1D` class in `telescope_1d.py` creates a 1 dimensional
telescope array with the specified properties. If `redundant` is `True`,
the distance between consecutive dishes is the diameter of the dishes, `DDish`
(i.e. a perfectly redundant array). If `False`, the distances are chosen
randomly from `DDish`, `DDish*1.25`, `DDish*1.5`, so that the array is less
redundant.

Examples of how to use this class are in `telescope_1d_play.ipynb` and
`telescope_1d_test_foreground_filtering`.
The general procedure is to initialize the telescope, create a sky
(either manually or by using the `get_gaussian_sky`, `get_point_source_sky`,
`get_poisson_sky`, or `get_uniform_sky` in the Telscope1D class).
Then, use `observe_image` to convolve and convert the image to uvplane (Fourier
space).
From here, you can get real space maps with error/foreground filtering
using `get_obs_rmap`, then get and plot the power spectra using `get_rmap_ps`
and `plot_rmap_ps_slice`.
Or, you can stay in Fourier space and get the new uvplane after adding
error/foreground filtering using `get_obs_uvplane`, then get and plot the power
spectra using `get_uvplane_ps` and `plot_uvplane_ps_slice`.
You can also plot the "wedge" using `plot_wedge`.
