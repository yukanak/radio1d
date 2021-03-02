#!/usr/bin/env bash
for npix in 2048 4086 8192; do
    for redundant in 'True' 'False'; do
        for sky in 'uniform' 'poisson' 'gaussian'; do
            for error in '1e-12' '10e-12' '100e-12' '300e-12' '1e-9'; do
                #echo "bsub -C 0 -W 1:00 -R select[rhel60] -o logfile_${npix}_${redundant}_${sky}_${error}.txt plot_random_sky_slices.py $npix $redundant $sky $error"
                bsub -C 0 -W 1:00 -R select[rhel60] -o logfile_${npix}_${redundant}_${sky}_${error}.txt ./plot_random_sky_slices.py $npix $redundant $sky $error
            done
        done
    done
done
