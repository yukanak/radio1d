#!/usr/bin/env bash
for npix in 2048 4096 8192; do
    for redundant in 'True' 'False'; do
        for sky in 'uniform' 'poisson' 'gaussian'; do
            for seed in 0 1 2; do
                for error in '1e-12' '10e-12' '100e-12' '300e-12' '1e-9'; do
                    for correlated in 'True' 'False'; do
                        for ndishes in 8 16 24 32; do
                            #echo "bsub -C 0 -W 1:00 -R select[rhel60] -o logfile_${npix}_${redundant}_${sky}_${error}.txt plot_random_sky_slices.py $npix $redundant $sky $error"
                            bsub -q long -C 0 -W 7:00 -R "select[rhel60] rusage[mem=24G]" -o logfile_${npix}_${redundant}_${sky}_${seed}_${error}_${correlated}_${ndishes}.txt ./plot_random_sky_slices.py $npix $redundant $sky $seed $error $correlated $ndishes
                        done
                    done
                done
            done
        done
    done
done
