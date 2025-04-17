# Normalised Mass Measurment PostnormPreRes
for approxtype in kronecker iid full_cov; do
    for seed in 1; do
        for width_mult in 1; do
            for lr in 0.008577119235556976; do
                python compare_variance_of_methods.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type $approxtype --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix comparevariances"$approxtype"postnormpreres --ptnormsprefix comparevariances"$approxtype" --normtype postnormpreres
            done
        done
    done
done
