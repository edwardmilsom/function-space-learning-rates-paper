# # Normalised
for seed in 0; do
    for depth_mult in 1; do
        for lr in 0.0004641609040922991; do
            python mlp_cifar10.py --ptprefix plotmasses --ptnormsprefix plotmasses --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --model resmlp --only_measure_masses
        done
    done
done
