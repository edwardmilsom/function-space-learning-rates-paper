# Normalised Mass Measurement Postnorm
for seed in 0; do
   for width_mult in 1; do
       for lr in 0.004641691597003633; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix plotmassespostnorm --ptnormsprefix plotmasses --normtype postnorm
       done
   done
done

# Normalised Mass Measurement Prenorm
for seed in 0; do
   for width_mult in 1; do
       for lr in 0.008577119235556976; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix plotmassesprenorm --ptnormsprefix plotmasses --normtype prenorm
       done
   done
done

# Normalised Mass Measurment PostnormPreRes
for seed in 0; do
   for width_mult in 1; do
       for lr in 0.008577119235556976; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix plotmassespostnormpreres --ptnormsprefix plotmasses --normtype postnormpreres
       done
   done
done