# seed 8 because we used 0-7 for the mass measurement runs

# ###### POSTNORMPRERES ######

# Normalised Mass Measurment PostnormPreRes
for seed in 0 1 2 3 4 5 6 7; do
   for width_mult in 1; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix schedulerrebuttalrecordmassespostnormpreresforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnormpreres --use_scheduler
       done
   done
done

# # Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststeppostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststeppostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation --use_scheduler
       done
   done
done

# # Unnormalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalpostnormpreresunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnormpreres --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalpostnormpreresunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnormpreres --use_scheduler
       done
   done
done

# ###### POSTNORM ######

# Normalised Mass Measurment postnorm
for seed in 0 1 2 3 4 5 6 7; do
   for width_mult in 1; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix schedulerrebuttalrecordmassespostnormforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnorm --use_scheduler
       done
   done
done

# # Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststeppostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnorm --only_flerm_first_step_ablation --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststeppostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnorm --only_flerm_first_step_ablation --use_scheduler
       done
   done
done

# # Unnormalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalpostnormunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnorm --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalpostnormunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype postnorm --use_scheduler
       done
   done
done

# ###### PRENORM ######

# Normalised Mass Measurment prenorm
for seed in 0 1 2 3 4 5 6 7; do
   for width_mult in 1; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --measure_masses_only --bptt 256 --ptprefix schedulerrebuttalrecordmassesprenormforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype prenorm --use_scheduler
       done
   done
done

# # Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststepprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype prenorm --only_flerm_first_step_ablation --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix schedulerrebuttalonlyflermfirststepprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype prenorm --only_flerm_first_step_ablation --use_scheduler
       done
   done
done

# # Unnormalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalprenormunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype prenorm --use_scheduler
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --transformer_depth_norm --seed $seed --bptt 256 --ptprefix schedulerrebuttalprenormunnormalisedwidthTransferforwardPassRootL --ptnormsprefix schedulerrebuttalforwardPassRootL --normtype prenorm --use_scheduler
       done
   done
done