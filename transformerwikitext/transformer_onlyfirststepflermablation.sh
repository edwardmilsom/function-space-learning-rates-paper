# seed 8 because we used 0-7 for the mass measurement runs
###### POSTNORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --only_flerm_first_step_ablation
        done
    done
done

# Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --only_flerm_first_step_ablation
       done
   done
done

# ###### PRENORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststepprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststepprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --only_flerm_first_step_ablation
        done
    done
done

# Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststepprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststepprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --only_flerm_first_step_ablation
       done
   done
done

# ###### POSTNORMPRERES ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation
        done
    done
done

# # Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation
       done
   done
done

# Normalised Init Scale
for seed in 8; do
   for initscale in 0.0625 0.25 1.0 4.0 16.0 64.0 256.0; do
       for lr in 1.0000566999999997e-05 1.8479481841824148e-05 3.414718876862761e-05 6.309865778602309e-05 0.0001165964390619342 0.0002154519617204074 0.0003981214879513986 0.0007356661684720361 0.001359395882445798 0.0025119507249446583 0.004641691597003633 0.008577119235556976 0.01584917326873923 0.02928679040174336 0.054117402686700296 0.10000048600000001; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult 1 --optimiser adam --init_scale $initscale --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix onlyflermfirststeppostnormpreresnormalisedinitscaleTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --only_flerm_first_step_ablation
       done
   done
done