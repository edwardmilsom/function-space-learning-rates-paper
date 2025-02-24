# seed 8 because we used 0-7 for the mass measurement runs
###### POSTNORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
done

# Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
done

# ###### PRENORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
done

# Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationprenormnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
done

# ###### POSTNORMPRERES ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_ablation --only_flerm_first_step_ablation
        done
    done
done

# # Normalised width Transfer
for seed in 8; do
   for width_mult in 1 2 4 8; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
   for width_mult in 16 32; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormpreresnormalisedwidthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
done

# Normalised Init Scale
for seed in 8; do
   for initscale in 0.0625 0.25 1.0 4.0 16.0 64.0 256.0; do
       for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
           python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult 1 --optimiser adam --init_scale $initscale --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassablationpostnormpreresnormalisedinitscaleTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_ablation --only_flerm_first_step_ablation
       done
   done
done