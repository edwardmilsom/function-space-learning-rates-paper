# seed 8 because we used 0-7 for the mass measurement runs
###### POSTNORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationpostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationpostnormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnorm --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
done

# ###### PRENORM ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002 18.478497974222936 34.145488738336084 63.09573444801947 116.59144011798328 215.44346900318868 398.10717055349795 735.6422544596429; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002 18.478497974222936 34.145488738336084 63.09573444801947 116.59144011798328 215.44346900318868 398.10717055349795 735.6422544596429; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationprenormnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype prenorm --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
done


# ###### POSTNORMPRERES ######
# Normalised depth Transfer
for seed in 8; do
    for depth_mult in 1 2 4 8 16; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002 18.478497974222936 34.145488738336084 63.09573444801947 116.59144011798328 215.44346900318868 398.10717055349795 735.6422544596429; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationpostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 32 64; do
        for lr in 0.0010000000000000002 0.0018478497974222922 0.003414548873833602 0.006309573444801936 0.011659144011798319 0.02154434690031885 0.03981071705534976 0.07356422544596417 0.1359356390878527 0.25118864315095835 0.46415888336127825 0.8576958985908951 1.584893192461116 2.92864456462524 5.41169526546464 10.000000000000002 18.478497974222936 34.145488738336084 63.09573444801947 116.59144011798328 215.44346900318868 398.10717055349795 735.6422544596429; do
            python claudemain_wrapper_load_masses.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --transformer_depth_norm --bptt 256 --ptprefix equalmassbutstillsplittingdepthproperlyablationpostnormpreresnormaliseddepthTransferforwardPassRootL --ptnormsprefix forwardPassRootL --normtype postnormpreres --equal_mass_but_still_splitting_depth_properly_ablation --only_flerm_first_step_ablation
        done
    done
done