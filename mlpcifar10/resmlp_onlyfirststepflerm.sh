# # # Normalised Width
for seed in 8; do
    for width_mult in 1 2 4 8 16; do
        for lr in 1.0000054876000003e-06 1.847859728124187e-06 3.4145668370462185e-06 6.309605922572629e-06 1.1659202703615059e-05 2.154445291070711e-05 3.981090843182247e-05 7.356457073826611e-05 0.00013593626172015485 0.0002511897651954686 0.0004641609040922991 0.0008576995353299024 0.001584899732871182 0.0029286563181930266 0.005411716370570592 0.010000037865000005; do
            python mlp_cifar10.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --model resmlp --ptprefix onlyflermfirststepnormalisedwidthTransferWeightInitRootL --ptnormsprefix weightInitRootL --only_flerm_first_step_ablation
        done
    done
    for width_mult in 32; do
        for lr in 1.0000054876000003e-06 1.847859728124187e-06 3.4145668370462185e-06 6.309605922572629e-06 1.1659202703615059e-05 2.154445291070711e-05 3.981090843182247e-05 7.356457073826611e-05 0.00013593626172015485 0.0002511897651954686 0.0004641609040922991 0.0008576995353299024 0.001584899732871182 0.0029286563181930266 0.005411716370570592 0.010000037865000005; do
            python mlp_cifar10.py --lr $lr --width_mult $width_mult --depth_mult 1 --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --model resmlp --ptprefix onlyflermfirststepnormalisedwidthTransferWeightInitRootL --ptnormsprefix weightInitRootL --only_flerm_first_step_ablation
        done
    done
done

# # Normalised Depth
for seed in 8; do
    for depth_mult in 1 2 4 8 16 32; do
        for lr in 1.0000054876000003e-06 1.847859728124187e-06 3.4145668370462185e-06 6.309605922572629e-06 1.1659202703615059e-05 2.154445291070711e-05 3.981090843182247e-05 7.356457073826611e-05 0.00013593626172015485 0.0002511897651954686 0.0004641609040922991 0.0008576995353299024 0.001584899732871182 0.0029286563181930266 0.005411716370570592 0.010000037865000005; do
            python mlp_cifar10.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --model resmlp --ptprefix onlyflermfirststepnormaliseddepthTransferWeightInitRootL --ptnormsprefix weightInitRootL --only_flerm_first_step_ablation
        done
    done
    for depth_mult in 64; do
        for lr in 1.0000054876000003e-06 1.847859728124187e-06 3.4145668370462185e-06 6.309605922572629e-06 1.1659202703615059e-05 2.154445291070711e-05 3.981090843182247e-05 7.356457073826611e-05 0.00013593626172015485 0.0002511897651954686 0.0004641609040922991 0.0008576995353299024 0.001584899732871182 0.0029286563181930266 0.005411716370570592 0.010000037865000005; do
            python mlp_cifar10.py --lr $lr --width_mult 1 --depth_mult $depth_mult --optimiser adam --normalised --normaliser_update_frequency 100 --normaliser_beta 0.9 --normaliser_approx_type kronecker --seed $seed --model resmlp --ptprefix onlyflermfirststepnormaliseddepthTransferWeightInitRootL --ptnormsprefix weightInitRootL --only_flerm_first_step_ablation
        done
    done
done