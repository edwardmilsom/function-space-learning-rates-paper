
script_name="train_scripts/gpt2.py"
dataset="cold_french_law"

b_lrs=("0.00001389"  "0.00007197"  "0.00037276" "0.00193070"  "0.00439397" "0.00999996" "0.02275833" "0.05179475" "0.26826642")
for lr in $(printf "%s\n" "${b_lrs[@]}" | shuf); do
    echo "lr=$lr"
    # normalized (no record_masses flag) then unnormalized (record_masses flag)
    python $script_name  --seed=1 --dataset=$dataset --widthmult=2 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=2 --lr_mode=fix_b --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_b --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_b --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_b --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_b --lr=$lr
done

a_lrs=("0.00000611"  "0.00001389" "0.00003162" "0.00007197" "0.00016379" "0.00037276" "0.00084834" "0.00193070" "0.00439397" "0.00999996" "0.02275833" "0.05179436" "0.26826642")
for lr in $(printf "%s\n" "${a_lrs[@]}" | shuf); do
    # normalized (no record_masses flag) then unnormalized (record_masses flag)
    python $script_name  --seed=1 --dataset=$dataset --widthmult=2 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=2 --lr_mode=fix_a --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_a --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_a --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_a --lr=$lr
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_a --lr=$lr
done