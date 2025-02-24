
script_name="train_scripts/llama3.py"
dataset="mathpile"

b_lrs=( "0.00000023"  "0.00000118"   "0.00000611"  "0.00001389"      "0.00007197"  "0.0001"    "0.00037276" "0.00084834" "0.00193070"  "0.00999996" "0.02275833"  "0.05179436")
for lr in $(printf "%s\n" "${b_lrs[@]}" | shuf); do
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_b --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_b --lr=$lr --record_masses
done

a_lrs=("0.00000023" "0.00000118"  "0.00000611" "0.00003162" "0.00016379" "0.00084834")
for lr in $(printf "%s\n" "${a_lrs[@]}" | shuf); do
    python $script_name  --seed=1 --dataset=$dataset --widthmult=4 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=8 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=16 --lr_mode=fix_a --lr=$lr --record_masses
    python $script_name  --seed=1 --dataset=$dataset --widthmult=32 --lr_mode=fix_a --lr=$lr --record_masses
done

