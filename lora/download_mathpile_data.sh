### usage: 1. make sure you're in the `lora`` directory
###        2. $ bash download_mathpile_data.sh

## download dataset...
huggingface-cli download --repo-type dataset GAIR/MathPile --local-dir=./data

## we only actually use a subset of the data...
mv ./data/train/arXiv/math_arXiv_v0.2_chunk_1.jsonl.gz ./data
gzip -d ./data/math_arXiv_v0.2_chunk_1.jsonl.gz

## remove remaining unnecessary files for our exps
rm -rf ./data/{README.md,train,imgs,validation}
