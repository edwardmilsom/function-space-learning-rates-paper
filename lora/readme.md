# LoRA adapter FLeRM experiments

You should be able to execute these experiments by running within the `lora` folder.
Make sure you have `pytorch`, `transformers`, `numpy`, `peft`, `datasets`, `tqdm`, `bitsandbytes` installed.

For datasets, `cold_french_law` should automatically download. For the mathpile
dataset, we provide a script that downloads the necessary files, and puts them
in `lora/data/` -- see `download_mathpile_data.sh`. This assumes you have
`huggingface-cli` installed, and that you have set up API keys.