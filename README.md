### Function-Space Learning Rates Paper

Temporary repo for anonymous submission. Apologies for the mess. We are working on an easy-to-use library and clear instructions to replicate results.

Code for FLeRM / recording function-space learning rates is found in normalised_optimiser.py. LoRA experiments used a slightly modified version in lora/train_scripts/normalised_optimizer.py to handle LoRA parameter properly.

mlpcifar10/mlp_cifar10.py is the training script for the ResMLP, pytorch_transformer_example/claudemain_wrapper_load_masses.py is the training script for the transformer, adapted from the transformer example given in the pytorch github. Both of these show how the code in normalised_optimiser.py is used with a model to use FLeRM. lora/train_scripts/{gpt2.py,llama3.py} are the training scripts for the LoRA experiments.

Usual requirements of pytorch, huggingface transformers, matplotlib, numpy etc.