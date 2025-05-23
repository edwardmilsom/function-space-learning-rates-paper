### Function-Space Learning Rates

Code for the paper "Function-Space Learning Rates".

If you are interested in trying FLeRM / function-space learning rates yourself, ignore all files except `flerm.py` and `flerm_example_training_script.py`. `flerm.py` is the entire "library" for FLeRM, making it very easy to use. `flerm_example_training_script.py` is an example training script that shows how to use FLeRM in a very standard setting, training an MLP with skip connections on CIFAR-10. We have made an effort to present and comment both of these files nicely, so hopefully they are fairly self-explanatory. Note that depth scaling is more complicated than width scaling because you need to copy / split the base function-space learning rates up between the new layers in the deeper model, but hopefully the code should be clear.

Please cite our paper if it helps you in your work 🙂
```
@misc{milsom2025functionspacelearningrates,
      title={Function-Space Learning Rates}, 
      author={Edward Milsom and Ben Anson and Laurence Aitchison},
      year={2025},
      eprint={2502.17405},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2502.17405}, 
}
```


All other files are for reproducing the plots / experiments from the paper. `mlpcifar10/mlp_cifar10.py` is the training script for the ResMLP, `pytorch_transformer_example/claudemain_wrapper_load_masses.py` is the training script for the transformer, adapted from the transformer example given in the pytorch github. `lora/train_scripts/{gpt2.py,llama3.py}` are the training scripts for the LoRA experiments. There are various bash files that run different experiment settings, which are hopefully not too indecipherable. Included also are .txt files containing bash commands for organising the outputs of the experiments. In the experiment_results folder is a python file along with more .txt files that show how to process the experiment results into aggregate results files. These can then be plotted with the `plot_grid.py` or `plot_lora.py` files (`plot_one_dict.py` is also available for producing single subplots).

Usual requirements of pytorch, huggingface transformers, matplotlib, numpy etc.
