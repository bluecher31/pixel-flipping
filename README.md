# pixel-flipping

First create a conda environment
```conda env create --file environment.yaml```

Then pip install the remaining local src code dependency
```
conda env create --file environment.yaml

pip install hydra-core --upgrade
pip install --upgrade pip  # make sure you have the up-to-data version of pip
pip install -e .
```

The jupyter notebook `results_pixel_flipping.ipynb` allows to reproduce all figures summarizing the PF benchmarks.

New PF experiments can be generated by first invoking `calculate_attributions.py` and then using `run_pixel_flipping.py`.
Each script has its own conf/*.yaml file. Your custom imagenet folder can be added at `src/config/helpers.py`.

