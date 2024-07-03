# pixel-flipping
This repository accompanies our paper on **Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks**. If you build on our research, please cite as follows 
```
@article{bluecher2024decoupling,
      title={Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks}, 
      author={Stefan Blücher and Johanna Vielhaben and Nils Strodthoff},
      journal={Transactions on Machine Learning Research},
      year={2024}
}
```

## Setup
First create a conda environment
```conda env create --file environment.yaml```

Then `pip` install the remaining local `src` code dependency
```
conda env create --file environment.yaml

pip install hydra-core --upgrade
pip install --upgrade pip  # make sure you have the up-to-data version of pip
pip install -e .
```

Use `scripts/results_pixel_flipping.ipynb` to reproduce all figures summarizing the PF benchmarks.

You can generate new PF experiments first calling `calculate_attributions.py` and then using `run_pixel_flipping.py`.
Each script has its own `conf/*.yaml` file. Your custom imagenet folder can be added at `src/config/helpers.py`.

