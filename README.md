# Uncertainty Quantification with the Empirical Neural Tangent Kernel
Code repository for experiments and figures in the main text of the paper ["Uncertainty Quantification with the Empirical Neural Tangent Kernel"](https://arxiv.org/pdf/2502.02870) .

## Setup
We use **Python** `3.11`.
```
pip install -r requirements.txt
```

## Image Classification
The image_classification.py script will save the predictions for each method in a `results/` folder. There is some incompatability between the `Laplace` package and our NUQLS method. Hence, for each dataset/model, 
you need to run the `image_classification.py` script once, and then again with the flag `--lla_incl`. Change the directory in `plotting_script.ipynb` to the respective files.

## Reference to Code
Please see paper for references to codebases for any implimented methods. 

