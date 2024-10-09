# Uncertainty Quantification with the Empirical Neural Tangent Kernel
Code repository for experiments and figures in the main text of the paper "Uncertainty Quantification with the Empirical Neural Tangent Kernel".

## Setup
We use **Python** `3.11`.
```
pip install -r requirements.txt
```

## Image Classification
The image_classification.py script will save the predictions for each method in a `results/` folder. There is some incompatability between the `Laplace` package and our NUQLS method. Hence, for each dataset/model, 
you need to run the `image_classification.py` script once, and then again with the flag `--lla_incl`. Change the directory in `plotting_script.ipynb` to the respective files.

## Reference to Code
Adapted code for [LLA](https://github.com/AlexImmer/BNN-predictions) method and [SWAG](https://github.com/wjmaddox/swa_gaussian) method.


