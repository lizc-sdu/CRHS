# CRHS
This is a PyTorch implementation of the CRHS model and its corresponding data. Paper: Contrastive Region Representation Learning with Hard Positive and Negative Sampling.
# Data
We conduct our experiments on three datasets, including Beijing (BJ), Chengdu (CD), and Xi'an (XA). The pre-processed dataset can be found in the corresponding folder.
# Hyperparameters
All hyperparameter settings for the three datasets are saved in the configue.py file.
# Run
You can train CRHS using the following commands:
```
python main.py --city ${name}
```
You need to replace ${name} with 'bj', 'cd', or 'xa'.
