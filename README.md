# UIPA
This is the code respository of **UIPA**, which is a cross-platform recommendation framework based on user and item prototype alignment. 

The repository contains the following contents:

1. **Datasets**: This folder contains the dataset files necessary to run the code.
2. **Models**: This folder contains the models that have been trained.

## 1 Download
Due to the large size of the data files, we use Git LFS to store them. You can utilize the following script to download this repository.
```
# Because of the double-blind principle of submission, we are sorry that the complete data cannot be provided at this time.
# The data will be provided after the paper is accepted.
```

## 2 Setup
You can utilize the following script to install the required packages. The corresponding file `requirements.txt` is provided under the main directory.
```
# Create the virtualenv UIPA
conda create -n UIPA python=3.8

# Activate the virtualenv UIPA
conda activate UIPA

# Install requirements with pip
pip install -r requirements.txt
```

## 3 Quick Start
Please make sure the packages required in the `requirements.txt` are properly installed. Then you can utilize the following script to run the UIPA:
```
python main.py
```
If you want to skip the training part and use the trained model, you can utilize the following script:
```
python main.py --pretrained_model=1
```

## 4 Parameters setting
All the parameters along with their descriptions are in `parse.py`. You can also run UIPA with any combination of parameters you want.

```
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="input GPU id", default="0")
    parser.add_argument("-m", "--model", type=str, help="the backbones, select from: lightgcn, MF", default='lightgcn')
    parser.add_argument("-s", "--source_platform", type=str, help="the source platform, select from: DB, ML, MT, DB_Bo, BX_Bo", default="DB")
    parser.add_argument("-t", "--target_platform", type=str, help="the target platform, select from: DB, ML, MT, DB_Bo, BX_Bo", default="ML")
    parser.add_argument('--bpr_batch_size', type=int, help="bpr_batch_size", default=1024)
    parser.add_argument('-dim', '--latent_dim_rec', type=int, help="the dim of embeddings", default=64)
    parser.add_argument('--lightGCN_n_layers', type=int, help="lightGCN_n_layers", default=3)
    parser.add_argument('--lr', type=float, help="the learning rate", default=0.005)
    parser.add_argument('--seed', type=int, help="seed", default=2024)
    parser.add_argument('--epochs', type=int, help="the max epoch number", default=1000)
    parser.add_argument('--topks', type=list, help="topks", default=[10])
    parser.add_argument('--pretrained_model', type=int, help="whether to use pretrained_model", default=0)
    parser.add_argument('-pn','--prototype_num', type=int, help="the number of prototypes", default=100)
    parser.add_argument('-lw','--llm_weight', type=float, help="the weight of llm", default=0.3)
    parser.add_argument('-pw','--prototype_weight', type=float, help="the weight of prototype", default=0.1)
    parser.add_argument('-pl','--prototype_loss_weight', type=float, help="the weight of prototype_loss", default=1e-2)
    args = parser.parse_args()
    return args
```
